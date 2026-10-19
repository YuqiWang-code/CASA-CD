#!/usr/bin/env python3

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import math
import torch
working_path = os.path.abspath('.')

import functools

_orig_torch_load = torch.load


torch.load = functools.partial(_orig_torch_load, weights_only=False)


import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from timm.models.layers import DropPath, trunc_normal_
from models.cd_modules import Decoder
from einops import rearrange

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import sys 
sys.path.append("..") 
from utils import weight_init

DINOV2_SMALL_EMBED_DIM = 384
DINOV2_SMALL_INTERMEDIATE_LAYER_INDICES = [2, 5, 8, 11]

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True); s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]


class ALA(nn.Module):
    def __init__(self, channels, bottleneck_dim=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(channels * 4, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 4),
            nn.Softmax(dim=1)
        )
        self.apply(weight_init)
        
    def forward(self, features_list):
        f1, f2, f3, f4 = features_list
        B, C, H, W = f1.shape
        
        v1 = self.gap(f1).view(B, C)
        v2 = self.gap(f2).view(B, C)
        v3 = self.gap(f3).view(B, C)
        v4 = self.gap(f4).view(B, C)
        
        all_v = torch.cat([v1, v2, v3, v4], dim=1)
        
        att_scores = self.attention_mlp(all_v)
        
        fused = f1 * att_scores[:, 0].view(B, 1, 1, 1) + \
                f2 * att_scores[:, 1].view(B, 1, 1, 1) + \
                f3 * att_scores[:, 2].view(B, 1, 1, 1) + \
                f4 * att_scores[:, 3].view(B, 1, 1, 1)
        return fused


class HybridAttention(nn.Module):
    def __init__(self, dim, heads=2, reduction=4, dropout=0.1):
        super().__init__()
        actual_reduction = reduction; actual_heads = heads
        if dim <= 0: self.reduced_dim = 0; self.heads = 0
        else:
            if dim < actual_heads: actual_heads = max(1, dim)
            if actual_heads == 0: actual_heads = 1
            max_possible_reduction = dim // (actual_heads * 1) if actual_heads > 0 else dim
            actual_reduction = min(reduction, max_possible_reduction); actual_reduction = max(1, actual_reduction)
            self.reduced_dim = dim // actual_reduction; self.heads = actual_heads
            if self.heads > 0:
                if not (self.reduced_dim > 0 and self.reduced_dim % self.heads == 0):
                    self.reduced_dim = (self.reduced_dim // self.heads) * self.heads
                    if self.reduced_dim == 0 and dim > 0:
                        self.reduced_dim = self.heads
                        if dim // self.reduced_dim == 0: self.heads = 0; self.reduced_dim = 0
            else: self.reduced_dim = 0
        self.norm = nn.LayerNorm(dim)
        if self.reduced_dim > 0 and self.heads > 0:
            self.to_qkv = nn.Conv2d(dim, self.reduced_dim * 3, kernel_size=1, bias=False)
            self.attn = nn.MultiheadAttention(self.reduced_dim, self.heads, dropout=dropout, batch_first=True)
            self.project_out = nn.Conv2d(self.reduced_dim, dim, kernel_size=1, bias=False)
        else: self.to_qkv = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 0: return x
        x_flat = x.flatten(2).transpose(1, 2); x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        if not self.to_qkv: return x
        qkv = self.to_qkv(x_norm); qkv = qkv.reshape(B, 3, self.reduced_dim, H * W)
        q, k, v = qkv.unbind(dim=1)
        q = q.permute(0, 2, 1); k = k.permute(0, 2, 1); v = v.permute(0, 2, 1)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, self.reduced_dim, H, W)
        out = self.dropout(self.project_out(attn_out))
        return x + out


class DLM(nn.Module):
    def __init__(self,
                 input_dims=[64, 128, 256, 512],
                 internal_channels_reduction_factor=2,
                 kernel_size=3,
                 heads=[2, 2, 4, 4],
                 hybrid_attention_reduction_factor=2,
                 output_embedding_dim=64,
                 use_abs_diff=True,
                 use_hybrid_attention=True,
                 ablate_hybrid_attention_only=False):
        super().__init__()
        self.input_dims = input_dims
        self.use_abs_diff = use_abs_diff
        self.use_hybrid_attention = use_hybrid_attention
        self.ablate_hybrid_attention_only = ablate_hybrid_attention_only

        num_stages = len(input_dims)
        self.processing_dims = []

        for i in range(num_stages):
            raw_proc_dim = input_dims[i] // internal_channels_reduction_factor
            heads_i = heads[i] if i < len(heads) else heads[-1]
            if self.use_hybrid_attention or self.ablate_hybrid_attention_only:
                min_dim_for_heads = heads_i
            else:
                min_dim_for_heads = 1
            self.processing_dims.append(max(raw_proc_dim, min_dim_for_heads, 1))

        input_concat_factor = 3 if use_abs_diff else 2
        self.diff_conv_blocks = nn.ModuleList()
        for i in range(num_stages):
            block = nn.Sequential(
                nn.Conv2d(input_dims[i]*input_concat_factor, self.processing_dims[i], kernel_size,
                          padding=kernel_size//2, bias=False),
                LayerNorm(self.processing_dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv2d(self.processing_dims[i], self.processing_dims[i], kernel_size,
                          padding=kernel_size//2, bias=False),
            )
            self.diff_conv_blocks.append(block)

        if self.use_hybrid_attention and not self.ablate_hybrid_attention_only:
            self.attns = nn.ModuleList([
                HybridAttention(self.processing_dims[i],
                                heads=heads[i] if i < len(heads) else heads[-1],
                                reduction=hybrid_attention_reduction_factor)
                for i in range(num_stages)
            ])
        elif self.ablate_hybrid_attention_only:
            self.attns = nn.ModuleList([nn.Identity() for _ in range(num_stages)])
        else:
            self.attns = None

        self.post_process_norms = nn.ModuleList([
            LayerNorm(self.processing_dims[i], eps=1e-6, data_format="channels_first")
            for i in range(num_stages)
        ])
        self.channel_aligners = nn.ModuleList([
            nn.Conv2d(self.processing_dims[i], output_embedding_dim, kernel_size=1, bias=False)
            for i in range(num_stages)
        ])
        self.apply(weight_init)

    def forward(self, pre_feat_list, post_feat_list):
        if len(pre_feat_list) != len(post_feat_list) or len(pre_feat_list) != len(self.input_dims):
            raise ValueError("Input feature lists length mismatch.")
        outs = []
        for i in range(len(self.input_dims)):
            pre_f, post_f = pre_feat_list[i], post_feat_list[i]
            concat_input_feat = (torch.cat([pre_f, post_f, torch.abs(pre_f - post_f)], dim=1)
                                 if self.use_abs_diff else torch.cat([pre_f, post_f], dim=1))
            x = self.diff_conv_blocks[i](concat_input_feat)
            if self.attns is not None:
                x = self.attns[i](x)   
            x = self.post_process_norms[i](x)
            outs.append(self.channel_aligners[i](x))
        return tuple(outs)


from models.backbone.dinov2 import DINOv2

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EfficientResidualFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        mid_channels = out_channels
        self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(mid_channels) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.conv2 = DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels) if norm_layer else nn.Identity()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels) if norm_layer else nn.Identity()
            )
        self.final_act = act_layer()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.final_act(out)
        return out

class BSM(nn.Module):
    def __init__(
        self,
        res_dims=[256, 512, 1024, 2048],
        dino_guidance_channels=DINOV2_SMALL_EMBED_DIM,
        output_channel_reduction_factor=4,
        num_fusion_blocks=1,
        adapter_act_layer=nn.GELU,
        adapter_norm_layer=nn.BatchNorm2d,
        inject_alpha=1.0 
    ):
        super().__init__()
        self.inject_alpha = inject_alpha  
        self.res_dims = res_dims
        self.dino_guidance_channels = dino_guidance_channels
        self.output_channel_reduction_factor = output_channel_reduction_factor
        self.num_fusion_blocks = num_fusion_blocks
        self.num_resnet_stages = len(res_dims)
        self.dino_guidance_adapters = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()
        for i in range(self.num_resnet_stages):
            resnet_stage_channels = self.res_dims[i]
            current_fi_output_channels = max(1, resnet_stage_channels // self.output_channel_reduction_factor)
            adapter_output_channels_for_concat = resnet_stage_channels
            adapter_layers = [
                nn.Conv2d(self.dino_guidance_channels, adapter_output_channels_for_concat, kernel_size=1, bias=False),
            ]
            if adapter_norm_layer:
                if adapter_norm_layer == LayerNorm:
                    adapter_layers.append(adapter_norm_layer(adapter_output_channels_for_concat, data_format="channels_first"))
                else:
                    adapter_layers.append(adapter_norm_layer(adapter_output_channels_for_concat))
            if adapter_act_layer:
                adapter_layers.append(adapter_act_layer())
            self.dino_guidance_adapters.append(nn.Sequential(*adapter_layers))
            fusion_conv_input_channels = resnet_stage_channels + adapter_output_channels_for_concat
            fusion_blocks = []
            for block_idx in range(self.num_fusion_blocks):
                in_ch = fusion_conv_input_channels if block_idx == 0 else current_fi_output_channels
                out_ch = current_fi_output_channels
                fusion_blocks.append(
                    EfficientResidualFusionBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        norm_layer=adapter_norm_layer,
                        act_layer=adapter_act_layer
                    )
                )
            self.fusion_convs.append(nn.Sequential(*fusion_blocks))
        self.apply(weight_init)

    def forward(self, res_feats, dino_guidance_feat_single_stage):
        if len(res_feats) != self.num_resnet_stages:
            raise ValueError(f"Expected {self.num_resnet_stages} ResNet features, got {len(res_feats)}")
        enhanced_res_features = []
        for i in range(self.num_resnet_stages):
            res_feat_stage = res_feats[i]
            dino_adapted_spatial = F.interpolate(
                dino_guidance_feat_single_stage, size=res_feat_stage.shape[2:],
                mode='bilinear', align_corners=False)
            dino_processed_for_stage = self.dino_guidance_adapters[i](dino_adapted_spatial)
            dino_processed_for_stage = self.inject_alpha * dino_processed_for_stage # 抑制强度实验
            concat_feat = torch.cat([res_feat_stage, dino_processed_for_stage], dim=1)
            enhanced_feat = self.fusion_convs[i](concat_feat)
            enhanced_res_features.append(enhanced_feat)
        return enhanced_res_features

class GCCA(nn.Module):
    def __init__(self,
                 resnet_encoder_type='resnet50',
                 resnet_dims=[256, 512, 1024, 2048],
                 resnet_pretrained=False,
                 dino_encoder_size="small",
                 injector_output_channel_reduction_factor=4,
                 diff_encoder_internal_channels_reduction_factor=2,
                 diff_encoder_output_embedding_dim=128,
                 diff_encoder_heads=[2, 2, 4, 4],
                 inject_alpha=1.0,   
                 diff_encoder_use_hybrid_attention=True,
                 diff_encoder_hybrid_attention_reduction_factor=2,
                 freeze_dinov2=False,
                 freeze_resnet=False,
                 ablate_dino_fusion=False,
                 ablate_injector=False,
                 ablate_hybrid_attention_only=False,
                 ablate_difference_encoder=False, 
                 **kwargs):
        super().__init__()

        self.ablate_dino_fusion = ablate_dino_fusion
        self.ablate_injector = ablate_injector
        self.ablate_difference_encoder = ablate_difference_encoder 
        self.ablate_hybrid_attention_only = ablate_hybrid_attention_only
        self.dino_encoder_size = dino_encoder_size.lower()
        try:
            self.backbone = DINOv2(model_name=self.dino_encoder_size)
        except:
            self.backbone = DINOv2(model_name=f'dinov2_vit{self.dino_encoder_size[0]}14')

        self.dino_intermediate_indices_to_extract = DINOV2_SMALL_INTERMEDIATE_LAYER_INDICES
        dino_guidance_channels_for_injector = DINOV2_SMALL_EMBED_DIM
        
        self.dino_fusion_module = ALA(
            channels=dino_guidance_channels_for_injector,
            bottleneck_dim=16
        )

        self.visual_encoder_res = create_model(
            resnet_encoder_type, pretrained=resnet_pretrained,
            features_only=True, out_indices=(1, 2, 3, 4))

        self.resnet_dims = list(self.visual_encoder_res.feature_info.channels())  


        diff_encoder_actual_input_dims = [
            max(1, d // injector_output_channel_reduction_factor) for d in self.resnet_dims
        ]


        if self.ablate_injector:
            self.injector = None
            self.res_to_diff_adapters = nn.ModuleList([
                nn.Conv2d(d, diff_encoder_actual_input_dims[i], kernel_size=1)
                for i, d in enumerate(self.resnet_dims)
            ])
        else:
            self.injector = BSM(
                res_dims=self.resnet_dims,                        
                dino_guidance_channels=DINOV2_SMALL_EMBED_DIM,          
                output_channel_reduction_factor=injector_output_channel_reduction_factor,
                inject_alpha=inject_alpha   
            )
            self.res_to_diff_adapters = None

        if self.ablate_difference_encoder:
            self.difference_encoder = None
            
            self.simple_diff_processors = nn.ModuleList([
                nn.Conv2d(in_channels, diff_encoder_output_embedding_dim, kernel_size=1)
                for in_channels in diff_encoder_actual_input_dims
            ])
            
        else:
            self.difference_encoder = DLM(
                input_dims=diff_encoder_actual_input_dims,
                internal_channels_reduction_factor=diff_encoder_internal_channels_reduction_factor,
                output_embedding_dim=diff_encoder_output_embedding_dim,
                heads=diff_encoder_heads,
                use_hybrid_attention=diff_encoder_use_hybrid_attention,
                hybrid_attention_reduction_factor=diff_encoder_hybrid_attention_reduction_factor
            )
            self.simple_diff_processors = None

        self.decoder = Decoder(embedding_dim=diff_encoder_output_embedding_dim)

        self.apply(weight_init)
        if freeze_dinov2:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_resnet:
            for param in self.visual_encoder_res.parameters():
                param.requires_grad = False
                
        if (not self.ablate_difference_encoder) and self.ablate_hybrid_attention_only:
            self._ablate_hybridattention_inplace()
    
    def set_inject_alpha(self, alpha):
        if self.injector is not None:
            self.injector.inject_alpha = alpha
    
    def _ablate_hybridattention_inplace(self):
        root = self.module if isinstance(self, nn.DataParallel) else self
        name2module = dict(root.named_modules())
        for name, mod in list(name2module.items()):
            if mod is root: 
                continue
            if mod.__class__.__name__ == "HybridAttention":
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = name2module[parent_name] if parent_name != "" else root
                setattr(parent, child_name, nn.Identity())
                
    def forward_visual_features_dino_fused(self, x):
        features_list = self.backbone.get_intermediate_layers(
            x, self.dino_intermediate_indices_to_extract,
            return_class_token=False, reshape=True)
        
        if self.ablate_dino_fusion:
            fused_feature = features_list[-1]
        else:
            fused_feature = self.dino_fusion_module(features_list)
            
        return fused_feature

    def forward_visual_features_resnet(self, x):
        return self.visual_encoder_res(x)

    def forward(self, pre_img, post_img, **kwargs):
        output_size = kwargs.get('output_size', (pre_img.shape[2], pre_img.shape[3]))
        
        pre_res_list = self.forward_visual_features_resnet(pre_img)
        post_res_list = self.forward_visual_features_resnet(post_img)
        
        if self.ablate_injector:
            inj_pre_list = [adapter(feat) for adapter, feat in zip(self.res_to_diff_adapters, pre_res_list)]
            inj_post_list = [adapter(feat) for adapter, feat in zip(self.res_to_diff_adapters, post_res_list)]
        else:
            dino_g_pre = self.forward_visual_features_dino_fused(pre_img)
            dino_g_post = self.forward_visual_features_dino_fused(post_img)
            
            inj_pre_list = self.injector(pre_res_list, dino_g_pre) 
            inj_post_list = self.injector(post_res_list, dino_g_post)

        if self.ablate_difference_encoder:
            diff_features = []
            for i in range(len(inj_pre_list)):
                simple_diff = torch.abs(inj_post_list[i] - inj_pre_list[i])
                processed_diff = self.simple_diff_processors[i](simple_diff)
                diff_features.append(processed_diff)
            
            d1, d2, d3, d4 = diff_features
        else:
            d1, d2, d3, d4 = self.difference_encoder(inj_pre_list, inj_post_list)
        
        # d1, d2, d3, d4 = self.difference_encoder(inj_pre_list, inj_post_list)
        preds = self.decoder(d1, d2, d3, d4)
        
        if not isinstance(preds, (list, tuple)): preds = [preds]
        interpolated_preds = [F.interpolate(p, size=output_size, mode='bilinear', align_corners=False) for p in preds]

        if len(interpolated_preds) == 1:
            return interpolated_preds[0], interpolated_preds[0], interpolated_preds[0], interpolated_preds[0]
        elif len(interpolated_preds) == 5:
            return interpolated_preds[0], interpolated_preds[1], interpolated_preds[2], interpolated_preds[3], interpolated_preds[4]
        else:
            if interpolated_preds: return interpolated_preds[0] if len(interpolated_preds) == 1 else tuple(interpolated_preds)
            return None


if __name__ == "__main__":
    print("GCCA")