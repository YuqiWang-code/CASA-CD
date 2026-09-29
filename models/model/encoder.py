# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange

from model.layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    CASAAAttention,
    NestedTensorBlock as Block,
)
from model.resnet import resnet18


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        casaa_layers=None,
        casaa_keep_ratio=0.25,
        casaa_change_share=0.5,
        casaa_router='change',
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        # CASAA（Run1）：指定 block 的 attention 换成变化感知非对称注意力。
        # qkv/proj 参数名与形状不变，DeiT-Tiny 预训练权重原位继承，零新增参数。
        self.casaa_layers = list(casaa_layers) if casaa_layers else []
        self.casaa_keep_ratio = casaa_keep_ratio
        self.casaa_change_share = casaa_change_share
        self.casaa_router = casaa_router

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            print("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            print("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            print("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = []
        for i in range(depth):
            if i in self.casaa_layers:
                attn_cls = partial(
                    CASAAAttention,
                    keep_ratio=casaa_keep_ratio,
                    change_share=casaa_change_share,
                    router=casaa_router,
                )
            else:
                attn_cls = MemEffAttention
            blocks_list.append(
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    init_values=init_values,
                    attn_class=attn_cls,
                )
            )
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return x_norm

    def forward_pair(self, x1, x2, score_hint=None):
        """双时相 paired forward（CASAA）。

        非 CASAA 层两个时相独立过 block；CASAA 层两时相共享变化感知路由
        一起过 block（Q 完整、K/V 压缩）。无 CASAA 层时与 `forward(x1);
        forward(x2)` 完全等价（eval 下 bitwise 一致）。
        score_hint: (B, N) 外部变化 score（仅 router='oracle' 诊断使用）。
        """
        x1 = self.prepare_tokens_with_masks(x1)
        x2 = self.prepare_tokens_with_masks(x2)

        for i, blk in enumerate(self.blocks):
            if i in self.casaa_layers:
                x1, x2 = blk.forward_pair(x1, x2, score_hint=score_hint)
            else:
                x1 = blk(x1)
                x2 = blk(x2)

        return self.norm(x1), self.norm(x2)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Encoder(nn.Module):
    def __init__(self, model_type='small', pretrained_path=None, resnet_pretrained=True,
                 mode='baseline', casaa_layers=None, casaa_keep_ratio=0.25,
                 casaa_change_share=0.5, casaa_router='change'):
        super().__init__()
        self.mode = mode
        if mode in ('casaa', 'saa'):
            if casaa_layers is None:
                casaa_layers = [8, 9, 10, 11]
            # mode='saa' 是 A1 SAA-style 内容对照：强制 content router（无变化感知保留）
            router = 'content' if mode == 'saa' else casaa_router
            self.casaa_layers = list(casaa_layers)
        else:
            router = casaa_router
            self.casaa_layers = []
        self.casaa_keep_ratio = casaa_keep_ratio
        self.casaa_change_share = casaa_change_share
        self.casaa_router = router

        if model_type == 'tiny':
            self.vit = DinoVisionTransformer(
                img_size=256,
                patch_size=16,
                embed_dim=192,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                block_fn=partial(Block, attn_class=MemEffAttention),
                num_register_tokens=0,
                casaa_layers=self.casaa_layers,
                casaa_keep_ratio=casaa_keep_ratio,
                casaa_change_share=casaa_change_share,
                casaa_router=router,
            )
            if pretrained_path is None:
                pretrained_path = "checkpoint/deit_tiny_patch16_224-a1311bcf.pth"

        elif model_type == 'small':
            self.vit = DinoVisionTransformer(
                img_size=256,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                block_fn=partial(Block, attn_class=MemEffAttention),
                num_register_tokens=0,
                casaa_layers=self.casaa_layers,
                casaa_keep_ratio=casaa_keep_ratio,
                casaa_change_share=casaa_change_share,
                casaa_router=router,
            )
            if pretrained_path is None:
                pretrained_path = "checkpoint/dinov2_vits14_pretrain.pth"
            
        else:
            assert False, r'Encoder: check the vit model type'

        state_dict = torch.load(pretrained_path, map_location='cpu')['model'] \
            if model_type == 'tiny' else torch.load(pretrained_path, map_location='cpu')

        for k in ['pos_embed', 'patch_embed.proj.weight']:
            del state_dict[k]
        msg = self.vit.load_state_dict(state_dict, strict=False)
        print(' missing_keys:{},\n unexpected_keys:{}'.format(msg.missing_keys, msg.unexpected_keys))
        print('model_type: {},\n checkpoint_path: {}'.format(model_type, pretrained_path))

        self.resnet = resnet18(pretrained=resnet_pretrained)
        self.drop = nn.Dropout(p=0.01)


    def detail_capture(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x2 = self.drop(self.resnet.layer1(x))
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)

        return [x2, x3, x4]


    def label_to_occupancy(self, label):
        """把 (B,1,H,W) 二值 label 转成与 ViT token 对齐的连续 patch occupancy。

        avg_pool(kernel=patch_size) -> (B,1,16,16) -> (B,256)。
        仅 router='oracle'（DIAGNOSTIC-ONLY）使用。
        """
        g = F.avg_pool2d(label.float(), kernel_size=self.vit.patch_size)
        return g.view(label.shape[0], -1)


    def forward(self, x, y, label=None):
        if self.casaa_layers:
            score_hint = None
            if self.casaa_router == 'oracle':
                # DIAGNOSTIC-ONLY：GT patch occupancy 作为变化 score 上界
                assert label is not None, "oracle router requires the GT label"
                score_hint = self.label_to_occupancy(label)
            v_x, v_y = self.vit.forward_pair(x, y, score_hint=score_hint)
        else:
            v_x = self.vit(x)
            v_y = self.vit(y)

        v_x = rearrange(v_x, 'b (h w) c -> b c h w', h=16, w=16)
        v_y = rearrange(v_y, 'b (h w) c -> b c h w', h=16, w=16)

        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        return c_x + [v_x], c_y + [v_y]
