import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math

from utils import weight_init

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, output_nc=2, upsample_kernel_size=4, upsample_stride=2, upsample_padding=1):
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        class ConvLayer(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
                super().__init__(); self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
            def forward(self, x): return self.conv2d(x)
        
        class UpsampleConvLayer(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                super().__init__(); self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=upsample_padding, output_padding=0)
            def forward(self, x): return self.conv2d(x)

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__(); self.conv1 = ConvLayer(channels, channels, 3, 1, 1)
                self.conv2 = ConvLayer(channels, channels, 3, 1, 1); self.GELU = nn.GELU()
            def forward(self, x):
                residual = x; out = self.GELU(self.conv1(x)); out = self.conv2(out)
                return self.GELU(out + residual)

        self.process_p5 = ResidualBlock(embedding_dim)
        self.process_p4 = ResidualBlock(embedding_dim)
        self.process_p3 = ResidualBlock(embedding_dim)
        self.process_p2 = ResidualBlock(embedding_dim)
        
        self.upsample_p5_to_p4 = UpsampleConvLayer(embedding_dim, embedding_dim, upsample_kernel_size, upsample_stride)
        self.upsample_p4_to_p3 = UpsampleConvLayer(embedding_dim, embedding_dim, upsample_kernel_size, upsample_stride)
        self.upsample_p3_to_p2 = UpsampleConvLayer(embedding_dim, embedding_dim, upsample_kernel_size, upsample_stride)
        
        self.cls_p5 = ConvLayer(embedding_dim, output_nc, 3, 1, 1)
        self.cls_p4 = ConvLayer(embedding_dim, output_nc, 3, 1, 1)
        self.cls_p3 = ConvLayer(embedding_dim, output_nc, 3, 1, 1)
        
        self.cls_small = ConvLayer(embedding_dim, output_nc, 3, 1, 1)  

        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU()
        )
        self.final_upsample = UpsampleConvLayer(embedding_dim, embedding_dim // 2, upsample_kernel_size, upsample_stride)
        self.final_cls = ConvLayer(embedding_dim // 2, output_nc, 3, 1, 1)

        self.apply(weight_init)

    def _ensure_spatial_match(self, feat_to_upsample, target_feat):
        if feat_to_upsample.shape[2:] != target_feat.shape[2:]:
            return F.interpolate(feat_to_upsample, size=target_feat.shape[2:], mode='bilinear', align_corners=False)
        return feat_to_upsample

    def forward(self, d2, d3, d4, d5):
        # FPN Top-Down Path
        p5_processed = self.process_p5(d5)
        up_p5 = self._ensure_spatial_match(self.upsample_p5_to_p4(p5_processed), d4)
        p4_processed = self.process_p4(d4 + up_p5)
        up_p4 = self._ensure_spatial_match(self.upsample_p4_to_p3(p4_processed), d3)
        p3_processed = self.process_p3(d3 + up_p4)
        up_p3 = self._ensure_spatial_match(self.upsample_p3_to_p2(p3_processed), d2)
        p2_processed = self.process_p2(d2 + up_p3)
        
        mask_p5 = self.cls_p5(p5_processed)
        mask_p4 = self.cls_p4(p4_processed)
        mask_p3 = self.cls_p3(p3_processed)
        
        small_mask = self.cls_small(p2_processed)

        p5_resized = F.interpolate(p5_processed, size=p2_processed.shape[2:], mode='bilinear', align_corners=False)
        p4_resized = F.interpolate(p4_processed, size=p2_processed.shape[2:], mode='bilinear', align_corners=False)
        p3_resized = F.interpolate(p3_processed, size=p2_processed.shape[2:], mode='bilinear', align_corners=False)
        
        final_fused = self.final_fusion_conv(torch.cat([p2_processed, p3_resized, p4_resized, p5_resized], dim=1))
        final_upsampled = self.final_upsample(final_fused)
        main_mask = self.final_cls(final_upsampled) 
        
        return main_mask, small_mask, mask_p3, mask_p4, mask_p5