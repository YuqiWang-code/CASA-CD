"""STRFusion Run1 encoder: frozen ViT4 depth-as-scale token pyramid.

Reuses the existing CASA-CD corrected DeiT loader WITHOUT touching encoder.py:
an internal `Encoder('tiny', vit_depth=4, detail_mode='none_b4', ...)` provides
the frozen depth-4 ViT (patch_embed in-place inherited, pos_embed 14x14->16x16
bicubic, blocks 4..11 absent). `forward` collects the four block states
B1..B4 (each passed through the final LN, same口径 as run6/run7 audits) and
resamples them to the four TAR/DCR scales with FIXED parameter-free ops:

    t4 (8x8)   <- B4  avgpool 2x2          (deepest, change-sensitive semantic)
    t3 (16x16) <- B3  native
    t2 (32x32) <- B2  bilinear x2
    t1 (64x64) <- B1  bilinear x4          (shallowest, detail-rich lateral)

All ViT parameters are frozen (CASA discipline: training ViT with the official
lr=2e-4 protocol collapses it to exact zero weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder


class STRViT4Encoder(nn.Module):
    def __init__(self, pretrained_path, vit_depth=4):
        super().__init__()
        self.vit_depth = vit_depth
        self.encoder = Encoder(
            "tiny", pretrained_path=pretrained_path, resnet_pretrained=False,
            mode="baseline", vit_depth=vit_depth, detail_mode="none_b4",
        )
        self.vit = self.encoder.vit
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # Explicit block loop (same口径 as run6 token_sources / run7 depth_pyramid):
        # raw state flows between blocks; each block output passes final LN for
        # observation. NOT vit.get_intermediate_layers() — that helper slices
        # out[:, 1:] assuming a cls/register token, but the corrected DeiT loader
        # strips the cls token, so it would silently drop one real patch token.
        p0 = self.vit.patch_embed(x)                                  # (B,256,192)
        tok = p0 + self.vit.interpolate_pos_encoding(p0, x.shape[-1], x.shape[-2])
        obs = []
        for i in range(self.vit_depth):
            tok = self.vit.blocks[i](tok)
            obs.append(self.vit.norm(tok))                            # Bk, (B,256,192)
        b1, b2, b3, b4 = [o.transpose(1, 2).reshape(o.shape[0], 192, 16, 16).contiguous()
                          for o in obs]                               # (B,192,16,16)
        t1 = F.interpolate(b1, scale_factor=4, mode="bilinear", align_corners=False)  # 64x64
        t2 = F.interpolate(b2, scale_factor=2, mode="bilinear", align_corners=False)  # 32x32
        t3 = b3                                                                       # 16x16
        t4 = F.avg_pool2d(b4, kernel_size=2, stride=2)                                # 8x8
        return [t1, t2, t3, t4]

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # frozen ViT stays a feature extractor
        return self
