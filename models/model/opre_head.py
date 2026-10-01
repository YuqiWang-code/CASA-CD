"""Run9 B4-OPRE head：Frozen B4 Semantic + Overlap Patch Re-Embedding + 语义门控。

设计（CASA-CD_Run9_可执行预注册方案.md §6）：
  输入 fx=[O_x, V_x]、fy=[O_y, V_y]：
    V：冻结 ViT4 final-LN B4 token（B×192×16×16）；
    O：O-PRE——共享 patch_embed.proj 权重、stride=8、reflect pad4 的
       B×192×32×32 局部重嵌入（encoder 侧产出，零新增 encoder 参数）。

  Z_s = Conv1×1(384→96) [ |V_A−V_B| , (V_A+V_B)/2 ]        # 对称 pair descriptor
  S32 = DW3×3( PixelShuffle2( Conv1×1 96→128 + BN + ReLU ) )   # 语义 32×32×32
  L32 = Conv1×1(192→32) + BN + ReLU ( |O_A−O_B| )              # 局部 evidence
  G32 = sigmoid( Conv1×1 32→1 (S32) )
  F32 = S32 + G32 ⊙ L32                                         # 语义门控 residual
  之后复用 Run8 已 smoke 的 32→64→128→256 PixelShuffle 重建 + 4→1 + sigmoid。

  严格时间交换对称（只依赖 |Δ| 与 mean）。无 attention / 无残差网络 / 无新 loss。

参数：pair 36,960 + 语义 expansion 12,896 + local_proj 6,208 + gate 33 +
     重建 3,984 + classifier 4 = **60,113**（全部 trainable）。
整体 = ViT4 1,976,832 + head 60,113 = **2,036,945**。
gate=False 时 F32 = S32 + L32（R9-A1 NOGATE 条件消融专用）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OPREHead(nn.Module):
    """B4-OPRE head：B4 语义 + O-PRE 局部 evidence 的语义门控融合重建。"""

    def __init__(self, gate=True):
        super().__init__()
        self.gate_enabled = gate
        # B4 对称 pair descriptor
        self.pair = nn.Conv2d(384, 96, 1, bias=True)
        # 语义 16→32 expansion
        self.exp16 = nn.Sequential(nn.Conv2d(96, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dw32 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, padding=1, groups=32, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # O-PRE 局部 evidence
        self.local_proj = nn.Sequential(nn.Conv2d(192, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.gate32 = nn.Conv2d(32, 1, 1, bias=True)
        # 32→64→128→256 重建
        self.exp32 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dw64 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, padding=1, groups=16, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.exp64 = nn.Sequential(nn.Conv2d(16, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dw128 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, padding=1, groups=16, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.exp128 = nn.Sequential(nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.dw256 = nn.Sequential(nn.Conv2d(4, 4, 3, 1, padding=1, groups=4, bias=False),
                                   nn.BatchNorm2d(4), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(4, 1, 1, bias=False)

    def forward(self, fx, fy):
        ox, vx = fx
        oy, vy = fy
        ds = torch.cat([torch.abs(vx - vy), (vx + vy) / 2.0], dim=1)     # (B,384,16,16)
        z = self.pair(ds)                                                # (B,96,16,16)
        s32 = self.dw32(F.pixel_shuffle(self.exp16(z), 2))               # (B,32,32,32)
        l32 = self.local_proj(torch.abs(ox - oy))                        # (B,32,32,32)
        if self.gate_enabled:
            g32 = torch.sigmoid(self.gate32(s32))                        # (B,1,32,32)
            f32 = s32 + g32 * l32
        else:
            f32 = s32 + l32                                             # NOGATE 消融
        x = self.dw64(F.pixel_shuffle(self.exp32(f32), 2))               # 64×64×16
        x = self.dw128(F.pixel_shuffle(self.exp64(x), 2))                # 128×128×16
        x = self.dw256(F.pixel_shuffle(self.exp128(x), 2))               # 256×256×4
        return torch.sigmoid(self.classifier(x))
