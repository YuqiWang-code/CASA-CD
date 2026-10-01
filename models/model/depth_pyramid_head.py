"""Run7 CSDP head：Cross-Depth Symmetric Difference + Sub-patch Expansion。

设计（CASA-CD_Run7_CSDP方案与预注册.md §4）：
  输入 fx = [B1_x, B2_x]、fy = [B1_y, B2_y]（DeiT-Tiny block1/2 输出过 final LN，
  各 B×256×192）。对每个深度 l∈{1,2} 构造严格时间交换对称的 pair descriptor：

      D_l = P_l([ |T_A^l − T_B^l| , (T_A^l + T_B^l)/2 ])     # 384 → 96
      Z   = D_1 + D_2                                        # B×256×96 → 96×16×16

  然后 Sub-patch Expansion（每级：Conv1×1 扩张 + BN + ReLU → PixelShuffle×2 →
  DWConv3×3 + BN + ReLU）：
      96×16×16 → 32×32×32 → 64×64×16 → 128×128×16 → 256×256×4 → 1×1 → sigmoid

  无独立 detail encoder、无 attention、无 residual、T1/T2 共享权重；
  head 只依赖 [|Δ|, mean] → pred(A,B) == pred(B,A)。

参数（设计值）：P1/P2 各 36,960（Linear(384→96, bias)）；expansion 各 stage 合计
16,912；**head 总计 90,832**。整体 = ViT2 1,087,104 + head 90,832 = 1,177,936。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPyramidHead(nn.Module):
    """CSDP head（Semantic-Source Depth-Pyramid 变化重建）。"""

    def __init__(self, token_dim=192, proj_dim=96):
        super().__init__()
        self.p1 = nn.Linear(token_dim * 2, proj_dim, bias=True)
        self.p2 = nn.Linear(token_dim * 2, proj_dim, bias=True)

        # Sub-patch expansion：96×16×16 → 256×256
        self.exp1 = nn.Sequential(nn.Conv2d(96, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dw1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, padding=1, groups=32, bias=False),
                                 nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.exp2 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dw2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, padding=1, groups=16, bias=False),
                                 nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.exp3 = nn.Sequential(nn.Conv2d(16, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dw3 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, padding=1, groups=16, bias=False),
                                 nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.exp4 = nn.Sequential(nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.dw4 = nn.Sequential(nn.Conv2d(4, 4, 3, 1, padding=1, groups=4, bias=False),
                                 nn.BatchNorm2d(4), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(4, 1, 1, bias=False)

    def _pair_descriptor(self, ta, tb, proj):
        d = torch.cat([torch.abs(ta - tb), (ta + tb) / 2.0], dim=-1)   # (B,256,384)
        return proj(d)                                                  # (B,256,96)

    def forward(self, fx, fy):
        b1x, b2x = fx
        b1y, b2y = fy
        z = self._pair_descriptor(b1x, b1y, self.p1) + self._pair_descriptor(b2x, b2y, self.p2)
        B, N, C = z.shape
        x = z.transpose(1, 2).reshape(B, C, 16, 16)

        x = self.dw1(F.pixel_shuffle(self.exp1(x), 2))      # 32×32×32
        x = self.dw2(F.pixel_shuffle(self.exp2(x), 2))      # 64×64×16
        x = self.dw3(F.pixel_shuffle(self.exp3(x), 2))      # 128×128×16
        x = self.dw4(F.pixel_shuffle(self.exp4(x), 2))      # 256×256×4
        return torch.sigmoid(self.out(x))
