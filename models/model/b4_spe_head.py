"""Run8 B4-SPE head：B4 Semantic Pair Evidence + Sub-patch Expansion。

设计（CASA-CD_Run8_B4-SPE最终路线与预注册.md §5）：
  输入 fx=[Vx]、fy=[Vy]，V = frozen ViT4 final-LN B4 token（B×256×192）。
  单一 Symmetric Pair Descriptor（严格时间交换对称）：

      D = [ |T_A − T_B| , (T_A + T_B)/2 ]     # 384
      Z = Linear(384 → 96, bias)              # B×256×96 → 96×16×16

  然后 Sub-patch Expansion（复用 Run7 已 smoke 的四级拓扑，无 attention/gate/残差）：

      96×16×16 → Conv1×1 96→128 + BN + ReLU → PixelShuffle×2 → 32×32×32 → DW3×3+BN+ReLU
              → Conv1×1 32→64  → PS×2 → 64×64×16 → DW3×3
              → Conv1×1 16→64  → PS×2 → 128×128×16 → DW3×3
              → Conv1×1 16→16  → PS×2 → 256×256×4 → DW3×3 → Conv1×1 4→1 → sigmoid

参数：pair 36,960 + expansion 16,912 = **53,872**（head 全部 trainable）。
整体 = ViT4 1,976,832 + head 53,872 = **2,030,704**。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class B4SPEHead(nn.Module):
    """B4-SPE：Change-Sensitive B4 Semantic Source + Symmetric Pair Descriptor + Sub-patch Expansion。"""

    def __init__(self, token_dim=192, proj_dim=96):
        super().__init__()
        self.pair = nn.Linear(token_dim * 2, proj_dim, bias=True)

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

    def forward(self, fx, fy):
        vx = fx[0]                                    # (B,192,16,16)
        vy = fy[0]
        B, C, H, W = vx.shape
        tx = vx.flatten(2).transpose(1, 2)            # (B,256,192)
        ty = vy.flatten(2).transpose(1, 2)
        d = torch.cat([torch.abs(tx - ty), (tx + ty) / 2.0], dim=-1)   # (B,256,384)
        z = self.pair(d)                              # (B,256,96)
        x = z.transpose(1, 2).reshape(B, 96, 16, 16)

        x = self.dw1(F.pixel_shuffle(self.exp1(x), 2))      # 32×32×32
        x = self.dw2(F.pixel_shuffle(self.exp2(x), 2))      # 64×64×16
        x = self.dw3(F.pixel_shuffle(self.exp3(x), 2))      # 128×128×16
        x = self.dw4(F.pixel_shuffle(self.exp4(x), 2))      # 256×256×4
        return torch.sigmoid(self.out(x))
