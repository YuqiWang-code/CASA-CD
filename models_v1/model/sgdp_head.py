"""Run5 模块：SGDP —— Semantic-Guided Difference Pyramid（语义门控差分金字塔）。

设计（CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md §15-26）：
  先把两个时相变成多尺度「变化证据」（|V1-V2|、|D8_1-D8_2|、|D4_1-D4_2|、
  |D2_1-D2_2|），再由 coarse semantic change 逐级 gate local detail difference，
  coarse-to-fine 重建 256×256 概率图。T1/T2 交换严格对称（只依赖差值）。

输入（encoder 输出 fx, fy）：
  d2: B×16×128×128, d4: B×16×64×64, d8: B×24×32×32, v: B×192×16×16

逐层（方案 §18-22）：
  Stage16: ProjBlock(192→160)                          → H16 160×16×16
  Stage8:  bilinear×2 + SepBlock(160→128)              → S8
           ProjBlock(24→128)                           → L8
           gate8 = sigmoid(Conv1×1 128→1)              → g8
           H8 = SepBlock(S8 + g8*L8)                   → 128×32×32
  Stage4:  bilinear×2 + SepBlock(128→96) + gate(96→1) + fuse Sep(96→96)   → 96×64×64
  Stage2:  bilinear×2 + SepBlock(96→64) + gate(64→1) + fuse Sep(64→64)    → 64×128×128
  Final:   bilinear×2 (128→256) + Conv3×3 64→1 + sigmoid                   → 1×256×256

参数合计 115,267（方案 §23）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjBlock(nn.Module):
    """Conv1×1 → BN → ReLU。"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SepBlock(nn.Module):
    """DWConv3×3 s1 → BN → ReLU → PWConv1×1 → BN → ReLU。"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


class SGDPHead(nn.Module):
    """语义门控差分金字塔：difference-first + semantic gate + coarse-to-fine。"""

    def __init__(self):
        super().__init__()
        # Stage 16
        self.sem16 = ProjBlock(192, 160)
        # Stage 8
        self.up8 = SepBlock(160, 128)
        self.detail8 = ProjBlock(24, 128)
        self.gate8 = nn.Conv2d(128, 1, 1, bias=True)
        self.fuse8 = SepBlock(128, 128)
        # Stage 4
        self.up4 = SepBlock(128, 96)
        self.detail4 = ProjBlock(16, 96)
        self.gate4 = nn.Conv2d(96, 1, 1, bias=True)
        self.fuse4 = SepBlock(96, 96)
        # Stage 2
        self.up2 = SepBlock(96, 64)
        self.detail2 = ProjBlock(16, 64)
        self.gate2 = nn.Conv2d(64, 1, 1, bias=True)
        self.fuse2 = SepBlock(64, 64)
        # Final
        self.classifier = nn.Conv2d(64, 1, 3, 1, padding=1, bias=False)

    def forward(self, fx, fy):
        d2x, d4x, d8x, vx = fx
        d2y, d4y, d8y, vy = fy

        dv = torch.abs(vx - vy)
        dd8 = torch.abs(d8x - d8y)
        dd4 = torch.abs(d4x - d4y)
        dd2 = torch.abs(d2x - d2y)

        h16 = self.sem16(dv)

        s8 = F.interpolate(h16, scale_factor=2, mode="bilinear", align_corners=False)
        s8 = self.up8(s8)
        l8 = self.detail8(dd8)
        g8 = torch.sigmoid(self.gate8(s8))
        h8 = self.fuse8(s8 + g8 * l8)

        s4 = F.interpolate(h8, scale_factor=2, mode="bilinear", align_corners=False)
        s4 = self.up4(s4)
        l4 = self.detail4(dd4)
        g4 = torch.sigmoid(self.gate4(s4))
        h4 = self.fuse4(s4 + g4 * l4)

        s2 = F.interpolate(h4, scale_factor=2, mode="bilinear", align_corners=False)
        s2 = self.up2(s2)
        l2 = self.detail2(dd2)
        g2 = torch.sigmoid(self.gate2(s2))
        h2 = self.fuse2(s2 + g2 * l2)

        out = F.interpolate(h2, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.classifier(out)
        return torch.sigmoid(out)
