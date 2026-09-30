"""Run4 模块 B：Ultra-Light Multi-Scale Detail Branch（LightDetail 32/64/128）。

设计（决策文档 CASA-CD_Run4_极轻量结构主线_可执行方案.md §6）：
  - 高分辨率路径只负责局部层级细节，全局关系交给 16×16 TinyViT4；
  - 保留 1/2、1/4、1/8 三尺度（ChangeViT 论文支持多尺度 detail 联合最优）；
  - 不再使用 2.7M 的 ResNet C2-C4，改用 depthwise-separable CNN（0.037M）；
  - T1/T2 共享权重（Siamese）。

结构（输入 3×256×256）：
    Stem:      Conv3×3 s2 3→32 + BN + ReLU          → 32×128×128
    Stage D2:  DSConv 32→32  s1                     → D2: 32×128×128 (1/2)
    Stage D4:  DSConv 32→64  s2; DSConv 64→64 s1    → D4: 64×64×64   (1/4)
    Stage D8:  DSConv 64→128 s2; DSConv 128→128 s1  → D8: 128×32×32  (1/8)

DSConv = DWConv3×3 → BN → ReLU → PWConv1×1 → BN → ReLU。
参数合计 37,024（stem 928 + 各 stage 见决策文档 §6.3）。
"""
import torch.nn as nn


class DSConv(nn.Module):
    """Depthwise-separable conv block：DW 3×3 → BN → ReLU → PW 1×1 → BN → ReLU."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1,
                            groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.dw(x)))
        x = self.relu2(self.bn2(self.pw(x)))
        return x


class LightDetail(nn.Module):
    """三尺度极轻量 detail branch（1/2、1/4、1/8），T1/T2 共享权重。

    widths=(c1, c2, c3) 为三尺度通道：
      - (32, 64, 128) 默认（0.037M）；
      - (48, 96, 160) 预注册容量 fallback（0.065M，决策文档 §57）。
    """

    def __init__(self, widths=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        self.widths = tuple(widths)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = DSConv(c1, c1, stride=1)        # D2: c1×128×128
        self.stage4a = DSConv(c1, c2, stride=2)
        self.stage4b = DSConv(c2, c2, stride=1)       # D4: c2×64×64
        self.stage8a = DSConv(c2, c3, stride=2)
        self.stage8b = DSConv(c3, c3, stride=1)       # D8: c3×32×32

    def forward(self, x):
        x = self.stem(x)
        d2 = self.stage2(x)
        x4 = self.stage4a(d2)
        d4 = self.stage4b(x4)
        x8 = self.stage8a(d4)
        d8 = self.stage8b(x8)
        return d2, d4, d8
