"""Run11 TASS: Task-Adaptive Spatial Stem（任务适配空间残差 Stem）。

预注册方案 §4.2/§4.3（CASA-CD_下一步方案_Run11_TASS_设计与预注册.md）：
共享 Siamese 权重的极小可训练局部空间 stem，从像素网格产生 1/4、1/8、1/16
三尺度 task-adaptive spatial residual；由 STRTASSNet 以 zero-init scalar alpha
残差注入到 B1↑4 / B2↑2 / B3 上。

结构（widths 固定 (32,64,128,192)，blocks (2,2,2)，expansion 2）：
    Stem0 : Conv3x3 s2 3->32, BN, SiLU                  -> 128x128（仅内部状态）
    Stage1: Conv3x3 s2 32->64, BN+SiLU, LocalMix(64)x2  -> S1 64x64
    Stage2: Conv3x3 s2 64->128, BN+SiLU, LocalMix(128)x2 -> S2 32x32
    Stage3: Conv3x3 s2 128->192, BN+SiLU, LocalMix(192)x2 -> S3 16x16
    P1/P2/P3: 1x1 -> 192 + BN（projectors）

约束（Run4/Run5 复盘）：下采样用 dense 3x3 stride-2（不用 DW-first downsampling）；
stage 内 LocalMixBlock 全 residual（防深层表示塌缩）；不含 attention/loss/label；
不加载第二套 ImageNet 预训练。

RNG 纪律：TASS 由 STRTASSNet 在所有共享模块（encoder/head/TAR/DCR）构造完成之后
创建，因此 C0/M1 的共享模块初始化逐位一致（smoke 断言 0/N shared keys differ）。
"""
import torch
import torch.nn as nn


class LocalMixBlock(nn.Module):
    """RepViT 风格的轻量残差局部混合：u = x + BN(DW3x3(x)); y = u + BN(PW2(SiLU(PW1(u))))."""

    def __init__(self, c, expansion=2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, c * expansion, 1, bias=False)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(c * expansion, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        u = x + self.bn1(self.dw(x))
        y = u + self.bn2(self.pw2(self.act(self.pw1(u))))
        return y


class TASSStem(nn.Module):
    WIDTHS = (32, 64, 128, 192)
    BLOCKS = (2, 2, 2)
    EXPANSION = 2

    def __init__(self, in_ch=3, project_to=192):
        super().__init__()
        self.project_to = project_to
        w0, w1, w2, w3 = self.WIDTHS
        b1, b2, b3 = self.BLOCKS

        self.stem0 = nn.Sequential(
            nn.Conv2d(in_ch, w0, 3, 2, 1, bias=False), nn.BatchNorm2d(w0), nn.SiLU())
        self.down1 = nn.Sequential(
            nn.Conv2d(w0, w1, 3, 2, 1, bias=False), nn.BatchNorm2d(w1), nn.SiLU())
        self.s1 = nn.Sequential(*[LocalMixBlock(w1, self.EXPANSION) for _ in range(b1)])
        self.down2 = nn.Sequential(
            nn.Conv2d(w1, w2, 3, 2, 1, bias=False), nn.BatchNorm2d(w2), nn.SiLU())
        self.s2 = nn.Sequential(*[LocalMixBlock(w2, self.EXPANSION) for _ in range(b2)])
        self.down3 = nn.Sequential(
            nn.Conv2d(w2, w3, 3, 2, 1, bias=False), nn.BatchNorm2d(w3), nn.SiLU())
        self.s3 = nn.Sequential(*[LocalMixBlock(w3, self.EXPANSION) for _ in range(b3)])

        self.p1 = nn.Sequential(nn.Conv2d(w1, project_to, 1, bias=False), nn.BatchNorm2d(project_to))
        self.p2 = nn.Sequential(nn.Conv2d(w2, project_to, 1, bias=False), nn.BatchNorm2d(project_to))
        self.p3 = nn.Sequential(nn.Conv2d(w3, project_to, 1, bias=False), nn.BatchNorm2d(project_to))

    def forward(self, x):
        x = self.stem0(x)                       # 128x128
        x = self.down1(x)
        s1 = self.s1(x)                         # 64x64
        x = self.down2(s1)
        s2 = self.s2(x)                         # 32x32
        x = self.down3(s2)
        s3 = self.s3(x)                         # 16x16
        return [self.p1(s1), self.p2(s2), self.p3(s3)]   # [192@64x64, 192@32x32, 192@16x16]

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
