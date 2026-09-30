"""Run4 R4-2d：PSD-Detail —— Pretrained Stem + Residual Depthwise Detail Pyramid。

设计（CASA-CD_Run4_R4-2失败后_下一步决策与PSD-Detail方案.md §8）：
  - Stage 0：直接复用 ImageNet ResNet18 的 conv1 7×7 s2 3→64 + bn1（原位复制权重，
    不注册整个 ResNet18）；输出 64×128×128。
  - Stage D2：ResidualDS-64（DW3×3 → BN → ReLU → PW1×1 → BN → +identity → ReLU）
    → D2 = 64×128×128 (1/2)。
  - Stage D4：MixDown 64→128（先 PW 1×1 跨通道混合，再 DW 3×3 stride2 空间降采样）
    + ResidualDS-128 → D4 = 128×64×64 (1/4)。
  - Stage D8：MixDown 128→256（不再加 residual block）→ D8 = 256×32×32 (1/8)。
  - 输出 [64, 128, 256] 与 FeatureInjector 直接兼容，不需要 adapters。
  - T1/T2 共享权重（Siamese）。

参数合计 78,464（决策文档 §9）。与当前 LightDetail 的关键差异：
  - pretrained stem（不再随机初始化）；
  - stride-2 时先 PW 跨通道混合、再 DW 降采样（先组合、后降采样）；
  - 1/2、1/4 两级有 residual identity（细节保真）。
"""
import torch
import torch.nn as nn

from model.resnet import resnet18


class ResidualDS(nn.Module):
    """Residual depthwise-separable block：DW 3×3 → BN → ReLU → PW 1×1 → BN → +x → ReLU."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return self.relu(x + y)


class MixDown(nn.Module):
    """stride-2 降采样：先 PW 1×1 跨通道混合，再 DW 3×3 stride2 空间降采样。

    （LightDetail 的 DSConv 是 DW stride2 先逐通道独立降采样、再 PW 混合；
     PSD 反转为先混合、后降采样，避免高分辨率路径过早丢失跨通道局部结构。）
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.dw = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1,
                            groups=c_out, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.pw(x)))
        x = self.relu(self.bn2(self.dw(x)))
        return x


class PSDDetail(nn.Module):
    """三尺度 PSD detail branch（1/2、1/4、1/8），T1/T2 共享权重，共 78,464 参数。

    pretrained=True 时：初始化阶段临时构造 ImageNet ResNet18，把 conv1/bn1 权重
    原位复制进 stem_conv/stem_bn 后立即释放；最终 state_dict 不含 layer1-4/fc。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)

        self.d2_refine = ResidualDS(64)          # D2: 64×128×128
        self.down4 = MixDown(64, 128)
        self.d4_refine = ResidualDS(128)         # D4: 128×64×64
        self.down8 = MixDown(128, 256)           # D8: 256×32×32（无 residual block）

        if pretrained:
            self._load_imagenet_stem()

    def _load_imagenet_stem(self):
        """原位复制 ImageNet ResNet18 的 conv1/bn1 权重（决策文档 §13）。"""
        ref = resnet18(pretrained=True)
        with torch.no_grad():
            self.stem_conv.weight.copy_(ref.conv1.weight.data)
        self.stem_bn.load_state_dict(ref.bn1.state_dict())
        del ref

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))   # 64×128×128
        d2 = self.d2_refine(x)                                # 64×128×128
        d4 = self.d4_refine(self.down4(d2))                   # 128×64×64
        d8 = self.down8(d4)                                   # 256×32×32
        return d2, d4, d8
