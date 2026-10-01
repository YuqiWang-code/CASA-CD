"""Run5 模块：MobileDetail-P3 —— MobileNetV3-Small 极浅 ImageNet 预训练 prefix。

设计（CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md §11）：
  - 完整继承 torchvision MobileNetV3-Small（ImageNet-1K V1 官方权重）的
    features 0–3，不是只复制一个 stem：
      f0: Conv3×3 s2 3→16 + BN + Hardswish     → D2: 16×128×128 (1/2)
      f1: InvertedResidual(16→16, s2, SE, ReLU) → D4: 16×64×64   (1/4)
      f2: InvertedResidual(16→72→24, s2, ReLU)  → 24×32×32      (1/8)
      f3: InvertedResidual(24→88→24, s1, 残差)   → D8: 24×32×32  (1/8 refinement)
  - 最终 state_dict 只含 f0–f3（features.4+ / classifier / avgpool 一律不注册）；
  - T1/T2 共享权重（Siamese）。

参数（实测 torchvision 0.29，与方案文档 §11 一致）：
  f0=464, f1=744, f2=3,864, f3=5,416 → 合计 10,488。
"""
import torch
import torch.nn as nn

from copy import deepcopy
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileDetail(nn.Module):
    """MobileNetV3-Small features 0–3 极浅预训练 detail branch。

    pretrained_weight_path=None 时走 torchvision 在线权重（仅本地诊断用）；
    正式训练必须传本地 .pth 路径（决策文档 §10：正式训练时不联网下载）。
    """

    def __init__(self, pretrained_weight_path=None):
        super().__init__()
        if pretrained_weight_path is None:
            ref = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            ref = mobilenet_v3_small(weights=None)
            sd = torch.load(pretrained_weight_path, map_location="cpu")
            ref.load_state_dict(sd)

        self.f0 = deepcopy(ref.features[0])
        self.f1 = deepcopy(ref.features[1])
        self.f2 = deepcopy(ref.features[2])
        self.f3 = deepcopy(ref.features[3])
        del ref

    def forward(self, x):
        d2 = self.f0(x)       # B×16×128×128
        d4 = self.f1(d2)      # B×16×64×64
        x8 = self.f2(d4)      # B×24×32×32
        d8 = self.f3(x8)      # B×24×32×32
        return d2, d4, d8
