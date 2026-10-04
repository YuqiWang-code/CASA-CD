"""STRFineHead（FRH，Run2 首选二）：128² 重参数化细粒度预测头。

结构（训练图，输入 decoder 输出 (B, D, 64, 64)）：
    x  --bilinear x2--> (B, D, 128, 128)
    logits = base(x) + gamma * [ PW1x1( RepDW3(x) ) ]          (gamma = 0 初始化)

    - base    : Conv1x1(D -> 1)，独立主路径（训练早期即可产出有效 logits）；
    - RepDW3  : 复用 str_reparam.RepDW3（DW3 + DW1x3 + DW3x1 + identity，各带 BN），
                在 128² 分辨率上提供 3×3 邻域细粒度修正；
    - gamma   : 零初始化标量门。epoch-0 修正项 == 0（deploy 输出与纯 base 一致）。

deploy（switch_to_deploy 后）：
    - RepDW3 折为单 DW3（kernel K_c + bias b_c），线性无激活；
    - PW1x1 与 base 的 1×1 与 DW3 全部折叠为一个 Conv2d(D -> 1, k=3, pad=1)：
        W_fused[c] = gamma * a_c * K_c + pad_center(base_w[c])
        b_fused    = gamma * (Σ_c a_c b_c + b_pw) + b_base
    - 分支属性删除（单层部署）；部署参数 865 = 96*9+1（相对 1×1 head 仅 +768）。

RNG 纪律：gamma zero-init 不消耗全局 RNG；dw3 用 kaiming、aux DW 用 skip_init
（与 str_reparam.RepDW3 约定一致），使同变体 A/B 初始化可复现。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_reparam import RepDW3


def pad_center_1x1_to_3x3(w):
    """(1, C, 1, 1) -> (1, C, 3, 3)：1×1 kernel 中心嵌入 3×3。"""
    return F.pad(w, (1, 1, 1, 1))


class STRFineHead(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.dim = dim
        self.deploy = False
        self.base = nn.Conv2d(dim, 1, 1)
        self.dw = RepDW3(dim, use_aux=True, use_residual=True, deploy=False)
        self.pw = nn.Conv2d(dim, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, D, 64, 64) -> logits (B, 1, 128, 128)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.deploy:
            return self.fused(x)
        y = self.base(x) + self.gamma * self.pw(self.dw(x))
        return y

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return self
        K, b = self.dw.get_equivalent_kernel_bias()      # (C, 1, 3, 3), (C,)
        K = K.double()
        b = b.double()
        a = self.pw.weight.double()                      # (1, C, 1, 1)
        b_pw = self.pw.bias.double()                     # (1,)
        g = self.gamma.detach().double().reshape(())
        w0 = self.base.weight.double()                   # (1, C, 1, 1)
        b0 = self.base.bias.double()                     # (1,)

        # per-channel 乘法：pw 权重 a 的 C 维必须对齐 DW kernel K 的 C 维（K.view 显式对齐，
        # 避免 (1,C)×(C,1) 广播成 (C,C) 的错误组合）
        w_f = g * (a * K.view(1, self.dim, 3, 3)) + pad_center_1x1_to_3x3(w0)   # (1, C, 3, 3)
        b_f = g * ((a * b.view(1, self.dim, 1, 1)).sum(dim=1) + b_pw) + b0      # (1, 1, 1)

        self.fused = nn.Conv2d(self.dim, 1, 3, 1, 1)
        self.fused.weight.data = w_f.float()
        self.fused.bias.data = b_f.view(1).float()
        self.deploy = True
        for name in ("base", "dw", "pw", "gamma"):
            if hasattr(self, name):
                delattr(self, name)
        return self
