"""CAACP-SS2D：Change-Aware Asymmetric Context Pooling for SS2D（创新一）。

在 TinyViM-S 的 Stage3（0-based index=2）最终 TViMBlock 的 SS2D 上，把官方
「均匀 AvgPool 低频下采样」替换为「双时相变化感知的 2×2 cell 加权聚合」：

    c_avg = AvgPool2x2(x_low)                       # 官方均匀池化
    c_ca  = change_weighted_pool2x2(x_low, score)   # 变化感知加权聚合（A/B 共享权重）
    c     = c_avg + beta * (c_ca - c_avg)           # β=0 初始化 → 精确恢复官方预训练

设计纪律（调研文档 §8 / §13.3）：
- 压缩发生在**规则二维网格**上（2×2 cell → 规则 8×8 lattice），再进行四向
  CrossScan——不做 TopK、不做 token 重排、不按方向分别压缩（STORM/官方结构前提）；
- 完整 dense local 路径（高频 RepDW 分支 + 16×16 residual）始终保留；
- score 由 1/16 双时相特征 cosine 计算（参数自由、no_grad、T1/T2 对称）；
- A/B 共享同一组 cell 权重（同一 spatial assignment）；
- 只使用一个零初始化机制：β=0 external gate（SS2D 全部预训练参数正常继承）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.ss2d import SS2D, cross_selective_scan


def change_score_cosine_2d(x1, x2):
    """双时相特征的变化分数：s = 1 - cos(x1, x2)，逐像素，参数自由，T1/T2 对称。

    x1/x2: (B, C, H, W) -> (B, H, W) in [0, 2]。
    """
    p1 = F.normalize(x1, dim=1, eps=1e-8)
    p2 = F.normalize(x2, dim=1, eps=1e-8)
    return (1.0 - (p1 * p2).sum(dim=1)).clamp(min=0.0)


def rank_normalize_2d(s):
    """逐图 rank 归一化到 [0, 1]（与 CASAA 的 rank_normalize_per_image 同口径）。"""
    B, H, W = s.shape
    flat = s.flatten(1)                                   # (B, N)
    ranks = flat.argsort(dim=1).argsort(dim=1).float()
    return (ranks / max(flat.shape[1] - 1, 1)).view(B, H, W)


def change_weighted_pool2x2(x, score, eps=1e-6):
    """结构保持的变化感知 2×2 cell 加权聚合（规则网格，非 TopK）。

    x: (B, C, H, W)；score: (B, H, W) 已 rank 归一化（调用方 no_grad）。
    每个 2×2 cell 内：w_i = (eps + s_i) / sum_j(eps + s_j)；A/B 共享权重。
    返回 (B, C, H/2, W/2)。对 x 保留梯度（权重路径无梯度）。
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    s = score.view(B, 1, H // 2, 2, W // 2, 2)
    w = (s + eps)
    w = w / w.sum(dim=(3, 5), keepdim=True)               # cell 内归一化
    xr = x.view(B, C, H // 2, 2, W // 2, 2)
    return (xr * w).sum(dim=(3, 5))                       # (B, C, H/2, W/2)


class CAACPSS2D(SS2D):
    """SS2D + CAACP（只用于 index=2 / Stage3 最终 TViM）。

    参数结构 = 官方 SS2D + 唯一新增 scalar beta（zero-init）。预训练加载时
    beta 不在 checkpoint 中 → 保持 0，epoch-0 输出与官方逐位一致。
    """

    def __init__(self, eps=1e-6, **kwargs):
        assert kwargs.get("index", -1) == 2, "CAACP only applies to Stage3 (index=2) final TViM"
        super().__init__(**kwargs)
        self.beta = nn.Parameter(torch.zeros(1))
        self._pair_score = None
        self._score_delta = 0.0    # |c_ca - c_avg| 均值（诊断记录）

    def set_pair_score(self, score):
        """注入共享变化分数（2B, H, W；A/B 两半相同）。调用方负责 no_grad。"""
        self._pair_score = score

    def forward_core(self, x, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)

        x_low, x_high = torch.split(x, self.split, dim=1)
        B, C, H, W = x.shape
        x_high = self.local_conv(x_high)
        if self.index < 3:
            x0 = x_low
            c_avg = self.pool(x_low)
            if self._pair_score is not None:
                c_ca = change_weighted_pool2x2(x_low, self._pair_score)
                with torch.no_grad():
                    self._score_delta = (c_ca - c_avg).abs().mean().item()
                c = c_avg + self.beta * (c_ca - c_avg)
            else:
                c = c_avg
            res = x0 - F.interpolate(c, (H, W), mode='nearest')
            x_low = c

        x_low = cross_selective_scan(
            x_low, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        x_low = x_low.permute(0, 3, 1, 2)

        if self.index < 3:
            x_low = F.interpolate(x_low, scale_factor=2 ** (3 - self.index), mode='bilinear') + res
        x = torch.cat((x_low, x_high), dim=1)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
