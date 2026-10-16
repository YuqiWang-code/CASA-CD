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


def confidence_preserving_pool2x2(x, abs_score, rank_score, eps=1e-6):
    """CP-CAACP（Run2 首选一）：置信度保留的 2×2 cell 加权聚合。

    动机：rank-only 权重在「零变化图」上仍强制 cell 内不均匀（rank 存在 min/max），
    使 c_ca 偏离 c_avg → 训练早期向低频特征注入噪声 → 弱化 dense 路径、拖低 Precision。
    CP 公式：q_i = s_i * r_i（abs 余弦变化 × rank 归一化），
            w_i = (1 + q_i) / Σ_j(1 + q_j)。
    - s_i = 0（该像素双时相特征完全一致）→ q_i = 0 → 与所有零变化邻居等权：
      全零变化 cell 严格退化为均匀池化（c_ca == c_avg，修正项精确为 0）；
    - s_i 高且 rank 高 → 权重显著抬升（变化区域主导聚合）；
    - 参数自由、A/B 共享权重、规则网格不变。

    abs_score: (B, H, W) in [0, 2]（clamp 后的 1-cos）；rank_score: (B, H, W) in [0, 1]。
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    q = abs_score * rank_score                            # (B, H, W)，无变化 → 0
    q = q.view(B, 1, H // 2, 2, W // 2, 2)
    w = (1.0 + q)
    w = w / w.sum(dim=(3, 5), keepdim=True)
    xr = x.view(B, C, H // 2, 2, W // 2, 2)
    return (xr * w).sum(dim=(3, 5))


class CAACPSS2D(SS2D):
    """SS2D + CAACP（只用于 index=2 / Stage3 最终 TViM）。

    参数结构 = 官方 SS2D + 唯一新增 scalar beta（zero-init）。预训练加载时
    beta 不在 checkpoint 中 → 保持 0，epoch-0 输出与官方逐位一致。

    score_mode:
      "rank" — Run1 公式 w ∝ eps + rank（默认，向后兼容）；
      "cp"   — Run2 CP 公式 w ∝ 1 + s·r（confidence-preserving）。
    residual_mode（Run3 E4 RA-CAACP）:
      "current"    — res = x - Up(c)（官方结构；change-aware c 同时进 SS2D 与 residual 扣除）
      "avg_anchor" — res = x - Up(c_avg)（RA：residual 永远锚定官方均匀 context，
                     保护 dense 高频残差不被 change-aware pooling 减掉；β=0 两者等价）
    """

    def __init__(self, eps=1e-6, score_mode="rank", residual_mode="current", **kwargs):
        assert kwargs.get("index", -1) == 2, "CAACP only applies to Stage3 (index=2) final TViM"
        assert score_mode in ("rank", "cp")
        assert residual_mode in ("current", "avg_anchor")
        super().__init__(**kwargs)
        self.score_mode = score_mode
        self.residual_mode = residual_mode
        self.beta = nn.Parameter(torch.zeros(1))
        self._pair_score = None
        self._pair_abs = None
        self._score_delta = 0.0    # |c_ca - c_avg| 均值（诊断记录）
        self._abs_mean = None      # cp 模式 abs score 均值（诊断记录）
        self._weight_entropy = None  # 2×2 cell 权重熵均值（诊断记录；均匀=ln4）

    def set_pair_score(self, score, abs_score=None):
        """注入共享变化分数（2B, H, W；A/B 两半相同）。调用方负责 no_grad。

        score: rank 归一化分数；abs_score: clamp 后的 [0,2] 余弦分数（cp 模式必填）。
        """
        self._pair_score = score
        self._pair_abs = abs_score

    @staticmethod
    def _cp_weights(abs_score, rank_score):
        """cp 模式 cell 权重 (B,1,H//2,2,W//2,2)，与 confidence_preserving_pool2x2 同口径（仅诊断）。"""
        B, H, W = abs_score.shape
        q = (abs_score * rank_score).view(B, 1, H // 2, 2, W // 2, 2)
        w = 1.0 + q
        return w / w.sum(dim=(3, 5), keepdim=True)

    @staticmethod
    def _rank_weights(score):
        """rank 模式 cell 权重 (B,1,H//2,2,W//2,2)，与 change_weighted_pool2x2 同口径（仅诊断）。"""
        B, H, W = score.shape
        s = score.view(B, 1, H // 2, 2, W // 2, 2)
        w = s + 1e-6
        return w / w.sum(dim=(3, 5), keepdim=True)

    def weight_entropy(self):
        return self._weight_entropy

    def abs_mean(self):
        return self._abs_mean

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
                if self.score_mode == "cp":
                    assert self._pair_abs is not None, "cp mode requires abs_score in set_pair_score"
                    c_ca = confidence_preserving_pool2x2(x_low, self._pair_abs, self._pair_score)
                    w = self._cp_weights(self._pair_abs, self._pair_score)
                else:
                    c_ca = change_weighted_pool2x2(x_low, self._pair_score)
                    w = self._rank_weights(self._pair_score)
                with torch.no_grad():
                    self._score_delta = (c_ca - c_avg).abs().mean().item()
                    # Run2 §12.1B 诊断：cell 权重熵（均匀=ln4≈1.386）+ abs score 均值
                    ent = -(w * (w + 1e-12).log()).sum(dim=(3, 5)).mean().item()
                    self._weight_entropy = ent
                    self._abs_mean = (float(self._pair_abs.mean().item())
                                      if self._pair_abs is not None else None)
                c = c_avg + self.beta * (c_ca - c_avg)
            else:
                c = c_avg
            # RA-CAACP（Run3 E4）：residual 锚定 c_avg（官方均匀 context），
            # 使 change-aware c 只进 SS2D、不从 dense residual 中扣除高频证据；
            # β=0 时两种模式与官方 TinyViM 逐位一致。
            if self.residual_mode == "avg_anchor":
                res = x0 - F.interpolate(c_avg, (H, W), mode='nearest')
            else:
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
