"""STGEO-SUP（Exp1 的 B 形态）：在**网络特征**上计算 SPD(2)/H² 测地变化分数，用 GT 掩码监督。

## 为什么必须作用在特征上（而不是原始图像）

v1/v2 的几何证据均从**原始图像**计算。那种设计下，监督信号的梯度只能流进
「几何→小投影头」这条 7k 参数的通路，**永远到不了主干** ⇒ 监督形态退化为
"训一个小辅助头然后删掉"，对推理**零影响**（这正是本项目反复强调的"注入/旁路"陷阱）。

因此本模块把**同一套几何**搬到网络特征上：
  F_t = proj(stage0 特征)            # (B,C_small,H/4,W/4)，A/B **共享** proj
  J_t = Σ_c [ G*(∂x F²), G*(∂x F ∂y F); ·, G*(∂y F²) ]     # 2×2 结构张量（特征域）
  z_t = ((Jxx−Jyy) + 2i·Jxy)/(Jxx+Jyy)                      # 双角度坐标（H² 的盘坐标）
  s   = √2·arccosh(1 + 2W)   ← 与 st_geodesic.py 完全同一个闭式距离
  logits = Conv1x1(s)  → 双线性上采样到 256²
  L_aux  = λ · (BCE + Dice)(sigmoid(logits), GT)
⇒ 梯度经 s 回到 z 回到 J 回到 ∂F 回到 **proj** 再回到 **stage0 特征**，
   即「用几何监督重塑表征」——与 Run7 E9（HFAE aux_only）**同一形态**，
   但被监督的量是**二阶结构统计在双曲平面上的测地距离**（参数自由、尺度不变）。

## 与 Run8 HQKV/HGSL 的区别

| | Run8 HGSL | 本模块 |
|---|---|---|
| 流形上的点 | 逐像素特征向量（一阶）经 Lorentz 映射 | **局部结构张量**（二阶）∈ SPD(2) |
| 距离 | 需人造非恒定半径才不退化 | 闭式 H² 测地距离，**无半径参数** |
| 尺度不变 | 无 | **精确**（J→c²J ⇒ z 不变） |
| 推理期 | 保留（HQKV 是真实算子） | **整支删除**（aux_only）⇒ deploy 与 CTRL 逐位同参数 |

## 部署纪律

`form="sup"` 时：训练期有该支路，`switch_to_deploy()` 里**整支置 None**
（与 Run7 HFAE `aux_only` 同口径）⇒ 部署图与 CTRL_R8 **完全相同**（非主干 2,038,868）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.st_geodesic import (
    TR_FLOOR,
    _central_diff,
    _gaussian_blur,
    poincare_shape_distance,
    shape_disk_coords,
)
from model.utils import BCEDiceLoss

STGEO_SUP_FORM = "stgeo_sup_feat_v1"
STGEO_SUP_LAMBDA = 0.2      # 预注册固定，禁止搜索（与 Run7 HFAE / Run8 HGSL 同值）
STGEO_SUP_SEED = 29


def feature_structure_tensor(feat, sigma_i=1.0):
    """特征域结构张量（通道求和）：feat (B,C,H,W) → (Jxx,Jxy,Jyy)，各 (B,1,H,W)。"""
    gx = _central_diff(feat, 3)
    gy = _central_diff(feat, 4)
    jxx = _gaussian_blur((gx * gx).sum(dim=1, keepdim=True), sigma_i)
    jxy = _gaussian_blur((gx * gy).sum(dim=1, keepdim=True), sigma_i)
    jyy = _gaussian_blur((gy * gy).sum(dim=1, keepdim=True), sigma_i)
    return jxx, jxy, jyy


def feature_geodesic_score(feat_a, feat_b, tr_floor=TR_FLOOR):
    """双时相特征 → H² 测地变化分数 s ∈ [0, ∞)，(B,1,H,W)。参数自由。"""
    ja = feature_structure_tensor(feat_a)
    jb = feature_structure_tensor(feat_b)
    za_x, za_y, ka, va = shape_disk_coords(*ja, tr_floor=tr_floor)
    zb_x, zb_y, kb, vb = shape_disk_coords(*jb, tr_floor=tr_floor)
    s = poincare_shape_distance(za_x, za_y, ka, zb_x, zb_y, kb)
    return s * (va * vb)          # 任一时相平坦 ⇒ 分数 0（形状无定义）


class STGeoFeatureSup(nn.Module):
    """特征域 SPD(2)/H² 测地变化分数 + 1×1 变化头（**训练期专用**）。

    参数：proj = C_in×C_small（默认 96×16=1,536）+ head 1×1（2）≈ **1,538**，deploy 全部删除。
    """

    def __init__(self, in_ch=96, small=16, lambda_=STGEO_SUP_LAMBDA, init_seed=STGEO_SUP_SEED):
        super().__init__()
        self.in_ch = int(in_ch)
        self.small = int(small)
        self.lambda_ = float(lambda_)
        self.form_tag = STGEO_SUP_FORM
        with torch.random.fork_rng(devices=[]):     # 不推进全局 RNG
            torch.manual_seed(int(init_seed))
            self.proj = nn.Conv2d(self.in_ch, self.small, 1, bias=False)
            self.head = nn.Conv2d(1, 1, 1, bias=True)
            self.apply(self._init_weights)
        self._last_score = None

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, feat_a, feat_b, out_size):
        """feat_a/feat_b: (B,C,H,W) 同一层的双时相特征；out_size: (H_out,W_out)。"""
        pa = self.proj(feat_a)
        pb = self.proj(feat_b)          # A/B **共享** proj（严格交换对称）
        s = feature_geodesic_score(pa, pb)
        self._last_score = s.detach()
        logits = self.head(s)
        if tuple(logits.shape[-2:]) != tuple(out_size):
            logits = F.interpolate(logits, size=out_size, mode="bilinear",
                                   align_corners=False)
        return logits

    def loss(self, logits, gt):
        """λ · (BCE + Dice)(sigmoid(logits), GT)。GT 用 max 池化保底到 logits 分辨率。"""
        if logits is None:
            return None
        tgt = gt.float()
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(1)
        if tuple(tgt.shape[-2:]) != tuple(logits.shape[-2:]):
            tgt = F.adaptive_max_pool2d(tgt, logits.shape[-2])
        return float(self.lambda_) * BCEDiceLoss(torch.sigmoid(logits), tgt)

    def score_stats(self):
        s = self._last_score
        if s is None:
            return {}
        return {"score_mean": float(s.mean()), "score_p99": float(s.flatten().quantile(0.99))}

    def param_report(self):
        return {"form": self.form_tag, "in_ch": self.in_ch, "small": self.small,
                "lambda": self.lambda_,
                "params": int(sum(p.numel() for p in self.parameters())),
                "deploy_dropped": True}
