"""Run7 HFAE：编码器前两层「频率分解高频分支」→ 中高频增强边缘检测辅助模块。

## 机制（与 `docs` 预注册文档 §2/§3 逐条对应）

TinyViM 的频率解耦在 `SS2D.forward_core`：

    x_low, x_high = torch.split(x, self.split, dim=1)
    x_high = self.local_conv(x_high)          # ← 高频分支（RepDW），全分辨率，不经 pool
    if self.index < 3:
        res = x_low - F.interpolate(self.pool(x_low), (H, W), 'nearest')   # Laplace 残差

本模块**只读地**接出 `local_conv` 的高频输出（forward hook），不改动 `ss2d.py` 任何算子。
实测频率分配（`ssm_ratio=1.0`、`split_list=[1/4,1/2,1/2,3/4]`）：

| stage | 分辨率 | d_model | 低频 ch | **高频 ch** | HF 占比 | SS2D 池化 |
|---|---|---|---|---|---|---|
| stage0 | 1/4 (64²) | 48 | 12 | **36** | 75% | 8× |
| stage1 | 1/8 (32²) | 64 | 32 | **32** | 50% | 4× |
| stage2 | 1/16 | 168 | 84 | 84 | 50% | 2× |
| stage4 | 1/32 | 224 | 168 | 56 | 25% | 1× |

**stage0 是全网络最"高频主导"的阶段**（75% 通道给 HF，低频被 8× 池化压到 8²）——
这正是"前两层、高分辨率、小目标清晰"的架构依据；Run5 只接过 1/8 与 1/16，从未接过 1/4，
且 Run5 是**无监督读出**（已被数据否决），本模块是**有监督的表征重塑**。

## 两个部署形态（由 `--hf_edge_form` 选定，实验前固定）

| form | 训练期 | `switch_to_deploy()` 后 | deploy 参数 |
|---|---|---|---|
| `aux_only` | 深监督边缘头（梯度经 tap 回流编码器 HF 分支） | **整个 head 被删除** | **不变（4,880,190）** |
| `aux_inject` | 深监督 + 零初始化注入 DCR 最小尺度输入 | 保留 proj/dw/inject，仅丢监督专用 `edge_out` | +≈9.9k |

- **双时相配对**：tap 为 `(2B,C,H,W)`（前 B = A、后 B = B，因为 `x2b = cat([pre,post],0)`）。
  边缘头输入取 `cat([A, B, |A−B|])`，即"中高频增强后的变化边缘证据"⇒ 输出 batch = **B**
  （每对样品**一张**变化边缘图，而不是给 A/B 各一份）。
  **首版踩坑**：曾把 head 输出当成 2B 并据此构造 2B 目标、又对 B 尺寸张量做"相位平均"，
  导致形状错配（服务器 dry-run 抓到）；现固定为 B 尺寸，语义也更干净。
- **边缘监督目标**：GT 轮廓带 `E = dilate₄(gt) \\ erode₄(gt)`（4 邻接，内外各 1px），
  下采样用 `adaptive_max_pool2d` **保底**（1px 轮廓不会被降采样吃掉），A/B 共享同一目标。
- **零初始化**：注入 1×1 的 weight/bias 全零 ⇒ `epoch-0` 与关闭开关**逐位相同**。
- **RNG 纪律**：head 在 `torch.random.fork_rng(devices=[])` + 固定局部种子内构造，
  **不推进全局随机流** ⇒ CTRL 与 EXP 的既有模块初始化逐位一致。

## 为什么这是"新机制"而不是重试旧方案

Run5（原生 HF → TAR）否证的是**无监督读出**；F14 否证的是**手工证据通道**；
Run6 否证的是**读出层门控**（信息已在 head，AUC 0.9565）。本模块改的是
**训练目标驱动的表征**（边缘监督的梯度直接作用于 `local_conv` 高频分支），
属于与上述三者都不同的机制类。机制自检 T5 强制验证"梯度真的进了 HF 分支"。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.ss2d import Conv2d_BN, trunc_normal_
from model.utils import BCEDiceLoss

HF_EDGE_FORM = "hfae_v1"
HF_EDGE_FORMS = ("none", "aux_only", "aux_inject")

# tap 规格：(TinyViMSlim.network 下标, stage 内 block 下标, HF 通道, 特征分辨率)
# network 布局 = [s0, emb, s1, emb, s2, emb, s4]（见 tinyvim_s_slim.TinyViMSlim）
HF_EDGE_TAPS = {
    "1_4_1_8": ((0, 2, 36, 64), (2, 2, 32, 32)),
    "1_4": ((0, 2, 36, 64),),
}
HF_EDGE_TAP_KEYS = {"1_4_1_8": ("n0b2", "n2b2"), "1_4": ("n0b2",)}

HF_EDGE_LAMBDA_DEFAULT = 0.2      # 预注册固定值；禁止搜索（见 README §3「不做 loss 调参」）
HF_EDGE_LOCAL_SEED = 16           # 局部 RNG 种子（与 fine_tap 同惯例）

_CROSS4 = ((0, 1, 0), (1, 1, 1), (0, 1, 0))     # 4 邻接结构元（与项目对象指标口径一致）


# --------------------------------------------------------------------------- 边缘目标
def _cross_kernel(dtype, device):
    return torch.tensor(_CROSS4, dtype=dtype, device=device).view(1, 1, 3, 3)


def edge_target_from_label(label, out_size):
    """GT 轮廓带 → `(B,1,out,out)`。

    注意口径：边缘头把双时相 **配对**成**一张**"变化边缘图"（`cat([A,B,|A−B|])` ⇒ 输出 batch = B），
    因此目标是**每对一张**变化轮廓（而不是给 A/B 各一份）。

    `E = dilate4(gt) \\ erode4(gt)`：4 邻接结构元下 `conv2d` 的 3×3 cross 和 ∈{0..5}；
    `>0` 给出膨胀、`>=5` 给出腐蚀（中心+4 邻域全为 1）。
    下采样用 `adaptive_max_pool2d` 保底，避免 1–16px 小对象的 1px 轮廓被降采样吃掉。
    """
    x = label.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.dim() == 4 and x.shape[1] == 1, f"unexpected label shape {tuple(x.shape)}"
    k = _cross_kernel(x.dtype, x.device)
    s = F.conv2d(x, k, padding=1)
    dil = (s > 0).float()
    ero = (s >= 5).float()
    band = dil - ero
    if band.shape[-1] != out_size:
        band = F.adaptive_max_pool2d(band, out_size)
    return band


def hf_edge_loss(logits, label, weight=HF_EDGE_LAMBDA_DEFAULT):
    """边缘辅助损失：`weight · (BCE + 1 − Dice)`（与主损失同族，复用 `BCEDiceLoss`）。"""
    if logits is None:
        return None
    tgt = edge_target_from_label(label, logits.shape[-2])
    return float(weight) * BCEDiceLoss(torch.sigmoid(logits), tgt)


# --------------------------------------------------------------------------- 模块
class HFEdgeHead(nn.Module):
    """多尺度高频证据 → 中高频增强 → 边缘 logit（监督）+ 相位聚合边缘特征（注入）。"""

    def __init__(self, taps="1_4_1_8", mid=32, inject_dim=96,
                 form="aux_only", local_seed=HF_EDGE_LOCAL_SEED):
        super().__init__()
        assert taps in HF_EDGE_TAPS, f"unknown taps {taps}"
        assert form in ("aux_only", "aux_inject"), f"unknown form {form}"
        self.taps = taps
        self.specs = HF_EDGE_TAPS[taps]
        self.mid = int(mid)
        self.form = form
        self.inject_dim = int(inject_dim)
        self.local_seed = int(local_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.local_seed)
            # 每尺度 1×1：cat([A, B, |A−B|]) → mid（变化高频证据）
            self.projs = nn.ModuleList([Conv2d_BN(3 * ch, self.mid, 1)
                                        for (_n, _b, ch, _r) in self.specs])
            self.dw = Conv2d_BN(self.mid, self.mid, 3, 1, 1, groups=self.mid)
            self.act = nn.GELU()
            self.edge_out = nn.Conv2d(self.mid, 1, 1)        # 监督专用，deploy 丢弃
            if self.form == "aux_inject":
                self.inject = nn.Conv2d(self.mid, self.inject_dim, 1)
            else:
                self.inject = None
            self.apply(self._init_weights)                   # 通用初始化（含 inject）
            if self.inject is not None:
                # **必须在 apply 之后**零初始化，否则会被 trunc_normal_ 覆盖
                # （首版就踩了这个坑：自检 inject_zero_init=False 暴露，会破坏 epoch-0 恒等）
                nn.init.zeros_(self.inject.weight)
                nn.init.zeros_(self.inject.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, taps):
        """`taps` = 按 `self.specs` 顺序的 `(2B,C,H,W)` 列表。

        双时相 **配对**成一张变化边缘图：`cat([A, B, |A−B|])` ⇒ 输出 batch = **B**。
        返回 `(edge_logits (B,1,h,w) | None, edge_feat (B,mid,h,w))`。
        """
        assert len(taps) == len(self.projs), f"expected {len(self.projs)} taps, got {len(taps)}"
        ref = taps[0].shape[-2:]
        z = None
        for t, proj in zip(taps, self.projs):
            if t.shape[-2:] != ref:
                t = F.interpolate(t, size=ref, mode="bilinear", align_corners=False)
            a, b = t.chunk(2, dim=0)                          # 前 B = A，后 B = B
            cur = proj(torch.cat([a, b, (a - b).abs()], dim=1))   # (B, mid, h, w)
            z = cur if z is None else z + cur                 # 同分辨率相加（细尺度为基准）
        z = self.act(self.dw(z))                              # 中高频增强
        logits = self.edge_out(z) if self.edge_out is not None else None
        return logits, z                                      # 均为 B 尺寸

    # ------------------------------------------------------------------
    def drop_supervision_head(self):
        """`aux_inject` 的 deploy 形态：丢弃监督专用 `edge_out`，保留可推理部分。"""
        if self.edge_out is not None:
            self.edge_out = None
        return self

    def param_report(self):
        return {
            "tap_specs": [list(s) for s in self.specs],
            "mid": self.mid,
            "form": self.form,
            "total_params": int(sum(p.numel() for p in self.parameters())),
            "supervision_only_params": (1 * self.mid + 1) if self.edge_out is not None else 0,
            "inject_params": (self.mid * self.inject_dim + self.inject_dim)
            if self.inject is not None else 0,
            "inject_zero_init": bool(self.inject is not None
                                     and float(self.inject.weight.detach().abs().sum()) == 0.0
                                     and float(self.inject.bias.detach().abs().sum()) == 0.0),
        }


def build_hf_edge_head(taps="1_4_1_8", form="aux_only", inject_dim=96, mid=32,
                       local_seed=HF_EDGE_LOCAL_SEED):
    return HFEdgeHead(taps=taps, mid=mid, inject_dim=inject_dim, form=form,
                      local_seed=local_seed)
