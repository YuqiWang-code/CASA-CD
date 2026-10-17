"""CASA-VMamba-STRNet v2（Run8b）：**抗自锁**版（CAACP-HQKV v2）。

与 v1（`casa_vmamba_net.py`，Run8 用）的差别只有三处，全部指向"让模块真的生效"：

| # | v1（自锁） | v2（抗自锁） |
|---|---|---|
| 1 | `γ` 零初始化 | `γ` 初值 **0.5**，并对每分支输出做 **RMS 归一化** |
| 2 | 三支输出先求和再乘 `γ_detail` ⇒ `γ_global`/`γ_lite` **从未进入前向** | **三支各自独立残差 + 独立门** |
| 3 | stage0 用 `lite_fuse` 但乘 `γ_detail` | stage0 用 `lite_fuse` 并乘 **`γ_lite`** |

**其余全部不变**（骨干 / 4 stage / 双分支 detail-global / Lorentz 双曲 / SAT 式 K/V 压缩
`k·N` 而 Q 保持 `N` / 注入 DCR 1/4 输入 / TAR-DCR / HGSL），以保证 Run8b 与 Run8 的
对比只变"抗自锁"这一个变量。

## 预训练纪律的取舍（必须披露）

v1 用"`γ=0` ⇒ epoch-0 逐位等同无该模块"作为预训练兼容的证明。v2 放弃这条，
改用**更直接、更强**的证明：`[PRETRAIN-LOAD] worst_diff == 0`（主干 212 个 key 逐位相等）。
理由：零初始化 + 乘性门会让 `∂L/∂γ = ⟨∂L/∂r, Δ⟩` 在 epoch-0 就是 0（因为 `r=0` 且
`Δ` 的梯度路径被 `γ` 截断）⇒ **必死自锁**，不是一个可接受的初始化方式。

## fail-fast（本版新增，用户要求）

`train.py` 在 **step 1000** 检查 `contrib_norm`：若 < `FAILFAST_MIN_CONTRIB`（默认 1e-3）
直接 `raise SystemExit`，并提示"回查 gamma 梯度"。这样"模块零贡献"在 20 分钟内暴露，
而不是跑满 80K 才发现（Run8 的教训）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.caacp_hqkv_v2 import (
    GAMMA_INIT,
    HQKV_FORM_V2,
    HQKV_SEED,
    build_caacp_hqkv_v2,
)
from model.str_dcr import DCRDecoder
from model.str_tar import MultiScaleTAR
from model.vmamba_tiny import build_vmamba_tiny

ENCODER_DIMS = (96, 192, 384, 768)
STR_DIM = 192
GEOMETRIES = ("hyperbolic", "euclidean")
INJECT_MODES = ("none", "dcr1", "dcr1_16")
HQKV_FORM = HQKV_FORM_V2
FAILFAST_MIN_CONTRIB = 1e-3


class CASAVMambaSTRNetV2(nn.Module):
    def __init__(self, vmamba_pretrained_path=None, hqkv=True, str_dim=STR_DIM,
                 geometry="hyperbolic", hgsl=True, hgsl_detach_score=True,
                 inject_mode="dcr1",
                 hqkv_rank=32, hqkv_lite_dim=192,
                 hqkv_detail_window=2, hqkv_global_window=4,
                 hqkv_attn_stages=(1, 2, 3),
                 rep_mode="full", fs_tar=False, vmamba_cfg=None,
                 hqkv_r0=0.5, hqkv_a=0.5, hqkv_seed=HQKV_SEED,
                 gamma_init=GAMMA_INIT):
        super().__init__()
        assert geometry in GEOMETRIES, geometry
        assert inject_mode in INJECT_MODES, inject_mode
        assert rep_mode in ("plain", "full")

        self.hqkv = bool(hqkv)
        self.geometry = geometry
        self.hgsl = bool(hgsl)
        self.hgsl_detach_score = bool(hgsl_detach_score)
        self.inject_mode = inject_mode
        self.rep_mode = rep_mode
        self.str_dim = int(str_dim)
        self.fs_tar = bool(fs_tar)
        self.hqkv_form = HQKV_FORM_V2 if self.hqkv else None
        self.gamma_init = float(gamma_init)
        self._failfast_done = False

        self.encoder = build_vmamba_tiny(pretrained_path=vmamba_pretrained_path,
                                         cfg=vmamba_cfg)
        self.head = nn.Conv2d(self.str_dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=ENCODER_DIMS, dim=self.str_dim,
            use_temporal_aux=(rep_mode == "full"),
            use_dcr_aux=(rep_mode == "full"), use_residual=True,
            fine_stage1=self.fs_tar,
        )
        self.decoder = DCRDecoder(dim=self.str_dim, use_aux=(rep_mode == "full"),
                                  use_residual=True)

        # HQKV **最后构造** + fork_rng
        self._scores = {}
        self._diag = {}
        if self.hqkv:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(hqkv_seed))
                self.hqkv_op = build_caacp_hqkv_v2(
                    stage_dims=ENCODER_DIMS, rank=hqkv_rank,
                    detail_window=hqkv_detail_window,
                    global_window=hqkv_global_window,
                    attn_stages=hqkv_attn_stages, lite_dim=hqkv_lite_dim,
                    r0=hqkv_r0, a=hqkv_a, init_seed=int(hqkv_seed),
                    gamma_init=gamma_init,
                    score_stages=(0, 1, 2, 3)
                    if (self.hgsl and not self.hgsl_detach_score) else ())
        else:
            self.hqkv_op = None

    # ------------------------------------------------------------------ 诊断
    def hqkv_report(self):
        if self.hqkv_op is None:
            return None
        rep = self.hqkv_op.param_report()
        rep.update({"geometry": self.geometry, "inject_mode": self.inject_mode,
                    "hgsl": self.hgsl, "hgsl_detach_score": self.hgsl_detach_score})
        return rep

    def hgsl_score_maps(self):
        return {} if self.hqkv_op is None else self.hqkv_op.score_maps()

    def hqkv_diag(self):
        return {} if self.hqkv_op is None else dict(getattr(self.hqkv_op, "_diag", {}))

    def snapshot_gamma_grad(self):
        if self.hqkv_op is not None:
            self.hqkv_op.snapshot_gamma_grad()

    def contrib_norm(self):
        """当前 `contrib_norm`（已乘门）。fail-fast 用。"""
        d = self.hqkv_diag()
        return float(d.get("contrib_norm", 0.0))

    def pretrain_load_stats(self):
        return self.encoder.load_stats()

    def param_breakdown(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        tar = sum(p.numel() for p in self.tar.parameters())
        dcr = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        hq = (sum(p.numel() for p in self.hqkv_op.parameters())
              if self.hqkv_op is not None else 0)
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": enc, "hqkv": hq, "tar": tar, "dcr": dcr, "head": head,
                "total": total, "non_backbone": total - enc,
                "non_backbone_no_hqkv": total - enc - hq}

    # ------------------------------------------------------------------
    def forward(self, pre, post, label=None):
        x2b = torch.cat([pre, post], dim=0)
        outs = self.encoder.forward_features(x2b)
        del x2b
        feats_a, feats_b = [], []
        for t in outs:
            a, b = t.chunk(2, dim=0)
            feats_a.append(a)
            feats_b.append(b)
        del outs

        if self.hqkv_op is not None:
            out, diag = self.hqkv_op(feats_a, feats_b, geometry=self.geometry)
            self._diag = diag
            self._scores = self.hqkv_op.score_maps()
            if self.inject_mode in ("dcr1", "dcr1_16"):
                delta = out
                if self.inject_mode == "dcr1_16" and self.str_dim == delta.shape[1]:
                    feats_a[1] = feats_a[1] + F.interpolate(
                        delta, size=feats_a[1].shape[-2:], mode="bilinear",
                        align_corners=False)
            else:
                delta = None
        else:
            delta = None
            self._diag = {}
            self._scores = {}

        feats = self.tar(feats_a, feats_b)
        if delta is not None:
            assert feats[0].shape[-2:] == delta.shape[-2:], \
                f"[INJECT] 尺度不匹配 {tuple(feats[0].shape[-2:])} vs {tuple(delta.shape[-2:])}"
            feats = list(feats)
            feats[0] = feats[0] + delta
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def switch_to_deploy(self):
        """折叠 TAR/DCR 的训练多分支；**HQKV 保留**（推理期真实算子）。

        理由与 v1 相同：`train↔deploy` 折叠门要求两图前向逐位一致，删掉 HQKV 会让
        `feats[0]` 少一项。轻量主张只能来自它参数小（≈0.31 M），不能声称零推理代价。
        """
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self
