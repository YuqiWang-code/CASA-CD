"""CASA-VMamba-STRNet v2（Run8b 血统）：**拆除双曲几何后**的 CASA-VMamba 网络。

## 2026-10-20 之后的两件事，分清楚

**（a）拆除**：原来的 `geometry=hyperbolic|euclidean` 走 Lorentz 映射 + 半径分解 + detail/global
双分支特化 QKV，实测 `B1_HYP` 四库 ΔF1 均值 −0.00085 且更贵 ⇒ **已拆**。
现在 `geometry` 只接受 `"none"`/`"lattice"`，传旧值直接 `SystemExit`。

**（b）新增 Run9 HSAM**：`hsam=True` 打开 `model/layers/hsam.py` 的
**Poincaré 球双曲相似度注意力**（论文 JSTARS 2024 的 HSAM，**零参数**）。
它作用在编码器**最深两层**的输出上：exp 映射 → 双曲距离 → `HW×HW` 注意力矩阵 →
双向聚合 + 残差，得到 `F^H_1/F^H_2` 后**替换**原特征再进 TAR/DCR。
注意：这两件事的几何**不是同一个东西** —— (a) 是 Lorentz 上做"变化分数→半径分解"，
(b) 是 Poincaré 球上做"相似度注意力"。

现在：

| 项 | 拆除后 |
|---|---|
| `geometry` | **只接受 `"none"` / `"lattice"`**；传 `"hyperbolic"`/`"euclidean"` **直接 `SystemExit`** |
| HQKV 模块 | `caacp_hqkv_lattice.CAACPHQKVLattice`（无几何、无半径、无分数；格点压缩 SAT-QKV） |
| `hqkv_detail_window` / `hqkv_global_window` / `hqkv_r0` / `hqkv_a` | **已删除**（双窗口与半径参数失去意义） |
| `hgsl` | **`hqkv=1 且 hgsl=1` 直接报错**：几何拆掉后没有分数可喂，不再静默跳过 |
| 诊断口 | `hqkv_report()` / `hqkv_diag()` / `snapshot_gamma_grad()` / `contrib_norm()` 全部保留 |
| 注入点 | 不变：`delta` 加到 `TAR` 输出的 `feats[0]`（`inject_mode=dcr1`） |

## 保留的、与几何无关的资产（拆除时**故意没动**）

- VMamba-Tiny 主干移植与预训练加载（`load_pretrained` 212/222、`worst_diff=0.0`）；
- **反自锁**工程手段：每支 `rms_normalize` + 独立 γ 门 + 非零初值 + `contrib_norm` fail-fast；
- TAR / DCR 折叠、`switch_to_deploy()` 语义（HQKV 是推理期真实算子，不折叠）；
- `param_breakdown()` / `pretrain_load_stats()` 等诊断。

## fail-fast（Run8 教训的延续）

`train.py` 在 **step 1000** 检查 `contrib_norm`：若 < 1e-3 直接 `SystemExit`，
让「模块零贡献」在 20 分钟内暴露，而不是跑满 80K 才发现。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.caacp_hqkv_lattice import (
    GAMMA_INIT,
    HQKV_FORM,
    HQKV_SEED,
    LATTICE_WINDOW_DEFAULT,
    build_caacp_hqkv_lattice,
)
from model.layers.hsam import ATTN_STAGES_DEFAULT as HSAM_STAGES_DEFAULT
from model.layers.hsam import HSAM_FORM, LOGIT_NORM_DEFAULT as HSAM_LOGIT_NORM_DEFAULT
from model.layers.hsam import RADIUS_MODE_DEFAULT as HSAM_RADIUS_DEFAULT
from model.layers.hsam import build_hsam
from model.str_dcr import DCRDecoder
from model.str_tar import MultiScaleTAR
from model.vmamba_tiny import build_vmamba_tiny

ENCODER_DIMS = (96, 192, 384, 768)
STR_DIM = 192
# 拆除后只允许这两项；"hyperbolic"/"euclidean" 已被移除（见模块头）
GEOMETRIES = ("none", "lattice")
REMOVED_GEOMETRIES = ("hyperbolic", "euclidean", "hyp", "euc", "lorentz", "manifold")
INJECT_MODES = ("none", "dcr1", "dcr1_16")
HQKV_FORM_NAME = HQKV_FORM
FAILFAST_MIN_CONTRIB = 1e-3


class CASAVMambaSTRNetV2(nn.Module):
    def __init__(self, vmamba_pretrained_path=None, hqkv=True, str_dim=STR_DIM,
                 geometry="lattice", hgsl=False, hgsl_detach_score=True,
                 inject_mode="dcr1",
                 hqkv_rank=32, hqkv_lite_dim=192,
                 hqkv_window=LATTICE_WINDOW_DEFAULT,
                 hqkv_attn_stages=(1, 2, 3),
                 rep_mode="full", fs_tar=False, vmamba_cfg=None,
                 hqkv_seed=HQKV_SEED, gamma_init=GAMMA_INIT,
                 hsam=False, hsam_stages=HSAM_STAGES_DEFAULT,
                 hsam_logit_norm=HSAM_LOGIT_NORM_DEFAULT,
                 hsam_radius=HSAM_RADIUS_DEFAULT,
                 hsam_feature_source="pre_norm"):
        super().__init__()
        g = str(geometry).lower()
        if g in REMOVED_GEOMETRIES:
            raise SystemExit(
                f"[HQKV-GEOMETRY-REMOVED] geometry='{geometry}' 已于 2026-10-20 拆除。\n"
                f"  原因：B1_HYP 四库 ΔF1 均值 −0.00085（1 平 3 负），且多付 0.304 M 参数 / "
                f"+0.46 G FLOPs / 约 +9% 时间。\n"
                f"  归档（字节可复核）："
                f"train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/\n"
                f"  现在只接受 geometry in {GEOMETRIES}；要重开流形几何请先写新的预注册。\n"
                f"  注：Run9 的双曲模块是 **HSAM**（Poincaré 球、零参数、只作相似度），"
                f"由 `hsam=True` 打开，与本条无关。")
        if g not in GEOMETRIES:
            raise SystemExit(f"[HQKV-GEOMETRY] 不支持 {geometry!r}，只接受 {GEOMETRIES}")
        if hqkv and hgsl:
            raise SystemExit(
                "[HGSL-REMOVED] hqkv=1 且 hgsl=1 已不可用：几何拆除后 CAACP 不再产生变化分数，"
                "HGSL 没有输入口。\n"
                "  HGSL 的实现留在 models/model/hgsl_loss.py（未删），但需要先有分数来源；"
                "若要用它，必须另开一轮预注册说明分数从哪来。")
        if hsam and hgsl:
            raise SystemExit(
                "[HGSL-REMOVED] hsam=1 且 hgsl=1 已不可用：HSAM 不产生变化分数（它只重写特征），"
                "HGSL 没有输入口。")
        assert inject_mode in INJECT_MODES, inject_mode
        assert rep_mode in ("plain", "full")

        self.hqkv = bool(hqkv)
        self.geometry = g
        self.hgsl = bool(hgsl)
        self.hgsl_detach_score = bool(hgsl_detach_score)
        self.inject_mode = inject_mode
        self.rep_mode = rep_mode
        self.str_dim = int(str_dim)
        self.fs_tar = bool(fs_tar)
        self.hqkv_form = HQKV_FORM_NAME if self.hqkv else None
        self.gamma_init = float(gamma_init)
        self.hsam = bool(hsam)
        self.hsam_stages = tuple(int(s) for s in hsam_stages)
        self.hsam_form = HSAM_FORM if self.hsam else None
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

        # HQKV **最后构造** + fork_rng（保证 RNG 消耗不影响主干初始化）
        self._diag = {}
        self._hsam_diag = {}
        if self.hqkv:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(hqkv_seed))
                self.hqkv_op = build_caacp_hqkv_lattice(
                    stage_dims=ENCODER_DIMS, rank=hqkv_rank,
                    window=hqkv_window,
                    attn_stages=hqkv_attn_stages, lite_dim=hqkv_lite_dim,
                    init_seed=int(hqkv_seed), gamma_init=gamma_init)
        else:
            self.hqkv_op = None
        # Run9：HSAM 零参数 ⇒ 不影响 RNG；放在 HQKV 之后再构造，保持既有 run 的 RNG 序列不变
        self.hsam_logit_norm = str(hsam_logit_norm)
        self.hsam_radius = str(hsam_radius)
        self.hsam_feature_source = str(hsam_feature_source)
        self.hsam_op = (build_hsam(stage_dims=ENCODER_DIMS, stages=self.hsam_stages,
                                   logit_norm=self.hsam_logit_norm,
                                   radius_mode=self.hsam_radius,
                                   feature_source=self.hsam_feature_source)
                        if self.hsam else None)

    # ------------------------------------------------------------------ 诊断
    def hqkv_report(self):
        if self.hqkv_op is None:
            return None
        rep = self.hqkv_op.param_report()
        rep.update({"geometry": self.geometry, "inject_mode": self.inject_mode,
                    "hgsl": self.hgsl, "hgsl_detach_score": self.hgsl_detach_score})
        return rep

    def hgsl_score_maps(self):
        """恒为 `{}`：几何拆除后没有变化分数（`hgsl=1` 在构造期已 fail-closed）。"""
        return {}

    def hqkv_diag(self):
        return {} if self.hqkv_op is None else dict(getattr(self.hqkv_op, "_diag", {}))

    def snapshot_gamma_grad(self):
        if self.hqkv_op is not None:
            self.hqkv_op.snapshot_gamma_grad()

    def contrib_norm(self):
        """当前 `contrib_norm`（已乘门）。fail-fast 用。"""
        d = self.hqkv_diag()
        return float(d.get("contrib_norm", 0.0))

    # ------------------------------------------------------------------ Run9 HSAM
    def hsam_report(self):
        return None if self.hsam_op is None else self.hsam_op.param_report()

    def hsam_diag(self):
        return {} if self.hsam_op is None else self.hsam_op.diag()

    def set_hsam_collect(self, flag=True):
        """打开/关闭 HSAM 统计。默认关闭（每步统计会触发 device sync，白吃吞吐）。"""
        if self.hsam_op is not None:
            self.hsam_op.collect_diag = bool(flag)

    def pretrain_load_stats(self):
        return self.encoder.load_stats()

    def param_breakdown(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        tar = sum(p.numel() for p in self.tar.parameters())
        dcr = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        hq = (sum(p.numel() for p in self.hqkv_op.parameters())
              if self.hqkv_op is not None else 0)
        hs = (sum(p.numel() for p in self.hsam_op.parameters())
              if self.hsam_op is not None else 0)
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": enc, "hqkv": hq, "hsam": hs, "tar": tar, "dcr": dcr,
                "head": head,
                "total": total, "non_backbone": total - enc,
                "non_backbone_no_hqkv": total - enc - hq - hs}

    # ------------------------------------------------------------------
    def forward(self, pre, post, label=None):
        x2b = torch.cat([pre, post], dim=0)
        if self.hsam_op is not None and self.hsam_op.feature_source == "pre_norm":
            # Run9：HSAM 必须在 `outnorm`（LayerNorm2d）**之前**混合。
            # 原因（实测）：outnorm 让每个 token 的范数精确等于 √C ⇒ Poincaré 半径恒定
            # ⇒ 双曲距离退化成余弦距离的单调变换（秩相关 1.000000）。
            pre_taps = self.encoder.forward_pre_norm(x2b)
            del x2b
            a_pre, b_pre = [], []
            for t in pre_taps:
                a, b = t.chunk(2, dim=0)
                a_pre.append(a)
                b_pre.append(b)
            del pre_taps
            a_pre, b_pre = self.hsam_op(a_pre, b_pre)
            self._hsam_diag = self.hsam_op.diag()
            feats_a = [getattr(self.encoder, f"outnorm{i}")(a_pre[i])
                       for i in range(len(a_pre))]
            feats_b = [getattr(self.encoder, f"outnorm{i}")(b_pre[i])
                       for i in range(len(b_pre))]
            del a_pre, b_pre
        else:
            outs = self.encoder.forward_features(x2b)
            del x2b
            feats_a, feats_b = [], []
            for t in outs:
                a, b = t.chunk(2, dim=0)
                feats_a.append(a)
                feats_b.append(b)
            del outs
            if self.hsam_op is not None:      # post_norm 负对照（已证明退化，仅作对照）
                feats_a, feats_b = self.hsam_op(feats_a, feats_b)
                self._hsam_diag = self.hsam_op.diag()
            else:
                self._hsam_diag = {}

        if self.hqkv_op is not None:
            out, diag = self.hqkv_op(feats_a, feats_b)
            self._diag = diag
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

        理由：`train↔deploy` 折叠要求两图前向逐位一致，删掉 HQKV 会让 `feats[0]` 少一项。
        轻量主张只能来自它参数小，不能声称零推理代价。
        """
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self
