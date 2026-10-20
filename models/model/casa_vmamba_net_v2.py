"""CASA-VMamba-STRNet v2（Run8b 血统）：**拆除双曲几何后**的 CASA-VMamba 网络。

## 2026-10-20 拆除（本文件的结构性变更）

拆除前，本文件的 `geometry` 参数会打开 `hyperbolic.py` 的 Lorentz 映射，
并由 `CAACPHQKVv2` 做「双曲变化分数 → 半径分解 → detail/global 双分支特化 QKV」。
实测 `B1_HYP` vs `CTRL_R8B` 四库 **ΔF1 均值 −0.00085**、还更贵 ⇒ 拆除。

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
                 hqkv_seed=HQKV_SEED, gamma_init=GAMMA_INIT):
        super().__init__()
        g = str(geometry).lower()
        if g in REMOVED_GEOMETRIES:
            raise SystemExit(
                f"[HQKV-GEOMETRY-REMOVED] geometry='{geometry}' 已于 2026-10-20 拆除。\n"
                f"  原因：B1_HYP 四库 ΔF1 均值 −0.00085（1 平 3 负），且多付 0.304 M 参数 / "
                f"+0.46 G FLOPs / 约 +9% 时间。\n"
                f"  归档（字节可复核）："
                f"train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/\n"
                f"  现在只接受 geometry in {GEOMETRIES}；要重开流形几何请先写新的预注册。")
        if g not in GEOMETRIES:
            raise SystemExit(f"[HQKV-GEOMETRY] 不支持 {geometry!r}，只接受 {GEOMETRIES}")
        if hqkv and hgsl:
            raise SystemExit(
                "[HGSL-REMOVED] hqkv=1 且 hgsl=1 已不可用：几何拆除后 CAACP 不再产生变化分数，"
                "HGSL 没有输入口。\n"
                "  HGSL 的实现留在 models/model/hgsl_loss.py（未删），但需要先有分数来源；"
                "若要用它，必须另开一轮预注册说明分数从哪来。")
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
