"""CASA-VMamba-STRNet：Run8 主线（VMamba-Tiny 骨干 + CAACP-HQKV + TAR/DCR）。

## 数据流（真实，对应 `forward()` 逐行）

```
T1 [B,3,256,256] ─┐
                   ├─ concat dim0 ─ VMamba-Tiny (2B) ──► f1(1/4,96) f2(1/8,192) f3(1/16,384) f4(1/32,768)
T2 [B,3,256,256] ─┘                                        │
                                                           ├─ chunk(2) → 每时相各 4 个 stage
                                                           │
                    CAACP-HQKV（Run8 创新）◄────────────────┘
                      · 每 stage 双曲映射 → detail / global 两分支
                      · 每分支特化 QKV，K/V 压缩到 k·N，Q 保持 N
                      · 4 个 stage 的增量上采样到 1/4 相加 → γ 零初始化门
                                                           │
                    feats[0] += γ·Δ        （R4：注入**跳层连接**/解码器最小尺度输入）
                                                           │
                       MultiScaleTAR(192,192,192,192, D) ──► DCRDecoder(D) ──► head Conv1x1
                                                           │
                                            bilinear 到 256² ──► sigmoid
```

## 变体开关（唯一变量纪律）

| 开关 | 含义 |
|---|---|
| `hqkv` | 0=不加 CAACP-HQKV（同期 CTRL）；1=加 |
| `geometry` | `hyperbolic`（双曲，主实验）/ `euclidean`（**同式欧氏对照**，调研强制要求） |
| `branch` | `both` / `detail_only` / `global_only`（分支特化的消融） |
| `hgsl` | 0=无新损失；1=HGSL 开启；`hgsl_detach_score` 决定梯度是否进入几何量 |
| `inject_mode` | `dcr1`（只注 1/4）/ `dcr1_16`（1/4 与 1/16 两处跳层）/ `none` |
| `hf_edge` | Run8 **不默认启用**（VMamba 没有 TinyViM 的 `local_conv` 高频旁路，HFAE 的 tap 不存在）；保留开关仅为兼容性检查 |

## 预训练纪律

1. 主干 `VMamba-Tiny` 的 222 个 key 全部逐位加载（`worst_diff` 必须为 **0**，见 `vmamba_tiny.load_pretrained`）；
2. CAACP-HQKV **最后构造** + `fork_rng` 固定局部种子 ⇒ 不推进全局随机流；
3. 三个门 `γ` **零初始化**（在 `apply` 之后）⇒ `hqkv=1` 与 `hqkv=0` 在 **epoch-0 逐位一致**
   （由 T0 smoke 用真实 batch 断言）；
4. TAR/DCR 的辅助分支保持 Run1–Run7 的零初始化约定 ⇒ `switch_to_deploy()` 折叠门仍然有效。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.caacp_hqkv import HQKV_FORM, HQKV_SEED, build_caacp_hqkv
from model.str_dcr import DCRDecoder
from model.str_tar import MultiScaleTAR
from model.vmamba_tiny import STAGE_RES, VMambaBackbone, build_vmamba_tiny

ENCODER_DIMS = (96, 192, 384, 768)
STR_DIM = 192
GEOMETRIES = ("hyperbolic", "euclidean")
BRANCHES = ("both", "detail_only", "global_only")
INJECT_MODES = ("none", "dcr1", "dcr1_16")


class CASAVMambaSTRNet(nn.Module):
    def __init__(self, vmamba_pretrained_path=None, hqkv=True, str_dim=STR_DIM,
                 geometry="hyperbolic", branch="both", hgsl=True,
                 hgsl_detach_score=True, inject_mode="dcr1",
                 hqkv_rank=32, hqkv_lite_dim=192,
                 hqkv_detail_window=2, hqkv_global_window=4,
                 hqkv_attn_stages=(1, 2, 3),
                 rep_mode="full", fs_tar=False,
                 vmamba_cfg=None, hf_edge=False, hf_edge_form="none",
                 hqkv_r0=0.5, hqkv_a=0.5, hqkv_seed=HQKV_SEED):
        super().__init__()
        assert geometry in GEOMETRIES, geometry
        assert branch in BRANCHES, branch
        assert inject_mode in INJECT_MODES, inject_mode
        assert rep_mode in ("plain", "full")
        assert not hf_edge, (
            "[ARCH] Run8 不启用 HFAE：VMamba 没有 TinyViM 的 `local_conv` 高频旁路，"
            "HF_EDGE_TAPS 指向的模块不存在。若确需高频损失，请改用 HGSL（已内置）。")

        self.hqkv = bool(hqkv)
        self.geometry = geometry
        self.branch = branch
        self.hgsl = bool(hgsl)
        self.hgsl_detach_score = bool(hgsl_detach_score)
        self.inject_mode = inject_mode
        self.rep_mode = rep_mode
        self.str_dim = int(str_dim)
        self.fs_tar = bool(fs_tar)
        self.hqkv_form = HQKV_FORM if self.hqkv else None
        self.hf_edge = False

        # ---- 主干（VMamba-Tiny，官方 v2 权重）
        self.encoder = build_vmamba_tiny(pretrained_path=vmamba_pretrained_path,
                                         cfg=vmamba_cfg)

        # ---- head / TAR / DCR（与 Run1–Run7 同族，输入维改为 VMamba 的 (96..768)）
        self.head = nn.Conv2d(self.str_dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=ENCODER_DIMS, dim=self.str_dim,
            use_temporal_aux=(rep_mode == "full"),
            use_dcr_aux=(rep_mode == "full"), use_residual=True,
            fine_stage1=self.fs_tar,
        )
        self.decoder = DCRDecoder(dim=self.str_dim, use_aux=(rep_mode == "full"),
                                  use_residual=True)

        # ---- CAACP-HQKV（**最后构造** + fork_rng）
        self._scores = {}
        self._diag = {}
        if self.hqkv:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(hqkv_seed))
                self.hqkv_op = build_caacp_hqkv(
                    stage_dims=ENCODER_DIMS, rank=hqkv_rank,
                    detail_window=hqkv_detail_window, global_window=hqkv_global_window,
                    attn_stages=hqkv_attn_stages, lite_dim=hqkv_lite_dim,
                    r0=hqkv_r0, a=hqkv_a, init_seed=int(hqkv_seed),
                    score_stages=(0, 1, 2, 3) if (self.hgsl and not self.hgsl_detach_score)
                    else ())
                # γ 零初始化（在 apply 之后，见 HQKV 内注释）
                nn.init.zeros_(self.hqkv_op.gamma_detail)
                nn.init.zeros_(self.hqkv_op.gamma_global)
                nn.init.zeros_(self.hqkv_op.gamma_lite)
        else:
            self.hqkv_op = None

    # ------------------------------------------------------------------ 诊断
    def hqkv_report(self):
        if self.hqkv_op is None:
            return None
        rep = self.hqkv_op.param_report()
        rep.update({"geometry": self.geometry, "branch": self.branch,
                    "inject_mode": self.inject_mode,
                    "hgsl": self.hgsl, "hgsl_detach_score": self.hgsl_detach_score})
        return rep

    def hgsl_score_maps(self):
        """供 `hgsl_loss` 使用：每 stage 的双曲变化分数（B,1,H_i,W_i）。"""
        if self.hqkv_op is None:
            return {}
        return self.hqkv_op.score_maps()

    def hqkv_diag(self):
        return dict(getattr(self.hqkv_op, "_diag", {}) if self.hqkv_op is not None else {})

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
        x2b = torch.cat([pre, post], dim=0)                  # (2B,3,256,256)
        outs = self.encoder.forward_features(x2b)            # 4 × (2B, C_i, H_i, W_i)
        del x2b

        # 每时相各自的 4 个 stage（**不是** A/B 拼接后再分）
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
            # R4：注入到 DCR 的最小尺度输入（1/4）。`dcr1_16` 额外在 1/16 处注入。
            if self.inject_mode in ("dcr1", "dcr1_16"):
                delta = out
                if self.inject_mode == "dcr1_16" and self.str_dim == delta.shape[1]:
                    feats_a[1] = feats_a[1] + F.interpolate(
                        delta, size=feats_a[1].shape[-2:], mode="bilinear", align_corners=False)
            else:
                delta = None
        else:
            delta = None
            self._diag = {}
            self._scores = {}

        feats = self.tar(feats_a, feats_b)                   # 4 × (B, D, H_i, W_i)
        if delta is not None:
            assert feats[0].shape[-2:] == delta.shape[-2:], \
                f"[INJECT] 尺度不匹配 feats[0]={tuple(feats[0].shape[-2:])} delta={tuple(delta.shape[-2:])}"
            feats = list(feats)
            feats[0] = feats[0] + delta
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def switch_to_deploy(self):
        """把 TAR/DCR 的训练多分支折叠为单卷积。

        **CAACP-HQKV 保留在部署图里**，理由（必须写进论文，不能含糊）：
        它是**推理期真实执行的算子**（和 Run1–Run7 的 CAACP 一样），而且
        `train↔deploy` 折叠门要求两条图的前向**逐位一致**——若部署时把 HQKV 删掉，
        `feats[0]` 就会少一项，折叠门必然失败。
        ⇒ Run8 的"轻量"主张只能来自 **HQKV 自身的参数很小**（见 `param_breakdown()`），
        不能声称它零推理代价。
        """
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self
