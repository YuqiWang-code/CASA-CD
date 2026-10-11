"""CASA-TViM-STRNet：CAACP-SS2D 主线架构（TinyViM-S-Slim + CAACP + TAR/DCR）。

数据流（调研文档 §10，2B 拼接 siamese，BN 共享 batch 统计）：
    [A;B] (2B) -> TinyViM-S-Slim shared
        stem -> F1(1/4,48) -> F2(1/8,64) -> stage2 prefix(1/16,168)
        -> [CAACP 共享 score] -> stage2 final TViM -> F3(1/16,168)
        -> slim stage4 -> F4(1/32,224)
    per-time [F1,F2,F3,F4] -> MultiScaleTAR(encoder_dims=(48,64,168,224), D=96)
        -> DCRDecoder(D=96) -> head Conv1x1 -> bilinear -> sigmoid

变体（caacp × score_mode × residual_mode × frh × fs_tar × rep_mode，共享同一实现）：
    A0_TVIM_PLAIN: caacp=0, plain    A1_CAACP: caacp=1, plain
    A2_STR:        caacp=0, full     M1_FULL:  caacp=1, full
    Run2 E1_CP_CAACP: caacp=1, score_mode=cp, frh=0
    Run2 E2_FRH:     caacp=1, score_mode=rank, frh=1
    Run2 E3_CP_FRH:  caacp=1, score_mode=cp, frh=1
    Run3 E4_RA_CAACP: caacp=1, residual_mode=avg_anchor（residual 锚定 c_avg）
    Run3 E5_FS_TAR:   caacp=1, fs_tar=1（stage1 TemporalRepFine3x3）
    Run4 E6_FET1:     caacp=1, fine_tap=1（1/4 细尺度证据 1×1 可折叠旁路 → DCR refine 之后相加）

RNG 纪律：head 先于 TAR/DCR aux 构造；CAACP 只有 zero-init β（无 RNG 消耗）；
FRH gamma zero-init（无 RNG），dw3 kaiming / aux skip_init；
FS-TAR 的 aux 分支 skip_init + zero-init（与 TemporalRep1x1 同流）；
FET1 **最后构造**且用 fork_rng + 固定局部种子（不消耗全局随机流，保证开关 fine_tap 时
既有模块初始化逐位相同）。
deploy：switch_to_deploy() 折叠 TAR/DCR/FRH/FET1；CAACP 是推理期真实模块。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tinyvim_s_slim import TinyViMSlim
from model.layers.caacp_ss2d import change_score_cosine_2d, rank_normalize_2d
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder
from model.str_fine_head import STRFineHead
from model.str_fine_tap import FineEvidenceTap1x1, FET_FORM, FET_FUSE, FET_SOURCE
from model.str_hf_edge import (HF_EDGE_FORM, HF_EDGE_FORMS, HF_EDGE_LAMBDA_DEFAULT,
                               HF_EDGE_TAP_KEYS, HF_EDGE_TAPS, build_hf_edge_head)

ENCODER_DIMS = (48, 64, 168, 224)
STR_DIM = 96


class CASATViMSTRNet(nn.Module):
    def __init__(self, tinyvim_pretrained_path, caacp=True, rep_mode="full",
                 str_dim=STR_DIM, caacp_score_mode="rank", frh=False,
                 caacp_residual_mode="current", fs_tar=False, fine_tap=False,
                 hf_edge=False, hf_edge_form="none", hf_edge_taps="1_4_1_8",
                 hf_edge_lambda=HF_EDGE_LAMBDA_DEFAULT):
        super().__init__()
        assert rep_mode in ("plain", "full")
        assert caacp_score_mode in ("rank", "cp")
        assert caacp_residual_mode in ("current", "avg_anchor")
        self.caacp = caacp
        self.rep_mode = rep_mode
        self.str_dim = str_dim
        self.caacp_score_mode = caacp_score_mode
        self.caacp_residual_mode = caacp_residual_mode
        self.frh = frh
        self.fs_tar = fs_tar
        self.fine_tap = bool(fine_tap)

        # ---- Run7 HFAE（fail-closed：开关与形态必须自洽）
        self.hf_edge = bool(hf_edge)
        if self.hf_edge:
            assert hf_edge_form in ("aux_only", "aux_inject"), \
                f"[ARCH-MISMATCH] hf_edge=1 requires hf_edge_form in aux_only|aux_inject, got {hf_edge_form}"
            assert not self.fine_tap, "[ARCH-MISMATCH] hf_edge and fine_tap are mutually exclusive"
            assert hf_edge_taps in HF_EDGE_TAPS, f"unknown hf_edge_taps {hf_edge_taps}"
        else:
            assert hf_edge_form in ("none", None), \
                f"[ARCH-MISMATCH] hf_edge=0 requires hf_edge_form=none, got {hf_edge_form}"
        self.hf_edge_form = hf_edge_form if self.hf_edge else "none"
        self.hf_edge_taps = hf_edge_taps if self.hf_edge else "none"
        self.hf_edge_lambda = float(hf_edge_lambda)
        self.hf_edge_form_tag = HF_EDGE_FORM if self.hf_edge else None
        self._hf_taps = {}
        self._hf_handles = []
        self._hf_edge_logits = None
        self._hf_edge_feat = None

        # R4 sidecar 标识（fine_tap=0 时为 None，保持旧行为逐位不变）
        self.fine_tap_form = FET_FORM if self.fine_tap else None
        self.fine_tap_source = FET_SOURCE if self.fine_tap else None
        self.fine_tap_fuse = FET_FUSE if self.fine_tap else None

        self.encoder = TinyViMSlim(pretrained_path=tinyvim_pretrained_path, caacp=caacp,
                                   caacp_score_mode=caacp_score_mode,
                                   caacp_residual_mode=caacp_residual_mode)
        if self.encoder.caacp_op is not None:
            assert self.encoder.caacp_op.score_mode == caacp_score_mode
            assert self.encoder.caacp_op.residual_mode == caacp_residual_mode

        # head 先构造（STR RNG 纪律）；CAACP 已在 encoder 内（β zero-init，无 RNG）
        if frh:
            self.head = STRFineHead(str_dim)
        else:
            self.head = nn.Conv2d(str_dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=ENCODER_DIMS, dim=str_dim,
            use_temporal_aux=(rep_mode == "full"),
            use_dcr_aux=(rep_mode == "full"), use_residual=True,
            fine_stage1=fs_tar,
        )
        self.decoder = DCRDecoder(dim=str_dim, use_aux=(rep_mode == "full"), use_residual=True)

        # R4-FET1：**最后构造**，局部 RNG 初始化（不推进全局随机流）
        if self.fine_tap:
            assert str_dim == 96 and ENCODER_DIMS[0] == 48, "FET1 假设 C=48(1/4) → D=96"
            self.fine_evidence_tap = FineEvidenceTap1x1(
                in_ch=ENCODER_DIMS[0], out_ch=str_dim, deploy=False)
        else:
            self.fine_evidence_tap = None

        # R7-HFAE：同样**最后构造**；HFEdgeHead 内部用 fork_rng + 固定局部种子，
        # 不推进全局随机流 ⇒ 开关 hf_edge 时既有模块初始化逐位相同（T2 验证）。
        if self.hf_edge:
            assert str_dim == 96, "HFAE 假设 D=96（注入维度）"
            self.hf_edge_head = build_hf_edge_head(
                taps=self.hf_edge_taps, form=self.hf_edge_form, inject_dim=str_dim)
            self._register_hf_hooks()
        else:
            self.hf_edge_head = None

    # ------------------------------------------------------------------ R7 HFAE
    def _hf_hook_targets(self):
        return [self.encoder.network[n][b].op.local_conv
                for (n, b, _c, _r) in HF_EDGE_TAPS[self.hf_edge_taps]]

    def _ensure_hf_hooks(self):
        """幂等地确保 tap hook 挂在**当前**模块上。

        存在的理由：`copy.deepcopy(model)` 会复制 nn.Module 树，但**不会**复制挂在原模块上的
        forward hook —— 而项目的 D0/eval 审计路径正是 `deepcopy` + `switch_to_deploy`。
        因此每次 forward 前做一次轻量检查：当前模块上若没有我们的 hook 就补挂。

        R7 hotfix（2026-10-11，详见预注册 §7.7）：标记必须打在 **hook 函数**上——
        `nn.Module._forward_hooks` 的 value 是 hook 函数本身，而 `register_forward_hook()`
        返回的 `RemovableHandle` 是**另一个对象**、并不存放于该 dict。此前误把标记打在 handle 上，
        于是这里的 `attached` 恒为 False，**每次 forward 都重复注册 1 个 hook/模块**：hook 数随
        forward 数无界增长（含每 epoch 的全测试集验证 ⇒ 80K 步约 20 万个），单步耗时按
        O(累计 forward) 线性上升（实测每 hook ≈ 2.2 μs/forward），波二四库因此劣化约 4×。
        该缺陷只影响 `hf_edge=1` 的 run（`hf_edge=0` 在下方立即 return ⇒ CTRL 不受影响）。

        每个 handle 与目标模块成对保存，`switch_to_deploy()` 只移除属于**本模型**的 hook，
        避免 deepcopy 的副本误删原模型上的 hook。
        """
        if not self.hf_edge or self.hf_edge_head is None:
            return
        keys = HF_EDGE_TAP_KEYS[self.hf_edge_taps]
        for key, mod in zip(keys, self._hf_hook_targets()):
            mine = [fn for fn in getattr(mod, "_forward_hooks", {}).values()
                    if getattr(fn, "_hfae_key", None) == key]
            # fail-closed 不变式：同一 tap 模块上属于本 key 的 hook 至多 1 个。
            # 若幂等检查再被写坏，这里在第 2 次 forward 就抛错，而不是静默把训练拖慢 N 倍
            # （该缺陷曾静默运行 7 小时 / 21 万步，见预注册 §7.7）。
            # 该判据对 deepcopy 路径安全：副本的模块上不带任何我们的 hook。
            if len(mine) > 1:
                raise RuntimeError(
                    f"[HFAE] hook 幂等性被破坏：tap {key} 上挂了 {len(mine)} 个 hook"
                    "（应恰好 1 个；见预注册 §7.7）")
            if mine:
                continue
            fn = (lambda k: (lambda m, i, o: self._hf_taps.__setitem__(k, o)))(key)
            fn._hfae_key = key
            self._hf_handles.append((mod.register_forward_hook(fn), mod))

    def _remove_hf_hooks(self):
        """只移除仍然挂在本模型模块上的 hook（deepcopy 副本不会误删原模型的）。"""
        for h, mod in self._hf_handles:
            try:
                if getattr(h, "id", None) in getattr(mod, "_forward_hooks", {}):
                    h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._hf_handles = []

    def _register_hf_hooks(self):
        """构造期注册（后续由 `_ensure_hf_hooks()` 保证幂等）。"""
        self._hf_handles = []
        self._ensure_hf_hooks()

    def hf_edge_hook_counts(self):
        """诊断（T12 回归测试用）：每个 tap 模块上属于本模型的 hook 数，应恒为 1。

        R7 hotfix §7.7：该值曾随 forward 次数无界增长（每次 forward +1/模块），
        是波二 4× 减速的直接原因；本访问器让"幂等性"成为可断言、可回归的量。
        """
        if not self.hf_edge:
            return {}
        keys = HF_EDGE_TAP_KEYS[self.hf_edge_taps]
        return {key: sum(1 for fn in getattr(mod, "_forward_hooks", {}).values()
                         if getattr(fn, "_hfae_key", None) == key)
                for key, mod in zip(keys, self._hf_hook_targets())}

    def _run_hf_edge(self):
        """从 hook 缓存取高频 tap → 边缘头。

        返回 `(edge_logits (B,1,h,w), edge_feat (B,mid,h,w))`——双时相在头内**配对**，
        因此是 B 尺寸（每对样品一张变化边缘图）。
        """
        keys = HF_EDGE_TAP_KEYS[self.hf_edge_taps]
        taps = []
        for k in keys:
            t = self._hf_taps.get(k)
            if t is None:
                raise RuntimeError(f"[HFAE] tap {k} missing: forward hooks did not fire")
            taps.append(t)
        logits, feat = self.hf_edge_head(taps)
        self._hf_taps.clear()                       # 释放对 autograd 图的引用
        return logits, feat

    def aux_edge_logits(self):
        """训练期边缘 logits（`train.py` 用它算辅助损失；步间不保留旧值）。"""
        return self._hf_edge_logits

    # ---- R7 HFAE 机制自检：辅助损失是否真的回流到「高频分支」
    def _hf_tap_modules(self):
        if not self.hf_edge:
            return []
        return [self.encoder.network[n][b].op.local_conv
                for (n, b, _c, _r) in HF_EDGE_TAPS[self.hf_edge_taps]]

    def hf_edge_tap_grad_norm(self):
        """被 tap 的高频分支（`local_conv`）上辅助损失的梯度 L2 范数。

        > 0 才说明边缘监督在**重塑表征**；若恒为 0，则本机制退化为"旁路头"，
        与已被否证的无监督注入等价 ⇒ T5 的判定依据。
        """
        tot = 0.0
        for mod in self._hf_tap_modules():
            for p in mod.parameters():
                if p.grad is not None:
                    tot += float(p.grad.detach().pow(2).sum())
        return tot ** 0.5

    def hf_edge_tap_grad_nonzero(self):
        n = 0
        for mod in self._hf_tap_modules():
            for p in mod.parameters():
                if p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0:
                    n += 1
        return n

    def hf_edge_head_grad_norm(self):
        if self.hf_edge_head is None:
            return 0.0
        tot = sum(float(p.grad.detach().pow(2).sum())
                  for p in self.hf_edge_head.parameters() if p.grad is not None)
        return tot ** 0.5

    def hf_edge_report(self):
        if not self.hf_edge or self.hf_edge_head is None:
            return None
        rep = self.hf_edge_head.param_report()
        rep.update({"form": self.hf_edge_form, "taps": self.hf_edge_taps,
                    "lambda": self.hf_edge_lambda, "tag": self.hf_edge_form_tag,
                    "deploy_dropped": bool(self.hf_edge_form == "aux_only"
                                           and not self._hf_handles
                                           and self.hf_edge_head is None)})
        return rep

    def forward(self, pre, post, label=None):
        if self.hf_edge and self.hf_edge_head is not None:
            self._ensure_hf_hooks()                             # deepcopy 后自动补挂
            self._hf_taps.clear()                               # 步间不持有旧图
            self._hf_edge_logits = None
            self._hf_edge_feat = None
        x2b = torch.cat([pre, post], dim=0)                     # (2B,3,256,256)
        f1, f2, x = self.encoder.forward_stem_s2_prefix(x2b)    # ← tap 在此处被 hook 捕获

        if self.hf_edge and self.hf_edge_head is not None:
            self._hf_edge_logits, self._hf_edge_feat = self._run_hf_edge()

        if self.caacp and self.encoder.caacp_op is not None:
            xa, xb = x.chunk(2, dim=0)
            with torch.no_grad():
                s_abs = change_score_cosine_2d(xa, xb)          # (B,H,W) in [0,2]
                s_rank = rank_normalize_2d(s_abs)
                s_rank2b = torch.cat([s_rank, s_rank], dim=0)   # A/B 共享权重
                s_abs2b = (torch.cat([s_abs, s_abs], dim=0)
                           if self.caacp_score_mode == "cp" else None)
            self.encoder.caacp_op.set_pair_score(s_rank2b, abs_score=s_abs2b)
        x = self.encoder.forward_caacp_block(x)
        f3 = self.encoder.norm4(x)
        f4 = self.encoder.forward_stage4(x)

        f1a, f1b = f1.chunk(2, dim=0)
        f2a, f2b = f2.chunk(2, dim=0)
        f3a, f3b = f3.chunk(2, dim=0)
        f4a, f4b = f4.chunk(2, dim=0)

        feats = self.tar([f1a, f2a, f3a, f4a], [f1b, f2b, f3b, f4b])
        if (self.hf_edge and self.hf_edge_form == "aux_inject"
                and self._hf_edge_feat is not None):
            # 零初始化注入 ⇒ epoch-0 逐位恒等；注入到 DCR 的最小尺度输入（1/4）
            assert self.hf_edge_head.inject is not None, "[HFAE] aux_inject requires inject proj"
            feats = list(feats)
            feats[0] = feats[0] + self.hf_edge_head.inject(self._hf_edge_feat)
        x = self.decoder(feats)                     # 原始 DCR.refine 输出 [B,96,64,64]
        if self.fine_tap:
            # R4-FET1：refine **之后**相加（不进入 SiLU/RepLocalBlock，保持 DCR 训练路径不变）
            x = x + self.fine_evidence_tap(f1a, f1b)
        logits = self.head(x)       # frh: 内部 64² -> 128²；plain: 64²
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        if self.frh:
            self.head.switch_to_deploy()
        if self.fine_tap:
            self.fine_evidence_tap.switch_to_deploy()
        if self.hf_edge and self.hf_edge_head is not None:
            self._remove_hf_hooks()
            if self.hf_edge_form == "aux_only":
                # 训练期深监督 ⇒ 推理图不含边缘头，deploy 参数与基线**完全相同**
                self.hf_edge_head = None
            else:
                self.hf_edge_head.drop_supervision_head()   # 只丢监督专用 edge_out
            self._hf_taps.clear()
            self._hf_edge_logits = None
            self._hf_edge_feat = None
        return self

    # ------------------------------------------------------------------ R4 diag
    def fet_gamma(self):
        if self.fine_tap and self.fine_evidence_tap is not None and not self.fine_evidence_tap.deploy:
            return float(self.fine_evidence_tap.gamma.detach().cpu().item())
        return None

    def fet_deploy_conv_shape(self):
        if self.fine_tap and self.fine_evidence_tap is not None and self.fine_evidence_tap.deploy:
            w = self.fine_evidence_tap.fused.weight
            return [int(w.shape[0]), int(w.shape[1]), int(w.shape[2]), int(w.shape[3])]
        return None


    def caacp_beta(self):
        if self.encoder.caacp_op is not None:
            return float(self.encoder.caacp_op.beta.detach().cpu().item())
        return None

    def caacp_score_delta(self):
        if self.encoder.caacp_op is not None:
            return float(self.encoder.caacp_op._score_delta)
        return None

    def caacp_abs_mean(self):
        if self.encoder.caacp_op is not None:
            return self.encoder.caacp_op.abs_mean()
        return None

    def caacp_weight_entropy(self):
        if self.encoder.caacp_op is not None:
            return self.encoder.caacp_op.weight_entropy()
        return None
