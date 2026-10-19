"""CASA-VMamba-STGeo（探索线 Exp1）：Run8 主干/编解码器 + SPD(2)/H² 结构张量测地证据。

设计纪律（与 Run1–Run8 完全一致，目的是让 **CTRL_R8** 直接充当同期对照）：
  1. 共享路径（VMamba-Tiny / MultiScaleTAR / DCRDecoder / head）与
     `models/model/casa_vmamba_net.py` 的 `hqkv=0` 分支**逐行同构**；
  2. ST-GEO 分支**最后构造**且在 `fork_rng` 内初始化 ⇒ 不推进全局随机流
     ⇒ epoch-0 时 `stgeo=none` 与 `stgeo=inject` 的前向**逐位相同**（dry-run 断言）；
  3. γ 零初始化 ⇒ epoch-0 增量恒为 0，与 Run8 `CTRL_R8`（同协议同 seed）逐位一致；
  4. ST-GEO 是**推理期真实算子**（保留在部署图里），不得声称零推理代价。

数据流：
    pre/post (B,3,256,256)
      ├─ cat(dim=0) → VMamba-Tiny(2B) → f1(1/4,96) f2(1/8,192) f3(1/16,384) f4(1/32,768)
      ├─ ST-GEO（原始图像上的 SPD(2) 测地证据，全分辨率 → 池化到 1/4）
      │     → γ · Stem(...)  ∈ (B,192,64,64)
      └─ feats = TAR(fA, fB)  → feats[0] += γ·Stem(...)  → DCR → head → bilinear → sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.st_geodesic import STGEO_FORM, STGEO_SEED, STGeoEvidence
from model.layers.stgeo_reliable import (STGEO_REL_FORM, STGEO_REL_SEED,
                                        STGeoReliable)
from model.layers.stgeo_sup import STGEO_SUP_FORM, STGEO_SUP_SEED, STGeoFeatureSup
from model.str_dcr import DCRDecoder
from model.str_tar import MultiScaleTAR
from model.vmamba_tiny import VMambaBackbone, build_vmamba_tiny

ENCODER_DIMS = (96, 192, 384, 768)
STR_DIM = 192
STGEO_FORMS = ("none", "inject", "sup", "rel")


class CASAVMambaSTGeoNet(nn.Module):
    def __init__(self, vmamba_pretrained_path=None, str_dim=STR_DIM,
                 stgeo="inject", stgeo_sigmas=(1.0, 2.0, 4.0), stgeo_mid=32,
                 stgeo_dim=192, stgeo_pool="max_shape", stgeo_xscale=False,
                 stgeo_sup_small=16, stgeo_sup_lambda=0.2,
                 stgeo_geom="hyperbolic", stgeo_detail_only=True,
                 rep_mode="full", fs_tar=False,
                 vmamba_cfg=None, hqkv=False, hf_edge=False, hf_edge_form="none"):
        super().__init__()
        assert stgeo in STGEO_FORMS, stgeo
        assert rep_mode in ("plain", "full")
        assert not hqkv and not hf_edge, (
            "[ARCH] casa_vmamba_stgeo 是独立探索架构：不启用 Run8 HQKV / Run7 HFAE "
            "（要对比它们请用各自的原架构，避免实验臂混叠）")
        self.stgeo = stgeo
        self.str_dim = int(str_dim)
        self.rep_mode = rep_mode
        self.fs_tar = bool(fs_tar)

        # ---- 与 casa_vmamba_net 同序构造（RNG 账本一致）
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

        # ---- ST-GEO（**最后构造** + 内部 fork_rng）
        self.stgeo_rel = None
        if stgeo == "rel":
            # 方向 A：可靠性感知 + 尺度选择性修正（推理期真实算子）
            self.stgeo_op = None
            self.stgeo_sup = None
            self.stgeo_rel = STGeoReliable(in_ch=3, sigmas=stgeo_sigmas,
                                           mid=stgeo_mid, out_dim=stgeo_dim,
                                           geom=stgeo_geom,
                                           detail_only=stgeo_detail_only,
                                           tr_floor=1e-10, pool_mode="topk_mean",
                                           init_seed=STGEO_REL_SEED)
        elif stgeo == "sup":
            # B 形态：特征域几何 + GT 掩码监督；**不注入**，deploy 整支删除
            self.stgeo_op = None
            self.stgeo_sup = STGeoFeatureSup(in_ch=ENCODER_DIMS[0],
                                             small=stgeo_sup_small,
                                             lambda_=stgeo_sup_lambda,
                                             init_seed=STGEO_SUP_SEED)
        elif stgeo == "inject":
            self.stgeo_op = STGeoEvidence(in_ch=3, sigmas=stgeo_sigmas, mid=stgeo_mid,
                                          out_dim=stgeo_dim, form=stgeo,
                                          pool_mode=stgeo_pool, xscale_gate=stgeo_xscale,
                                          init_seed=STGEO_SEED)
            self.stgeo_sup = None
            self.stgeo_rel = None
        else:
            self.stgeo_op = None
            self.stgeo_sup = None
            self.stgeo_rel = None
        if not hasattr(self, "stgeo_rel"):
            self.stgeo_rel = None
        self._aux_logits = None

    # ------------------------------------------------------------------ 诊断
    def aux_change_logits(self):
        """训练期辅助变化 logits（`train_stgeo.py` 用它算 λ·BCE+Dice）。"""
        return self._aux_logits

    def stgeo_report(self):
        if self.stgeo_sup is not None:
            return self.stgeo_sup.param_report()
        if self.stgeo_rel is not None:
            return self.stgeo_rel.param_report()
        if self.stgeo_op is None:
            # deploy 后（sup 形态）支路已删除：返回带标记的报告，绝不抛错
            return {"form": STGEO_SUP_FORM, "dropped_at_deploy": True, "params": 0,
                    "lambda": None, "small": None, "in_ch": None}
        rep = self.stgeo_op.param_report()
        rep.update({"sigmas": list(self.stgeo_op.sigmas), "form_tag": STGEO_FORM})
        return rep

    def stgeo_diag(self):
        if self.stgeo_sup is not None:
            return self.stgeo_sup.score_stats()
        if self.stgeo_rel is not None:
            return self.stgeo_rel.diag()
        if self.stgeo_op is None:
            return {"note": "dropped_at_deploy"}      # deploy 后无诊断量，绝不抛错
        return self.stgeo_op.diag()

    def pretrain_load_stats(self):
        return self.encoder.load_stats()

    def param_breakdown(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        tar = sum(p.numel() for p in self.tar.parameters())
        dcr = sum(p.numel() for p in self.decoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        stg = 0
        if self.stgeo_op is not None:
            stg += sum(p.numel() for p in self.stgeo_op.parameters())
        if self.stgeo_sup is not None:
            stg += sum(p.numel() for p in self.stgeo_sup.parameters())
        if self.stgeo_rel is not None:
            stg += sum(p.numel() for p in self.stgeo_rel.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"backbone": enc, "stgeo": stg, "tar": tar, "dcr": dcr, "head": head,
                "total": total, "non_backbone": total - enc,
                "non_backbone_no_stgeo": total - enc - stg}

    # ------------------------------------------------------------------
    def forward(self, pre, post, label=None):
        # ST-GEO 证据（在原始图像上算，全分辨率 → 1/4）；`sup` 形态无此支路
        delta = None
        if self.stgeo_rel is not None:
            delta = self.stgeo_rel(pre, post, (pre.shape[-2] // 4, pre.shape[-1] // 4))
        if self.stgeo_op is not None:
            delta, _ = self.stgeo_op(pre, post,
                                     (pre.shape[-2] // 4, pre.shape[-1] // 4))

        x2b = torch.cat([pre, post], dim=0)
        outs = self.encoder.forward_features(x2b)
        del x2b
        feats_a, feats_b = [], []
        for t in outs:
            a, b = t.chunk(2, dim=0)
            feats_a.append(a)
            feats_b.append(b)
        del outs

        if self.stgeo_sup is not None:
            # 特征域几何监督（只影响梯度方向，不改变主输出）
            self._aux_logits = self.stgeo_sup(feats_a[0], feats_b[0],
                                             (pre.shape[-2], pre.shape[-1]))
        else:
            self._aux_logits = None

        feats = self.tar(feats_a, feats_b)
        if delta is not None:
            assert feats[0].shape[-2:] == delta.shape[-2:], \
                f"[STGEO-INJECT] 尺度不匹配 {tuple(feats[0].shape[-2:])} vs {tuple(delta.shape[-2:])}"
            feats = list(feats)
            feats[0] = feats[0] + delta
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def switch_to_deploy(self):
        """折叠 TAR/DCR。

        · `inject` 形态：ST-GEO 是推理期真实算子，**保留**（与 Run8 HQKV 同理）；
        · `sup` 形态：整支**删除**（Run7 HFAE `aux_only` 同口径）⇒ 部署图与 CTRL_R8 相同。
        """
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        if self.stgeo_sup is not None:
            self.stgeo_sup = None
            self._aux_logits = None
        return self
