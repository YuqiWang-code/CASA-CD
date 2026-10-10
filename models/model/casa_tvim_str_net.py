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

ENCODER_DIMS = (48, 64, 168, 224)
STR_DIM = 96


class CASATViMSTRNet(nn.Module):
    def __init__(self, tinyvim_pretrained_path, caacp=True, rep_mode="full",
                 str_dim=STR_DIM, caacp_score_mode="rank", frh=False,
                 caacp_residual_mode="current", fs_tar=False, fine_tap=False):
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

    def forward(self, pre, post, label=None):
        x2b = torch.cat([pre, post], dim=0)                     # (2B,3,256,256)
        f1, f2, x = self.encoder.forward_stem_s2_prefix(x2b)

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
