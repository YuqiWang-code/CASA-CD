"""CASA-TViM-STRNet：CAACP-SS2D 主线架构（TinyViM-S-Slim + CAACP + TAR/DCR）。

数据流（调研文档 §10，2B 拼接 siamese，BN 共享 batch 统计）：
    [A;B] (2B) -> TinyViM-S-Slim shared
        stem -> F1(1/4,48) -> F2(1/8,64) -> stage2 prefix(1/16,168)
        -> [CAACP 共享 score] -> stage2 final TViM -> F3(1/16,168)
        -> slim stage4 -> F4(1/32,224)
    per-time [F1,F2,F3,F4] -> MultiScaleTAR(encoder_dims=(48,64,168,224), D=96)
        -> DCRDecoder(D=96) -> head Conv1x1 -> bilinear -> sigmoid

变体（caacp × rep_mode，共享同一实现）：
    A0_TVIM_PLAIN: caacp=0, plain    A1_CAACP: caacp=1, plain
    A2_STR:        caacp=0, full     M1_FULL:  caacp=1, full

RNG 纪律：head 先于 TAR/DCR aux 构造；CAACP 只有 zero-init β（无 RNG 消耗）。
deploy：switch_to_deploy() 只折叠 TAR/DCR；CAACP 是推理期真实模块。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tinyvim_s_slim import TinyViMSlim
from model.layers.caacp_ss2d import change_score_cosine_2d, rank_normalize_2d
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder

ENCODER_DIMS = (48, 64, 168, 224)
STR_DIM = 96


class CASATViMSTRNet(nn.Module):
    def __init__(self, tinyvim_pretrained_path, caacp=True, rep_mode="full",
                 str_dim=STR_DIM):
        super().__init__()
        assert rep_mode in ("plain", "full")
        self.caacp = caacp
        self.rep_mode = rep_mode
        self.str_dim = str_dim

        self.encoder = TinyViMSlim(pretrained_path=tinyvim_pretrained_path, caacp=caacp)

        # head 先构造（STR RNG 纪律）；CAACP 已在 encoder 内（β zero-init，无 RNG）
        self.head = nn.Conv2d(str_dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=ENCODER_DIMS, dim=str_dim,
            use_temporal_aux=(rep_mode == "full"),
            use_dcr_aux=(rep_mode == "full"), use_residual=True,
        )
        self.decoder = DCRDecoder(dim=str_dim, use_aux=(rep_mode == "full"), use_residual=True)

    def forward(self, pre, post, label=None):
        x2b = torch.cat([pre, post], dim=0)                     # (2B,3,256,256)
        f1, f2, x = self.encoder.forward_stem_s2_prefix(x2b)

        if self.caacp and self.encoder.caacp_op is not None:
            xa, xb = x.chunk(2, dim=0)
            with torch.no_grad():
                s = change_score_cosine_2d(xa, xb)              # (B,H,W)
                s = rank_normalize_2d(s)
                s2b = torch.cat([s, s], dim=0)                  # A/B 共享权重
            self.encoder.caacp_op.set_pair_score(s2b)
        x = self.encoder.forward_caacp_block(x)
        f3 = self.encoder.norm4(x)
        f4 = self.encoder.forward_stage4(x)

        f1a, f1b = f1.chunk(2, dim=0)
        f2a, f2b = f2.chunk(2, dim=0)
        f3a, f3b = f3.chunk(2, dim=0)
        f4a, f4b = f4.chunk(2, dim=0)

        feats = self.tar([f1a, f2a, f3a, f4a], [f1b, f2b, f3b, f4b])
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self

    def caacp_beta(self):
        if self.encoder.caacp_op is not None:
            return float(self.encoder.caacp_op.beta.detach().cpu().item())
        return None

    def caacp_score_delta(self):
        if self.encoder.caacp_op is not None:
            return float(self.encoder.caacp_op._score_delta)
        return None
