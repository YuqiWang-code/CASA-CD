"""CASASTRNet：主线重构最终架构（SHViT-S1 truncated + CASAA@1/16 + TAR + DCR）。

数据流（本地实施注意事项 §44 修正版）：
    [A;B] (2B) -> shared pretrained SHViT-S1 truncated
        stem s2 -> s4(F1,C32) -> s8(F2,C64) -> s16
        -> blocks1 -> F3 (1/16, C128)
        -> split A/B -> [CASAA pair @1/16, score 来自 F2] -> F3'
        -> concat back 2B -> blocks2 -> F4 (1/32, C224)
    per-time feats [F1,F2,F3',F4] -> MultiScaleTAR(encoder_dims=(32,64,128,224))
        -> DCRDecoder -> head Conv1x1(D->1) -> bilinear -> sigmoid

变体（attn_mode × rep_mode，共享同一实现）：
    A0: none   × plain     A1: change × plain     C1: full × plain
    A2: none   × full      M1: change × full      C2: content × plain

RNG 纪律：head 先于 TAR/DCR 的 aux 分支构造；CASAA 在共享模块之后构造。
deploy：switch_to_deploy() 只折叠 TAR/DCR；CASAA 是推理期真实模块。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.shvit_s1_trunc import SHViTS1Truncated
from model.layers.casaa_hier import HierCASAA
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder

ENCODER_DIMS = (32, 64, 128, 224)


class CASASTRNet(nn.Module):
    def __init__(self, pretrained_path, attn_mode="none", rep_mode="full",
                 str_dim=160, keep_ratio=0.25, change_share=0.5):
        super().__init__()
        assert attn_mode in ("none", "full", "content", "change")
        assert rep_mode in ("plain", "full")
        self.attn_mode = attn_mode
        self.rep_mode = rep_mode
        self.str_dim = str_dim
        self.keep_ratio = keep_ratio
        self.change_share = change_share

        self.encoder = SHViTS1Truncated(pretrained_path=pretrained_path)

        # head 先构造（STR RNG 纪律）；CASAA 在共享模块之后
        self.head = nn.Conv2d(str_dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=ENCODER_DIMS, dim=str_dim,
            use_temporal_aux=(rep_mode == "full"),
            use_dcr_aux=(rep_mode == "full"), use_residual=True,
        )
        self.decoder = DCRDecoder(dim=str_dim, use_aux=(rep_mode == "full"), use_residual=True)

        self.casaa = None
        if attn_mode != "none":
            self.casaa = HierCASAA(dim=128, qk_dim=16, mode=attn_mode,
                                   keep_ratio=keep_ratio, change_share=change_share)

    def forward(self, pre, post, label=None):
        x2b = torch.cat([pre, post], dim=0)                      # (2B,3,256,256)
        f1, f2, f3 = self.encoder.forward_stem_blocks1(x2b)      # 2B 各 (2B,C,H,W)
        f3a, f3b = f3.chunk(2, dim=0)
        if self.casaa is not None:
            f2a, f2b = f2.chunk(2, dim=0)
            ya, yb = self.casaa(f3a, f3b, f2a, f2b)
            f3a = f3a + self.casaa.beta * ya
            f3b = f3b + self.casaa.beta * yb
        x2b = torch.cat([f3a, f3b], dim=0)
        f4 = self.encoder.forward_blocks2(x2b)

        f1a, f1b = f1.chunk(2, dim=0)
        f2a, f2b = f2.chunk(2, dim=0)
        f4a, f4b = f4.chunk(2, dim=0)

        feats_a = [f1a, f2a, f3a, f4a]
        feats_b = [f1b, f2b, f3b, f4b]
        feats = self.tar(feats_a, feats_b)
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    def train(self, mode=True):
        super().train(mode)
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self

    def shared_state_dict(self):
        """不含 rep-aux 分支键的共享模块 state dict（A1/M1 初始化逐位对拍用）。
        rep aux 分支（sum/diff/aux convs/BNs）在 plain 中不存在，因此这里只比较
        两者都存在的主分支。为简单起见返回 backbone+casaa+head 的键（TAR/DCR
        主分支由 smoke 的 aux-零初始化 epoch-0 一致性覆盖）。
        """
        keep_prefix = ("encoder.", "casaa.", "head.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep_prefix)}
