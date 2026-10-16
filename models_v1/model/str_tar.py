"""TAR bridge for STRFusion (Run1): Temporal Algebraic Re-parameterization.

Ported from STR-RepNet `models/changedetection/models/tar.py` (Run2 generation).
Not ported: BOTR reverse-concat branch (Run5, pre-registered FAIL on STR side).

Each scale projects the bi-temporal pair (pre, post) with
Concat + Sum + signed-Diff branches, each with its own BatchNorm (aux branches
zero-init, BN gamma=beta=0), collapsing to a single 1x1 temporal projection at
deploy. The signed diff is Q - P (direction-sensitive change basis), NOT abs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_reparam import fold_conv_bn, switch_module_to_deploy
from model.str_dcr import RepLocalBlock


def pad_center_1x1_to_3x3(w):
    """(D, C, 1, 1) -> (D, C, 3, 3)：1×1 kernel 中心嵌入 3×3。"""
    return F.pad(w, (1, 1, 1, 1))


class TemporalRep1x1(nn.Module):
    """Bi-temporal 1x1 re-parameterization: concat + sum + signed-diff (each BN) -> single 1x1."""

    def __init__(self, in_channels, out_channels, use_aux=True, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.proj = nn.Conv2d(2 * in_channels, out_channels, 1, bias=True)
        else:
            self.proj_c = nn.Conv2d(2 * in_channels, out_channels, 1, bias=False)
            self.bn_c = nn.BatchNorm2d(out_channels)
            nn.init.kaiming_normal_(self.proj_c.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                # skip_init: aux conv CONSTRUCTION itself consumes global RNG
                # (reset_parameters kaiming draw) even though the weights are
                # zeroed right after; skipping it keeps the C0 (no aux) / M1 (aux)
                # global-RNG streams aligned so all shared modules init identically.
                from torch.nn.utils import skip_init
                self.proj_s = skip_init(nn.Conv2d, in_channels, out_channels, 1, bias=False)
                self.bn_s = nn.BatchNorm2d(out_channels)
                self.proj_d = skip_init(nn.Conv2d, in_channels, out_channels, 1, bias=False)
                self.bn_d = nn.BatchNorm2d(out_channels)
                nn.init.zeros_(self.proj_s.weight)
                nn.init.zeros_(self.proj_d.weight)

    def forward(self, P, Q):
        if self.deploy:
            return self.proj(torch.cat([P, Q], dim=1))
        y = self.bn_c(self.proj_c(torch.cat([P, Q], dim=1)))
        if self.use_aux:
            y = y + self.bn_s(self.proj_s(P + Q)) + self.bn_d(self.proj_d(Q - P))
        return y

    def get_equivalent_kernel_bias(self):
        C = self.in_channels
        W_c, b_c = fold_conv_bn(self.proj_c.weight, None, self.bn_c)
        W_P = W_c[:, :C]
        W_Q = W_c[:, C:]
        b = b_c
        if self.use_aux:
            W_s, b_s = fold_conv_bn(self.proj_s.weight, None, self.bn_s)
            W_d, b_d = fold_conv_bn(self.proj_d.weight, None, self.bn_d)
            W_P = W_P + W_s - W_d
            W_Q = W_Q + W_s + W_d
            b = b + b_s + b_d
        W = torch.cat([W_P, W_Q], dim=1)
        return W, b

    def branch_stats(self):
        s = {"concat": self.proj_c.weight.norm().item()}
        if self.use_aux:
            s["sum"] = self.proj_s.weight.norm().item()
            s["diff"] = self.proj_d.weight.norm().item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.proj = nn.Conv2d(2 * self.in_channels, self.out_channels, 1, bias=True)
        self.proj.weight.data = kernel.float()
        self.proj.bias.data = bias.float()
        self.deploy = True
        for name in ("proj_s", "proj_d", "bn_c", "bn_s", "bn_d", "proj_c"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class TemporalRepFine3x3(nn.Module):
    """Fine-scale bi-temporal 3x3 re-parameterization（Run3 E5 FS-TAR，只给 stage1 1/4 用）。

    训练图：
        main  : [P,Q] concat -> 1x1 + BN            (kaiming init)
        aux_s : (P+Q) -> 1x1 + BN                   (zero-init)
        aux_d : (Q-P) -> 3x3 Conv + BN              (zero-init；tiny-change spatial-temporal branch)
    输出 = main + aux_s + aux_d（stage1 在 1/4 尺度先提取双时相空间邻域差异证据，
    small target 在 1/4 仍有 <16 feature cells，尚未退化为 sub-token）。

    deploy 折叠为单个 Conv2d(2C -> D, 3, padding=1)：
        W_P3 = pad_center(W_c[:, :C]) + pad_center(W_s) - W_d
        W_Q3 = pad_center(W_c[:, C:]) + pad_center(W_s) + W_d
    即 signed-diff 3×3 加到 Q half、负加到 P half；全 FP64 组合、单次 FP32 cast。
    """

    def __init__(self, in_channels, out_channels, use_aux=True, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.proj = nn.Conv2d(2 * in_channels, out_channels, 3, 1, 1, bias=True)
        else:
            self.proj_c = nn.Conv2d(2 * in_channels, out_channels, 1, bias=False)
            self.bn_c = nn.BatchNorm2d(out_channels)
            nn.init.kaiming_normal_(self.proj_c.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                # skip_init：aux 构造不消耗全局 RNG（与 TemporalRep1x1 约定一致）
                from torch.nn.utils import skip_init
                self.proj_s = skip_init(nn.Conv2d, in_channels, out_channels, 1, bias=False)
                self.bn_s = nn.BatchNorm2d(out_channels)
                self.proj_d = skip_init(nn.Conv2d, in_channels, out_channels, 3, 1, 1, bias=False)
                self.bn_d = nn.BatchNorm2d(out_channels)
                nn.init.zeros_(self.proj_s.weight)
                nn.init.zeros_(self.proj_d.weight)

    def forward(self, P, Q):
        if self.deploy:
            return self.proj(torch.cat([P, Q], dim=1))
        y = self.bn_c(self.proj_c(torch.cat([P, Q], dim=1)))
        if self.use_aux:
            y = y + self.bn_s(self.proj_s(P + Q)) + self.bn_d(self.proj_d(Q - P))
        return y

    def get_equivalent_kernel_bias(self):
        C = self.in_channels
        W_c, b_c = fold_conv_bn(self.proj_c.weight, None, self.bn_c)   # (D,2C,1,1) fp64
        W_P = pad_center_1x1_to_3x3(W_c[:, :C])
        W_Q = pad_center_1x1_to_3x3(W_c[:, C:])
        b = b_c
        if self.use_aux:
            W_s, b_s = fold_conv_bn(self.proj_s.weight, None, self.bn_s)   # (D,C,1,1)
            W_d, b_d = fold_conv_bn(self.proj_d.weight, None, self.bn_d)   # (D,C,3,3)
            W_P = W_P + pad_center_1x1_to_3x3(W_s) - W_d
            W_Q = W_Q + pad_center_1x1_to_3x3(W_s) + W_d
            b = b + b_s + b_d
        W = torch.cat([W_P, W_Q], dim=1)          # (D, 2C, 3, 3)
        return W, b

    def branch_stats(self):
        s = {"concat": self.proj_c.weight.norm().item()}
        if self.use_aux:
            s["sum"] = self.proj_s.weight.norm().item()
            s["diff3x3"] = self.proj_d.weight.norm().item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.proj = nn.Conv2d(2 * self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        self.proj.weight.data = kernel.float()
        self.proj.bias.data = bias.float()
        self.deploy = True
        for name in ("proj_s", "proj_d", "bn_c", "bn_s", "bn_d", "proj_c"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class TARStage(nn.Module):
    """One scale: TemporalRep(1x1 | Fine3x3) -> SiLU -> RepLocalBlock."""

    def __init__(self, in_channels, dim, use_temporal_aux=True, use_dcr_aux=True,
                 use_residual=True, deploy=False, fine_3x3=False):
        super().__init__()
        if fine_3x3:
            self.temporal = TemporalRepFine3x3(in_channels, dim, use_aux=use_temporal_aux, deploy=deploy)
        else:
            self.temporal = TemporalRep1x1(in_channels, dim, use_aux=use_temporal_aux, deploy=deploy)
        self.block = RepLocalBlock(dim, use_aux=use_dcr_aux, use_residual=use_residual, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, P, Q):
        x = self.act(self.temporal(P, Q))
        return self.block(x)

    def switch_to_deploy(self):
        self.temporal.switch_to_deploy()
        self.block.switch_to_deploy()
        return self


class MultiScaleTAR(nn.Module):
    """Four-scale TAR bridge: encoder_dims (192,192,192,192) -> dim.

    Input scale order matches STR convention: stage1 (finest, 64x64) .. stage4
    (coarsest, 8x8); output list [t1, t2, t3, t4] feeds DCRDecoder.
    """

    def __init__(self, encoder_dims=(192, 192, 192, 192), dim=160,
                 use_temporal_aux=True, use_dcr_aux=True, use_residual=True, deploy=False,
                 fine_stage1=False):
        super().__init__()
        self.stage1 = TARStage(encoder_dims[0], dim, use_temporal_aux, use_dcr_aux,
                               use_residual, deploy, fine_3x3=fine_stage1)
        self.stage2 = TARStage(encoder_dims[1], dim, use_temporal_aux, use_dcr_aux,
                               use_residual, deploy)
        self.stage3 = TARStage(encoder_dims[2], dim, use_temporal_aux, use_dcr_aux,
                               use_residual, deploy)
        self.stage4 = TARStage(encoder_dims[3], dim, use_temporal_aux, use_dcr_aux,
                               use_residual, deploy)

    def forward(self, pre_feats, post_feats):
        t1 = self.stage1(pre_feats[0], post_feats[0])
        t2 = self.stage2(pre_feats[1], post_feats[1])
        t3 = self.stage3(pre_feats[2], post_feats[2])
        t4 = self.stage4(pre_feats[3], post_feats[3])
        return [t1, t2, t3, t4]

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
