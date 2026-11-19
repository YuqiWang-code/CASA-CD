"""Four-scale temporal fusion with foldable concat, sum and signed-difference branches.

The ``plain`` variant keeps the same stage topology (temporal 1x1 -> local block)
but uses a single concat 1x1 convolution plus batch norm instead of the foldable
three-branch re-parameterization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_reparam import fold_conv_bn, switch_module_to_deploy
from model.str_dcr import PlainLocalBlock, RepLocalBlock


def embed_depthwise_kernel(weight, target):
    """Place a k x k depthwise kernel in the centre of a target x target kernel."""
    k = weight.shape[-1]
    if k == target:
        return weight
    offset = (target - k) // 2
    out = torch.zeros(weight.shape[0], weight.shape[1], target, target,
                      dtype=weight.dtype, device=weight.device)
    out[..., offset:offset + k, offset:offset + k] = weight
    return out


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
                # Keep auxiliary construction independent of the shared initialization stream.
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


class TARStage(nn.Module):
    """One scale: temporal 1x1 projection, SiLU and local refinement."""

    def __init__(self, in_channels, dim, use_temporal_aux=True, use_dcr_aux=True,
                 use_residual=True, deploy=False):
        super().__init__()
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


class PlainTemporal1x1(nn.Module):
    """Ablation: single concat 1x1 convolution plus batch norm, no auxiliary branches."""

    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.proj = nn.Conv2d(2 * in_channels, out_channels, 1, bias=deploy)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, P, Q):
        fused = self.proj(torch.cat([P, Q], dim=1))
        return fused if self.deploy else self.bn(fused)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = fold_conv_bn(self.proj.weight, None, self.bn)
        self.proj = nn.Conv2d(2 * self.in_channels, self.out_channels, 1, bias=True)
        self.proj.weight.data = kernel.float()
        self.proj.bias.data = bias.float()
        self.deploy = True
        if hasattr(self, "bn"):
            del self.bn
        return self


class PlainTARStage(nn.Module):
    """Ablation: one scale with plain temporal fusion and plain local refinement."""

    def __init__(self, in_channels, dim, deploy=False):
        super().__init__()
        self.temporal = PlainTemporal1x1(in_channels, dim, deploy=deploy)
        self.block = PlainLocalBlock(dim, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, P, Q):
        return self.block(self.act(self.temporal(P, Q)))

    def switch_to_deploy(self):
        self.temporal.switch_to_deploy()
        self.block.switch_to_deploy()
        return self


class DifferenceTemporal1x1(nn.Module):
    """Ablation: fixed absolute temporal difference plus one 1x1 read-out.

    Removes the learned bi-temporal mixing entirely: the two dates are compared
    by |P - Q| and only a per-channel projection is learned.
    """

    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=deploy)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, P, Q):
        fused = self.proj((P - Q).abs())
        return fused if self.deploy else self.bn(fused)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = fold_conv_bn(self.proj.weight, None, self.bn)
        self.proj = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True)
        self.proj.weight.data = kernel.float()
        self.proj.bias.data = bias.float()
        self.deploy = True
        if hasattr(self, "bn"):
            del self.bn
        return self


class DifferenceTARStage(nn.Module):
    """Ablation: one scale with fixed absolute difference and plain local refinement.

    A single learned 1x1 mixes the two dates, so ``forward_with_difference``
    exposes the mixed difference stream for the difference-conditioned retrieval
    without any multi-scale pooling on top of it.
    """

    def __init__(self, in_channels, dim, deploy=False):
        super().__init__()
        self.temporal = DifferenceTemporal1x1(in_channels, dim, deploy=deploy)
        self.block = PlainLocalBlock(dim, deploy=deploy)
        self.act = nn.SiLU()

    def forward_with_difference(self, P, Q):
        detail = self.temporal(P, Q)
        return self.block(self.act(detail)), detail

    def forward(self, P, Q):
        return self.forward_with_difference(P, Q)[0]

    def switch_to_deploy(self):
        self.temporal.switch_to_deploy()
        self.block.switch_to_deploy()
        return self


class RepSPPBlock(nn.Module):
    """Multi-scale depthwise pooling that folds into one k x k depthwise convolution.

    Parallel 1x1 / 3x3 / 5x5 / 7x7 depthwise branches (each with its own batch
    norm) are summed, refined by a 1x1 pointwise convolution and added
    residually. At deployment every branch folds into a single 7x7 depthwise
    kernel plus one 1x1 convolution, so the training-time multi-scale path costs
    nothing extra at inference beyond a wider depthwise kernel.
    """

    def __init__(self, dim, kernels=(1, 3, 5, 7), use_residual=True, deploy=False):
        super().__init__()
        self.dim = dim
        self.kernels = tuple(kernels)
        self.max_kernel = max(self.kernels)
        self.use_residual = use_residual
        self.deploy = deploy
        if deploy:
            self.dw = nn.Conv2d(dim, dim, self.max_kernel, padding=self.max_kernel // 2,
                                groups=dim, bias=True)
            self.pw = nn.Conv2d(dim, dim, 1, bias=True)
            return
        self.dw_branches = nn.ModuleList(
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False) for k in self.kernels)
        self.bn_branches = nn.ModuleList(nn.BatchNorm2d(dim) for _ in self.kernels)
        for conv, bn, k in zip(self.dw_branches, self.bn_branches, self.kernels):
            if k == 1:
                nn.init.dirac_(conv.weight)
            else:
                # RepVGG-style: the branch starts switched off through its BN scale,
                # which keeps the initial function equal to the cheap 1x1 path while
                # gradients can still open the wider kernels.
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(bn.weight)
                nn.init.zeros_(bn.bias)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.dirac_(self.pw.weight)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        if self.deploy:
            y = self.pw(self.dw(x))
        else:
            pooled = self.bn_branches[0](self.dw_branches[0](x))
            for conv, bn in zip(self.dw_branches[1:], self.bn_branches[1:]):
                pooled = pooled + bn(conv(x))
            y = self.bn(self.pw(pooled))
        return x + y if self.use_residual else y

    def get_equivalent_kernel(self):
        target = self.max_kernel
        device = self.dw_branches[0].weight.device
        weight = torch.zeros(self.dim, 1, target, target, dtype=torch.float64, device=device)
        bias = torch.zeros(self.dim, dtype=torch.float64, device=device)
        for conv, bn in zip(self.dw_branches, self.bn_branches):
            kernel, shift = fold_conv_bn(conv.weight, None, bn)
            weight = weight + embed_depthwise_kernel(kernel, target)
            bias = bias + shift
        return weight, bias

    def switch_to_deploy(self):
        if self.deploy:
            return self
        weight, bias = self.get_equivalent_kernel()
        pointwise, pointwise_bias = fold_conv_bn(self.pw.weight, None, self.bn)
        self.dw = nn.Conv2d(self.dim, self.dim, self.max_kernel,
                            padding=self.max_kernel // 2, groups=self.dim, bias=True)
        self.dw.weight.data = weight.float()
        self.dw.bias.data = bias.float()
        self.pw = nn.Conv2d(self.dim, self.dim, 1, bias=True)
        self.pw.weight.data = pointwise.float()
        self.pw.bias.data = pointwise_bias.float()
        self.deploy = True
        for name in ("dw_branches", "bn_branches", "bn"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class SPPTARStage(nn.Module):
    """One scale: learned temporal mixing plus a multi-scale difference branch.

    The fused feature keeps the foldable concat/sum/signed-difference mixing; the
    absolute temporal difference is passed through the multi-scale depthwise
    pooling so small changed structures survive pooling. The pooled difference is
    added to the fused feature and also returned as a separate stream for the
    difference-conditioned retrieval.
    """

    def __init__(self, in_channels, dim, use_temporal_aux=True, use_dcr_aux=True,
                 use_residual=True, deploy=False, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.temporal = TemporalRep1x1(in_channels, dim, use_aux=use_temporal_aux, deploy=deploy)
        self.difference = DifferenceTemporal1x1(in_channels, dim, deploy=deploy)
        self.spp = RepSPPBlock(dim, kernels=kernels, deploy=deploy)
        self.block = RepLocalBlock(dim, use_aux=use_dcr_aux, use_residual=use_residual, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, P, Q):
        detail = self.spp(self.difference(P, Q))
        fused = self.act(self.temporal(P, Q)) + detail
        return self.block(fused), detail

    def switch_to_deploy(self):
        self.temporal.switch_to_deploy()
        self.difference.switch_to_deploy()
        self.spp.switch_to_deploy()
        self.block.switch_to_deploy()
        return self


class LocalContrast(nn.Module):
    """Divide every channel by its local mean magnitude (exposure invariant)."""

    def __init__(self, kernel=7):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        scale = F.avg_pool2d(x.abs(), self.kernel, 1, self.kernel // 2).clamp_min(1e-3)
        return x / scale


class ChangeGateTARStage(nn.Module):
    """Content path plus a signed, exposure-normalised change path.

    ``content = SiLU(1x1[P, Q])`` keeps appearance; ``delta = 1x1[d+, d-]`` keeps
    the signed change after dividing both dates by their local mean magnitude, so
    a global exposure shift cancels while a real change survives. The change gate
    modulates the content multiplicatively and adds a zero-initialised copy of
    ``delta``; the local block is the same foldable DW+PW as elsewhere.
    """

    def __init__(self, in_channels, dim, use_temporal_aux=True, use_dcr_aux=True,
                 use_residual=True, deploy=False):
        super().__init__()
        self.contrast = LocalContrast(7)
        self.content = nn.Conv2d(2 * in_channels, dim, 1, bias=False)
        self.bn_content = nn.BatchNorm2d(dim)
        self.difference = nn.Conv2d(2 * in_channels, dim, 1, bias=False)
        self.bn_difference = nn.BatchNorm2d(dim)
        self.gate = nn.Conv2d(dim + in_channels, dim, 1)
        self.beta = nn.Parameter(torch.zeros(1))
        self.block = RepLocalBlock(dim, use_aux=use_temporal_aux,
                                   use_residual=use_residual, deploy=deploy)

    def forward(self, P, Q):
        content = F.silu(self.bn_content(self.content(torch.cat((P, Q), dim=1))))
        p, q = self.contrast(P), self.contrast(Q)
        positive = (p - q).relu()
        negative = (q - p).relu()
        delta = self.bn_difference(self.difference(torch.cat((positive, negative), dim=1)))
        gate = self.gate(torch.cat((delta, positive + negative), dim=1)).sigmoid()
        fused = self.block(content * (1 + gate) + self.beta * delta)
        return fused, delta


class MultiScaleTAR(nn.Module):
    """Four-scale temporal fusion from paired encoder features to decoder inputs.

    Input scale order matches STR convention: stage1 (finest, 64x64) .. stage4
    (coarsest, 8x8); output list [t1, t2, t3, t4] feeds DCRDecoder.
    """

    def __init__(self, encoder_dims=(64, 64, 64, 64), dim=64,
                 use_temporal_aux=True, use_dcr_aux=True, use_residual=True, deploy=False,
                 variant="reparam", return_difference=False):
        super().__init__()
        if variant not in ("reparam", "plain", "diff", "spp", "spp1", "cg"):
            raise ValueError(f"unknown TAR variant: {variant}")
        self.variant = variant
        self.return_difference = return_difference
        if variant == "reparam":
            self.stage1 = TARStage(encoder_dims[0], dim, use_temporal_aux, use_dcr_aux,
                                   use_residual, deploy)
            self.stage2 = TARStage(encoder_dims[1], dim, use_temporal_aux, use_dcr_aux,
                                   use_residual, deploy)
            self.stage3 = TARStage(encoder_dims[2], dim, use_temporal_aux, use_dcr_aux,
                                   use_residual, deploy)
            self.stage4 = TARStage(encoder_dims[3], dim, use_temporal_aux, use_dcr_aux,
                                   use_residual, deploy)
        elif variant == "plain":
            self.stage1 = PlainTARStage(encoder_dims[0], dim, deploy)
            self.stage2 = PlainTARStage(encoder_dims[1], dim, deploy)
            self.stage3 = PlainTARStage(encoder_dims[2], dim, deploy)
            self.stage4 = PlainTARStage(encoder_dims[3], dim, deploy)
        elif variant == "diff":
            self.stage1 = DifferenceTARStage(encoder_dims[0], dim, deploy)
            self.stage2 = DifferenceTARStage(encoder_dims[1], dim, deploy)
            self.stage3 = DifferenceTARStage(encoder_dims[2], dim, deploy)
            self.stage4 = DifferenceTARStage(encoder_dims[3], dim, deploy)
        elif variant == "cg":
            self.stage1 = ChangeGateTARStage(encoder_dims[0], dim, use_temporal_aux,
                                             use_dcr_aux, use_residual, deploy)
            self.stage2 = ChangeGateTARStage(encoder_dims[1], dim, use_temporal_aux,
                                             use_dcr_aux, use_residual, deploy)
            self.stage3 = ChangeGateTARStage(encoder_dims[2], dim, use_temporal_aux,
                                             use_dcr_aux, use_residual, deploy)
            self.stage4 = ChangeGateTARStage(encoder_dims[3], dim, use_temporal_aux,
                                             use_dcr_aux, use_residual, deploy)
        else:
            # "spp" is the multi-scale pooling; "spp1" keeps the same branch
            # structure but with the 1x1 kernel only, so the gain of widening the
            # pooling window can be read off directly.
            kernels = (1, 3, 5, 7) if variant == "spp" else (1,)
            self.stage1 = SPPTARStage(encoder_dims[0], dim, use_temporal_aux, use_dcr_aux,
                                      use_residual, deploy, kernels=kernels)
            self.stage2 = SPPTARStage(encoder_dims[1], dim, use_temporal_aux, use_dcr_aux,
                                      use_residual, deploy, kernels=kernels)
            self.stage3 = SPPTARStage(encoder_dims[2], dim, use_temporal_aux, use_dcr_aux,
                                      use_residual, deploy, kernels=kernels)
            self.stage4 = SPPTARStage(encoder_dims[3], dim, use_temporal_aux, use_dcr_aux,
                                      use_residual, deploy, kernels=kernels)

    def forward(self, pre_feats, post_feats):
        if self.variant == "cg":
            t1, d1 = self.stage1(pre_feats[0], post_feats[0])
            t2, d2 = self.stage2(pre_feats[1], post_feats[1])
            t3, d3 = self.stage3(pre_feats[2], post_feats[2])
            t4, d4 = self.stage4(pre_feats[3], post_feats[3])
            # coarse-to-fine evidence propagation: a change seen at 8x8 seeds the
            # finer scales, so a small object inherits its parent's evidence.
            d3 = d3 + F.interpolate(d4, size=d3.shape[-2:], mode="bilinear", align_corners=False)
            d2 = d2 + F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            d1 = d1 + F.interpolate(d2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            fused = [t1, t2, t3, t4]
            if self.return_difference:
                return fused, [d1, d2, d3, d4]
            return fused
        if self.variant in ("spp", "spp1"):
            t1, d1 = self.stage1(pre_feats[0], post_feats[0])
            t2, d2 = self.stage2(pre_feats[1], post_feats[1])
            t3, d3 = self.stage3(pre_feats[2], post_feats[2])
            t4, d4 = self.stage4(pre_feats[3], post_feats[3])
            fused = [t1, t2, t3, t4]
            if self.return_difference:
                return fused, [d1, d2, d3, d4]
            return fused
        if self.variant == "diff" and self.return_difference:
            t1, d1 = self.stage1.forward_with_difference(pre_feats[0], post_feats[0])
            t2, d2 = self.stage2.forward_with_difference(pre_feats[1], post_feats[1])
            t3, d3 = self.stage3.forward_with_difference(pre_feats[2], post_feats[2])
            t4, d4 = self.stage4.forward_with_difference(pre_feats[3], post_feats[3])
            return [t1, t2, t3, t4], [d1, d2, d3, d4]
        t1 = self.stage1(pre_feats[0], post_feats[0])
        t2 = self.stage2(pre_feats[1], post_feats[1])
        t3 = self.stage3(pre_feats[2], post_feats[2])
        t4 = self.stage4(pre_feats[3], post_feats[3])
        return [t1, t2, t3, t4]

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
