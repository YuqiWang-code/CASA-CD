"""Top-down DCR decoder with foldable local refinement.

TAR features at 8/16/32/64 spatial resolution are fused from coarse to fine.
The four decoder outputs support deep supervision at input resolution.
The ``plain`` variant keeps the same topology but replaces every foldable
multi-branch re-parameterized layer with a single convolution plus batch norm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_reparam import (
    RepDW3,
    RepPW1x1,
    RepPairFuse1x1,
    fold_conv_bn,
    switch_module_to_deploy,
)


class RepLocalBlock(nn.Module):
    """Local refinement block: DW3 (residual) -> SiLU -> PW1 (residual) -> SiLU."""

    def __init__(self, dim, use_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.dim = dim
        self.dw = RepDW3(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        self.pw = RepPW1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.dw(x))
        x = self.act(self.pw(x))
        return x

    def switch_to_deploy(self):
        self.dw.switch_to_deploy()
        self.pw.switch_to_deploy()
        return self


class PlainLocalBlock(nn.Module):
    """Ablation: plain depthwise 3x3 and pointwise 1x1 without multi-branch re-parameterization."""

    def __init__(self, dim, deploy=False):
        super().__init__()
        self.dim = dim
        self.deploy = deploy
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=deploy)
        self.pw = nn.Conv2d(dim, dim, 1, bias=deploy)
        if not deploy:
            self.bn_dw = nn.BatchNorm2d(dim)
            self.bn_pw = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.deploy:
            return self.act(self.pw(self.act(self.dw(x))))
        x = self.act(self.bn_dw(self.dw(x)))
        return self.act(self.bn_pw(self.pw(x)))

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel_dw, bias_dw = fold_conv_bn(self.dw.weight, None, self.bn_dw)
        kernel_pw, bias_pw = fold_conv_bn(self.pw.weight, None, self.bn_pw)
        self.dw = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim, bias=True)
        self.pw = nn.Conv2d(self.dim, self.dim, 1, bias=True)
        self.dw.weight.data, self.dw.bias.data = kernel_dw.float(), bias_dw.float()
        self.pw.weight.data, self.pw.bias.data = kernel_pw.float(), bias_pw.float()
        self.deploy = True
        for name in ("bn_dw", "bn_pw"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class PlainFuse1x1(nn.Module):
    """Ablation: plain concat 1x1 fusion without sum and signed-difference branches."""

    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        self.fuse = nn.Conv2d(2 * channels, channels, 1, bias=deploy)
        if not deploy:
            self.bn = nn.BatchNorm2d(channels)

    def forward(self, left, right):
        fused = self.fuse(torch.cat((left, right), 1))
        return fused if self.deploy else self.bn(fused)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = fold_conv_bn(self.fuse.weight, None, self.bn)
        self.fuse = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        self.fuse.weight.data, self.fuse.bias.data = kernel.float(), bias.float()
        self.deploy = True
        if hasattr(self, "bn"):
            del self.bn
        return self


class DCRDecoder(nn.Module):
    """Top-down multi-scale decoder over TAR features [t1, t2, t3, t4] (dim ch)."""

    def __init__(self, dim=64, use_aux=True, use_residual=True, deploy=False, variant="reparam"):
        super().__init__()
        if variant not in ("reparam", "plain"):
            raise ValueError(f"unknown decoder variant: {variant}")
        self.dim = dim
        self.use_aux = use_aux
        self.variant = variant
        self.act = nn.SiLU()

        if variant == "reparam":
            self.fuse3 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.block3 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.fuse2 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.block2 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.fuse1 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.block1 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.refine = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        else:
            self.fuse3 = PlainFuse1x1(dim, deploy=deploy)
            self.block3 = PlainLocalBlock(dim, deploy=deploy)
            self.fuse2 = PlainFuse1x1(dim, deploy=deploy)
            self.block2 = PlainLocalBlock(dim, deploy=deploy)
            self.fuse1 = PlainFuse1x1(dim, deploy=deploy)
            self.block1 = PlainLocalBlock(dim, deploy=deploy)
            self.refine = PlainLocalBlock(dim, deploy=deploy)

    def forward(self, feats):
        return self.forward_pyramid(feats)[0]

    def forward_pyramid(self, feats):
        """Return the four decoder scales for deep supervision."""
        t1, t2, t3, t4 = feats

        d4 = t4

        u4 = F.interpolate(d4, size=t3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.block3(self.act(self.fuse3(t3, u4)))

        u3 = F.interpolate(d3, size=t2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.block2(self.act(self.fuse2(t2, u3)))

        u2 = F.interpolate(d2, size=t1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.block1(self.act(self.fuse1(t1, u2)))

        d1 = self.refine(d1)
        return [d1, d2, d3, d4]

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
