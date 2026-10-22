"""WDMF-Net MobileNetV2/HFEA encoder with TAR skips and DCR decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from model.wdmf_mobilenet_v2 import mobilenet_v2
from model.wdmf_encoder import EncoderFusion
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder
from model.layers.change_context_qkv import ChangeAwareCompressedQKV
from model.layers.hyperbolic_context_qkv import HyperbolicCompressedQKV


class CASAWDMFSTRNet(nn.Module):
    def __init__(self, pretrained_path=None, module="none"):
        super().__init__()
        if module not in ("none", "hyperbolic", "qkv"):
            raise ValueError(module)
        self.module = module
        self.encoder = mobilenet_v2(pretrained_path)
        self.encoder_fusion = EncoderFusion([16, 24, 32, 96, 320], midc=32, outc=64)
        self.tar = MultiScaleTAR((64, 64, 64, 64), dim=64,
                                 use_temporal_aux=True, use_dcr_aux=True,
                                 use_residual=True)
        self.decoder = DCRDecoder(dim=64, use_aux=True, use_residual=True)
        self.head = nn.Conv2d(64, 1, 1)
        # Construct after the common path so all variants share its initialization.
        with torch.random.fork_rng():
            torch.manual_seed(2333)
            block = {"hyperbolic": HyperbolicCompressedQKV,
                     "qkv": ChangeAwareCompressedQKV}.get(module)
            self.stage_modules = nn.ModuleList([block(64, 32) for _ in range(4)] if block else [])

    def forward(self, a, b):
        features_a = self.encoder_fusion(*self.encoder(a))
        features_b = self.encoder_fusion(*self.encoder(b))
        if self.stage_modules:
            pairs = [op(x, y) for op, x, y in zip(self.stage_modules, features_a, features_b)]
            features_a, features_b = zip(*pairs)
        fused = self.tar(features_a, features_b)
        pyramid = self.decoder.forward_pyramid(fused)
        return tuple(torch.sigmoid(F.interpolate(self.head(x), size=a.shape[-2:],
                                                mode="bilinear", align_corners=False))
                     for x in pyramid)

    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self

    def diagnostics(self):
        return {f"stage{i + 1}": dict(op.last_diag) for i, op in enumerate(self.stage_modules)}
