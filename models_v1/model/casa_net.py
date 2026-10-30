"""Native MobileNetV2 features, shared TAR, heterogeneous evidence experts and DCR."""
import torch
from torch import nn
from torch.nn import functional as F

from model.mobilenet_v2 import mobilenet_v2
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder
from model.layers.pyramid_context import AuxiliaryContext
from model.layers.moe_context import MODEL_VARIANTS, MoEContext


class CASANet(nn.Module):
    def __init__(self, pretrained_path=None, module="none"):
        super().__init__()
        if module not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant: {module}")
        self.module = module
        self.encoder = mobilenet_v2(pretrained_path)
        self.tar = MultiScaleTAR((24, 32, 96, 320), dim=64,
                                 use_temporal_aux=True, use_dcr_aux=True, use_residual=True)
        self.decoder = DCRDecoder(dim=64, use_aux=True, use_residual=True)
        self.head = nn.Conv2d(64, 1, 1)
        with torch.random.fork_rng():
            torch.manual_seed(2333)
            self.context = AuxiliaryContext() if module == "none" else MoEContext(module)

    def _decode(self, a, b):
        features_a, features_b = self.encoder(a)[1:], self.encoder(b)[1:]
        enhanced, auxiliary = self.context(self.tar(features_a, features_b))
        return self.decoder.forward_pyramid(enhanced), auxiliary

    def forward(self, a, b, return_aux=False):
        """Return four probabilities, or raw logits and change-guide supervision."""
        pyramid, auxiliary = self._decode(a, b)
        logits = tuple(F.interpolate(self.head(x), size=a.shape[-2:],
                                     mode="bilinear", align_corners=False)
                       for x in pyramid)
        return (logits, auxiliary) if return_aux else tuple(x.sigmoid() for x in logits)

    def predict(self, a, b):
        """Return the final change probability without auxiliary prediction heads."""
        pyramid, _ = self._decode(a, b)
        logits = F.interpolate(self.head(pyramid[0]), size=a.shape[-2:],
                               mode="bilinear", align_corners=False)
        return logits.sigmoid()

    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self

    def diagnostics(self):
        return dict(self.context.last_diag)

    def context_metadata(self):
        if self.module != "none":
            return self.context.metadata()
        return dict(variant="none", active_experts=["tar"], routing_mode="identity",
                    execution="shared_tar_only", context_grid_sizes=list(self.context.grid_sizes),
                    context_prototypes_per_scale=[0, 0, 0, 0], joint_contexts=0,
                    qk_geometry={}, value_geometry="euclidean", heads=0, qkv_dim=0,
                    query_key_head_dim=0, value_head_dim=0, compression_power=0, support_power=0,
                    self_reference=False, hyperbolic_radius=None, key_mode=None, readout_mode="identity",
                    context_gamma_shape=[], local_gamma_shape=[], frequency_width=0,
                    frequency_transform=None, router_hidden_channels=0)
