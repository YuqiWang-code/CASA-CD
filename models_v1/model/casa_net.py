"""Native MobileNetV2 features, shared TAR, heterogeneous evidence experts and DCR."""
import torch
from torch import nn
from torch.nn import functional as F

from model.mobilenet_v2 import mobilenet_v2
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder
from model.layers.candidate_context import MODEL_VARIANTS, CandidateContext


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
            self.context = CandidateContext(module)
        self.configuration = self.context.metadata()
        self.deployed = False

    def _decode(self, a, b):
        features_a, features_b = self.encoder(a)[1:], self.encoder(b)[1:]
        tar = self.tar(features_a, features_b)
        if self.deployed and self.module == 'none':
            return self.decoder.forward_pyramid(tar), dict(guide_logits=())
        x, normalized, guide, bank, waves = self.context.prepare(features_a, features_b)
        decoded = [None] * 4
        for s in (3, 2, 1, 0):
            up = None if s == 3 else F.interpolate(decoded[s+1], size=x[s].shape[-2:], mode='bilinear', align_corners=False)
            z = self.context.mix(s, x[s], normalized[s], tar[s], up, bank, waves)
            decoded[s] = z if s == 3 else getattr(self.decoder, f'block{s+1}')(
                self.decoder.act(getattr(self.decoder, f'fuse{s+1}')(z, up)))
        decoded[0] = self.decoder.refine(decoded[0])
        return decoded, dict(guide_logits=tuple(guide))

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
        if self.module == 'none':
            self.context = nn.Identity()
        else:
            self.context.stem.switch_to_deploy()
        self.deployed = True
        return self

    def diagnostics(self):
        return dict(getattr(self.context, 'last_diag', {}))

    def context_metadata(self):
        return dict(self.configuration)
