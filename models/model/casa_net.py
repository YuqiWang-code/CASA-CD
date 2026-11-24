"""Native MobileNetV2 features, TAR, joint pyramid context and DCR."""
import torch
from torch import nn
from torch.nn import functional as F

from model.mobilenet_v2 import mobilenet_v2
from model.str_reparam import fold_conv_bn
from model.str_tar import MultiScaleTAR
from model.str_dcr import build_decoder
from model.layers.pyramid_context import AuxiliaryContext
from model.layers.euclidean_context import EuclideanContext
from model.layers.hyperbolic_context import (COMPACT_GRIDS, HyperbolicContext,
                                             MatchedEuclideanContext)
from model.layers.manifold_context import ManifoldContext
from model.layers.difference_context import DifferenceContext


class FineRefine(nn.Module):
    """One 128x128 refinement of the change logits, fused with the raw stem.

    The decoder's finest map is 64x64, so a one-pixel boundary at 256x256 is a
    quarter of a cell there; upsampling that logit directly is what caps the
    boundary F1 (interior 96.5 vs boundary 0.59 on LEVIR). The encoder already
    has 16 channels at 128x128, so the head can spend one small block there
    instead of widening the decoder: ``1x1(1 + stem) -> BN -> SiLU -> DW3x3 ->
    BN -> SiLU -> 1x1 -> 0``, added to the upsampled logit. The last 1x1 is
    zero-initialised, so the block starts as an exact identity.
    """

    def __init__(self, stem_channels=16, dim=64):
        super().__init__()
        self.stem_channels = stem_channels
        self.dim = dim
        self.deploy = False
        self.fuse = nn.Conv2d(1 + stem_channels, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="replicate",
                            groups=dim, bias=False)
        self.dw_norm = nn.BatchNorm2d(dim)
        self.out = nn.Conv2d(dim, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, logits, stem):
        x = self.fuse(torch.cat((logits, stem), dim=1))
        if not self.deploy:
            x = self.norm(x)
        x = F.silu(x)
        x = self.dw(x)
        if not self.deploy:
            x = self.dw_norm(x)
        return logits + self.out(F.silu(x))

    def switch_to_deploy(self):
        """Keep the two BatchNorms: folding them is numerically unsafe here.

        The head runs at 128x128 where several channels are dead, so
        ``running_var`` reaches 1e-45 and ``gamma / sqrt(var + eps)`` reaches 1e7.
        The fold is exact in float64, but the FP32 evaluation of the unfolded
        path amplifies the same rounding: a trained D_CTRL checkpoint showed a
        4.8e-2 probability gap (best-model weights: 1.7e-4). Two BatchNorms on a
        64-channel 128x128 map cost ~0.05 ms, far less than a deploy graph that
        disagrees with the scored model, so the deploy form keeps them.
        """
        return self

class CASANet(nn.Module):
    def __init__(self, pretrained_path=None, module="none", context_seed=2333,
                 context_heads=1, detail_head=False, detail_target="area",
                 tar_variant="reparam", decoder_variant="reparam", support="cell",
                 kv_hierarchy="flat", freeze_tau=False, edge_supervision=False,
                 edge_detach="retrieval", aggregate_mode="lorentz",
                 prototype_pooling="euclidean", retrieval_scales=(0, 1, 2, 3),
                 compact_prototypes=False):
        super().__init__()
        context_type = {"none": AuxiliaryContext, "hyperbolic": HyperbolicContext,
                        "manifold": ManifoldContext,
                        "qkv": EuclideanContext, "matched": MatchedEuclideanContext,
                        "difference": DifferenceContext}[module]
        if module in ("difference", "manifold") and tar_variant not in ("spp", "spp1", "diff", "cg"):
            raise ValueError("the difference context needs a temporal difference stream "
                             "(--tar-variant spp, spp1 or diff)")
        self.module = module
        self.tar_variant = tar_variant
        self.decoder_variant = decoder_variant
        self.encoder = mobilenet_v2(pretrained_path)
        self.tar = MultiScaleTAR((24, 32, 96, 320), dim=64,
                                 use_temporal_aux=True, use_dcr_aux=True, use_residual=True,
                                 variant=tar_variant,
                                 return_difference=module in ("difference", "manifold"),
                                 edge_supervision=edge_supervision and tar_variant == "cg")
        self.decoder = build_decoder(dim=64, use_aux=True, use_residual=True,
                                     variant=decoder_variant)
        self.head = nn.Conv2d(64, 1, 1)
        # 128x128 refinement of the head logits, fused with the raw encoder stem
        self.fine = FineRefine(stem_channels=32, dim=64)
        with torch.random.fork_rng():
            torch.manual_seed(context_seed)
            options = dict(heads=context_heads, head_dim=16) if context_heads != 1 else {}
            if issubclass(context_type, HyperbolicContext):
                options["edge_supervision"] = bool(edge_supervision)
                options["edge_detach"] = edge_detach
                options["prototype_pooling"] = prototype_pooling
                if context_type is ManifoldContext:
                    options["aggregate_mode"] = aggregate_mode
            if module in ("hyperbolic", "matched", "difference", "manifold"):
                options["support_mode"] = support
                # only the hyperbolic family exposes the two Run10 switches
                if issubclass(context_type, HyperbolicContext):
                    options["retrieval_scales"] = retrieval_scales
                    if compact_prototypes:
                        options["grid_sizes"] = COMPACT_GRIDS
                options["kv_hierarchy"] = kv_hierarchy
                options["freeze_tau"] = freeze_tau
            self.context = context_type(**options)
            self.detail = None
            self.detail_target = detail_target
            if detail_head:
                self.detail = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False),
                                            nn.Conv2d(64, 64, 3, padding=1, padding_mode="replicate",
                                                      groups=64, bias=False),
                                            nn.SiLU(), nn.Conv2d(64, 1, 1))
                nn.init.zeros_(self.detail[-1].weight)
                nn.init.zeros_(self.detail[-1].bias)

    def _decode(self, a, b, return_aux=False):
        features_a, features_b = self.encoder(a), self.encoder(b)
        # the TAR may hand back (fused, difference, edges) when an edge head is on
        result = self.tar(features_a[1:], features_b[1:])
        difference, tar_edges = None, None
        if len(result) == 3:
            tar, difference, tar_edges = result
        elif len(result) == 2:
            tar, difference = result
        else:
            tar = result
        if self.module in ("difference", "manifold"):
            enhanced, auxiliary = self.context(tar, difference,
                                               return_aux=return_aux and self.training)
        elif self.module in ("hyperbolic", "matched"):
            enhanced, auxiliary = self.context(tar, return_aux=return_aux and self.training)
        else:
            enhanced, auxiliary = self.context(tar)
        if tar_edges is not None:
            auxiliary = dict(auxiliary or {})
            merged = list(auxiliary.get("edge_logits") or ())
            auxiliary["edge_logits"] = tuple(list(tar_edges) + merged)
        stem = torch.cat((features_a[0], features_b[0]), 1)
        detail = self.detail(stem) if self.detail is not None else None
        return self.decoder.forward_pyramid(enhanced), auxiliary, detail, stem

    @staticmethod
    def _resize(logits, size):
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def _primary(self, coarse, stem, detail):
        """64x64 logit -> 128x128 fine refinement -> 256x256, plus detail residual."""
        refined = self.fine(self._resize(coarse, stem.shape[-2:]), stem)
        return refined if detail is None else refined + detail

    def forward(self, a, b, return_aux=False):
        """Return four probabilities, or raw logits and change-guide supervision."""
        pyramid, auxiliary, detail, stem = self._decode(a, b, return_aux=return_aux)
        refined = self._primary(self.head(pyramid[0]), stem, detail)
        primary = self._resize(refined, a.shape[-2:])
        if detail is not None and return_aux:
            auxiliary["detail_logits"] = refined
            auxiliary["detail_target"] = self.detail_target
        logits = (primary, *(self._resize(self.head(feature), a.shape[-2:]) for feature in pyramid[1:]))
        return (logits, auxiliary) if return_aux else tuple(x.sigmoid() for x in logits)

    def predict(self, a, b):
        """Return the final change probability without auxiliary prediction heads."""
        pyramid, _, detail, stem = self._decode(a, b)
        logits = self._resize(self._primary(self.head(pyramid[0]), stem, detail), a.shape[-2:])
        return logits.sigmoid()

    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        self.fine.switch_to_deploy()
        # the edge heads are training-only supervision taps; they never take part
        # in the forward pass of the deployed model, so drop them from the budget
        self.tar.edge_head = None
        if hasattr(self.context, "edge_head"):
            self.context.edge_head = None
        return self

    def diagnostics(self):
        return dict(self.context.last_diag)
