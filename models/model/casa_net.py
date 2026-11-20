"""Native MobileNetV2 features, TAR, joint pyramid context and DCR."""
import torch
from torch import nn
from torch.nn import functional as F

from model.mobilenet_v2 import mobilenet_v2
from model.str_tar import MultiScaleTAR
from model.str_dcr import build_decoder
from model.layers.pyramid_context import AuxiliaryContext
from model.layers.euclidean_context import EuclideanContext
from model.layers.hyperbolic_context import HyperbolicContext, MatchedEuclideanContext
from model.layers.manifold_context import ManifoldContext
from model.layers.difference_context import DifferenceContext


class CASANet(nn.Module):
    def __init__(self, pretrained_path=None, module="none", context_seed=2333,
                 context_heads=1, detail_head=False, detail_target="area",
                 tar_variant="reparam", decoder_variant="reparam", support="cell",
                 kv_hierarchy="flat", freeze_tau=False, edge_supervision=False,
                 edge_detach="retrieval", aggregate_mode="lorentz"):
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
        with torch.random.fork_rng():
            torch.manual_seed(context_seed)
            options = dict(heads=context_heads, head_dim=16) if context_heads != 1 else {}
            if issubclass(context_type, HyperbolicContext):
                options["edge_supervision"] = bool(edge_supervision)
                options["edge_detach"] = edge_detach
                if context_type is ManifoldContext:
                    options["aggregate_mode"] = aggregate_mode
            if module in ("hyperbolic", "matched", "difference"):
                options["support_mode"] = support
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
        detail = self.detail(torch.cat((features_a[0], features_b[0]), 1)) if self.detail is not None else None
        return self.decoder.forward_pyramid(enhanced), auxiliary, detail

    @staticmethod
    def _resize(logits, size):
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def _primary(self, coarse, detail, size):
        primary = self._resize(coarse, size)
        return primary if detail is None else primary + self._resize(detail, size)

    def forward(self, a, b, return_aux=False):
        """Return four probabilities, or raw logits and change-guide supervision."""
        pyramid, auxiliary, detail = self._decode(a, b, return_aux=return_aux)
        coarse = self.head(pyramid[0])
        primary = self._primary(coarse, detail, a.shape[-2:])
        if detail is not None and return_aux:
            auxiliary["detail_logits"] = self._resize(coarse, detail.shape[-2:]) + detail
            auxiliary["detail_target"] = self.detail_target
        logits = (primary, *(self._resize(self.head(feature), a.shape[-2:]) for feature in pyramid[1:]))
        return (logits, auxiliary) if return_aux else tuple(x.sigmoid() for x in logits)

    def predict(self, a, b):
        """Return the final change probability without auxiliary prediction heads."""
        pyramid, _, detail = self._decode(a, b)
        logits = self._primary(self.head(pyramid[0]), detail, a.shape[-2:])
        return logits.sigmoid()

    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        # the edge heads are training-only supervision taps; they never take part
        # in the forward pass of the deployed model, so drop them from the budget
        self.tar.edge_head = None
        if hasattr(self.context, "edge_head"):
            self.context.edge_head = None
        return self

    def diagnostics(self):
        return dict(self.context.last_diag)
