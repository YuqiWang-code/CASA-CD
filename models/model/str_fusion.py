"""STRFusion Run1 top-level model.

Frozen ViT4 depth-as-scale encoder (CASA-CD ChangeViT) + TAR bi-temporal bridge
+ DCR decoder (STR-RepNet structural re-parameterization), CASA-CD training
contract: output = sigmoid probability map (B,1,256,256) in [0,1].

rep_mode in {"plain", "full"} (the Run1 single variable):
    plain (C0): no temporal aux, no DCR aux  -> single-path convs (+ BN-FR alpha)
    full  (M1): temporal + DCR aux branches  -> deploy folds to the SAME
                single-path function class as C0 (per-op identical shapes)

Both modes keep the foldable alpha residual (BN-FR) on every linear op, so the
deploy function class of C0 and M1 is identical; M1 - C0 isolates the training-
time structural re-parameterization effect.

RNG discipline (STR Run8/9/10): within every rep module the main branch is
constructed BEFORE the aux branches, so C0 (aux off) and M1 (aux on) share
bitwise-identical main-branch initialization (asserted in smoke).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.str_encoder import STRViT4Encoder
from model.str_tar import MultiScaleTAR
from model.str_dcr import DCRDecoder


class STRFusionNet(nn.Module):
    def __init__(self, pretrained_path, dim=160, rep_mode="full"):
        super().__init__()
        assert rep_mode in ("plain", "full"), f"rep_mode must be plain|full, got {rep_mode}"
        self.dim = dim
        self.rep_mode = rep_mode

        use_temporal_aux = rep_mode == "full"
        use_dcr_aux = rep_mode == "full"

        self.encoder = STRViT4Encoder(pretrained_path)
        # RNG discipline (STR Run9/Run10 lesson): the head is constructed BEFORE
        # any aux branch of TAR/DCR. nn.Conv2d construction consumes RNG even when
        # weights are later zeroed, so aux branches built before the head would
        # shift the head's init between C0 (no aux) and M1 (aux) and break the
        # C0/M1 epoch-0 bitwise identity. Main branches of every rep module are
        # constructed before their own aux branches, so shared inits stay aligned.
        self.head = nn.Conv2d(dim, 1, 1)
        self.tar = MultiScaleTAR(
            encoder_dims=(192, 192, 192, 192), dim=dim,
            use_temporal_aux=use_temporal_aux, use_dcr_aux=use_dcr_aux,
            use_residual=True,
        )
        self.decoder = DCRDecoder(dim=dim, use_aux=use_dcr_aux, use_residual=True)

    def forward(self, pre, post, label=None):
        # `label` is accepted for CASA train/val call contract and ignored
        # (no oracle routing in the fusion). The encoder is fully frozen
        # (requires_grad=False), so its outputs never build an autograd graph.
        pre_feats = self.encoder(pre)
        post_feats = self.encoder(post)
        feats = self.tar(pre_feats, post_feats)
        x = self.decoder(feats)
        logits = self.head(x)
        logits = F.interpolate(logits, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # frozen ViT stays a feature extractor
        return self

    @torch.no_grad()
    def switch_to_deploy(self):
        self.tar.switch_to_deploy()
        self.decoder.switch_to_deploy()
        return self
