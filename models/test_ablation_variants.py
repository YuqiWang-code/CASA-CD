"""TAR and DCR ablation variants: parameter counts, folding and gradients."""
import copy

import torch
from torch import nn

from model.casa_net import CASANet
from model.str_dcr import DCRDecoder, PlainFuse1x1, PlainLocalBlock
from model.str_tar import MultiScaleTAR, PlainTARStage, PlainTemporal1x1

PARAMS = {("reparam", "reparam"): 2047125, ("plain", "reparam"): 1974157,
          ("reparam", "plain"): 2010250, ("plain", "plain"): 1937282}
DEPLOY = 1935810


def build(tar, decoder, seed=2333):
    torch.manual_seed(seed)
    return CASANet(None, "none", 2333, 1, False, "area", tar, decoder)


def main():
    torch.set_num_threads(4)
    image = torch.randn(2, 6, 256, 256)
    reference = None
    for (tar, decoder), expected in PARAMS.items():
        model = build(tar, decoder).eval()
        assert sum(p.numel() for p in model.parameters()) == expected
        folded = copy.deepcopy(model).eval().switch_to_deploy()
        assert sum(p.numel() for p in folded.parameters()) == DEPLOY
        with torch.no_grad():
            outputs = model(image[:, :3], image[:, 3:])
            deployed = folded(image[:, :3], image[:, 3:])
        assert len(outputs) == 4 and all(out.shape == (2, 1, 256, 256) for out in outputs)
        assert all(torch.allclose(a, b, atol=2e-4) for a, b in zip(outputs, deployed))
        if reference is None:
            reference = outputs
    # Plain variants keep the backbone path, so the finest logits stay finite and non-degenerate.
    assert all(torch.isfinite(out).all() for out in reference)

    plain = build("plain", "plain")
    losses = nn.BCEWithLogitsLoss()
    logits, _ = plain(image[:, :3], image[:, 3:], return_aux=True)
    losses(logits[0], (torch.rand(2, 1, 256, 256) > .5).float()).backward()
    names = ["tar.stage1.temporal.proj.weight", "tar.stage4.temporal.proj.weight",
             "decoder.fuse1.fuse.weight", "decoder.refine.dw.weight"]
    grads = dict(plain.named_parameters())
    for name in names:
        assert grads[name].grad is not None and torch.isfinite(grads[name].grad).all()
        assert grads[name].grad.norm() > 0, name

    for factory in (lambda: MultiScaleTAR(variant="bogus"), lambda: DCRDecoder(variant="bogus")):
        try:
            factory()
        except ValueError:
            pass
        else:
            raise AssertionError("unknown variant must be rejected")

    stage = PlainTARStage(24, 64)
    feature = torch.randn(1, 24, 32, 32)
    assert stage.block.dw.groups == 64 and isinstance(stage.temporal, PlainTemporal1x1)
    assert stage(feature, feature).shape == (1, 64, 32, 32)
    assert isinstance(DCRDecoder(variant="plain").fuse1, PlainFuse1x1)
    assert isinstance(DCRDecoder(variant="plain").refine, PlainLocalBlock)
    assert isinstance(DCRDecoder().refine, nn.Module) and DCRDecoder().variant == "reparam"
    print("TAR/DCR plain variants: parameter counts, folding and gradients PASS")


if __name__ == "__main__":
    main()
