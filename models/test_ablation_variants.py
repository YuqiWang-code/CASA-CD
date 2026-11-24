"""TAR and DCR ablation variants: parameter counts, folding and gradients."""
import copy

import torch
from torch import nn

from model.casa_net import CASANet
from model.str_dcr import (DCRDecoder, PlainFuse1x1, PlainLocalBlock, RepLocalBlock,
                           SingleLevelDecoder, build_decoder)
from model.str_tar import (DifferenceTARStage, DifferenceTemporal1x1, MultiScaleTAR,
                           PlainTARStage, PlainTemporal1x1)

# Run10 lean head: the decoder's trailing refine block moved to the 128x128
# FineRefine, so every arm pays that block once in deploy instead of twice.
PARAMS = {("reparam", "reparam"): (2042324, 1933891), ("reparam", "plain"): (2008331, 1933891),
          ("reparam", "none"): (1977739, 1903683), ("plain", "reparam"): (1969356, 1933891),
          ("plain", "plain"): (1935363, 1933891), ("plain", "none"): (1904771, 1903683),
          ("diff", "reparam"): (1939148, 1903683), ("diff", "plain"): (1905155, 1903683),
          ("diff", "none"): (1874563, 1873475)}


def build(tar, decoder, seed=2333):
    torch.manual_seed(seed)
    return CASANet(None, "none", 2333, 1, False, "area", tar, decoder)


def main():
    torch.set_num_threads(4)
    image = torch.randn(2, 6, 256, 256)
    outputs_by_arm = {}
    for (tar, decoder), (expected, deployed) in PARAMS.items():
        model = build(tar, decoder).eval()
        assert sum(p.numel() for p in model.parameters()) == expected, (tar, decoder)
        folded = copy.deepcopy(model).eval().switch_to_deploy()
        assert sum(p.numel() for p in folded.parameters()) == deployed
        with torch.no_grad():
            outputs = model(image[:, :3], image[:, 3:])
            moved = folded(image[:, :3], image[:, 3:])
        assert len(outputs) == 4 and all(out.shape == (2, 1, 256, 256) for out in outputs)
        assert all(torch.allclose(a, b, atol=2e-4) for a, b in zip(outputs, moved))
        assert all(torch.isfinite(out).all() for out in outputs)
        outputs_by_arm[(tar, decoder)] = outputs
    # Removing a module must change behaviour (relative to the arm's own output scale),
    # and the graph itself must differ, otherwise the arm is a silent no-op.
    control = outputs_by_arm[("reparam", "reparam")]
    for arm in (("diff", "reparam"), ("reparam", "none"), ("diff", "none")):
        difference = max(float((a - b).abs().max()) for a, b in zip(control, outputs_by_arm[arm]))
        assert difference > 1e-4, arm
    diff_model = build("diff", "none")
    assert isinstance(diff_model.tar.stage1, DifferenceTARStage)
    assert isinstance(diff_model.tar.stage4, DifferenceTARStage)
    assert isinstance(diff_model.decoder, SingleLevelDecoder)
    assert not hasattr(diff_model.decoder, "fuse1") and not hasattr(diff_model.decoder, "refine")

    gradients = {("plain", "plain"): ["tar.stage1.temporal.proj.weight", "tar.stage4.temporal.proj.weight",
                                      "decoder.fuse1.fuse.weight", "fine.out.weight"],
                 ("diff", "reparam"): ["tar.stage1.temporal.proj.weight", "tar.stage4.temporal.proj.weight",
                                       "decoder.fuse1.fuse_c.weight", "decoder.block1.dw.dw3.weight"],
                 ("reparam", "none"): ["tar.stage1.temporal.proj_c.weight", "decoder.proj.weight",
                                       "decoder.block.dw.weight"]}
    losses = nn.BCEWithLogitsLoss()
    for (tar, decoder), names in gradients.items():
        model = build(tar, decoder)
        logits, _ = model(image[:, :3], image[:, 3:], return_aux=True)
        losses(logits[0], (torch.rand(2, 1, 256, 256) > .5).float()).backward()
        parameters = dict(model.named_parameters())
        for name in names:
            assert parameters[name].grad is not None, name
            assert torch.isfinite(parameters[name].grad).all(), name
            assert parameters[name].grad.norm() > 0, name

    # the fine block is an exact identity at init (its last 1x1 is zero), so the
    # upstream convolutions only see gradient once training moves that 1x1
    model = build("reparam", "reparam")
    model.fine.out.weight.data.fill_(0.01)
    logits, _ = model(image[:, :3], image[:, 3:], return_aux=True)
    losses(logits[0], (torch.rand(2, 1, 256, 256) > .5).float()).backward()
    assert dict(model.named_parameters())["fine.fuse.weight"].grad.norm() > 0

    for factory in (lambda: MultiScaleTAR(variant="bogus"), lambda: DCRDecoder(variant="bogus"),
                    lambda: build_decoder(variant="bogus")):
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
    assert isinstance(DCRDecoder().block1, RepLocalBlock) and DCRDecoder().variant == "reparam"
    # the trailing refine block is gone: its work moved to the 128x128 head
    assert not hasattr(DCRDecoder(), "refine")

    difference_stage = DifferenceTARStage(24, 64)
    assert isinstance(difference_stage.temporal, DifferenceTemporal1x1)
    assert difference_stage(feature, -feature).shape == (1, 64, 32, 32)
    single = build_decoder(variant="none")
    assert isinstance(single, SingleLevelDecoder)
    pyramid = single.forward_pyramid([torch.randn(1, 64, 64, 64), torch.randn(1, 64, 32, 32),
                                      torch.randn(1, 64, 16, 16), torch.randn(1, 64, 8, 8)])
    assert [tuple(x.shape) for x in pyramid] == [(1, 64, 64, 64), (1, 64, 32, 32),
                                                 (1, 64, 16, 16), (1, 64, 8, 8)]
    print("TAR/DCR ablation variants: parameter counts, folding, gradients and no-op guards PASS")


if __name__ == "__main__":
    main()
