"""Checks for the Run4 modules: change-gated TAR and manifold retrieval."""
import copy
import sys

import torch
from torch import nn

sys.path.insert(0, "models")

from model.casa_net import CASANet
from model.str_tar import ChangeGateTARStage, MultiScaleTAR
from model.layers.hyperbolic_context import FLAT_GRIDS, HyperbolicContext
from model.layers.lorentz_ops import lift, minkowski_barycenter
from model.layers.manifold_context import ManifoldContext
from model.layers.pyramid_context import channel_norm


def test_contrast_normalisation_ignores_exposure():
    stage = ChangeGateTARStage(8, 16).eval()
    P = torch.rand(1, 8, 16, 16)
    Q = torch.rand(1, 8, 16, 16)
    _, d_ref = stage(P, Q)
    _, d_scaled = stage(P * 1.0, Q * 3.0)
    # interior pixels normalise identically; borders share the padded window
    rel = (d_ref[:, :, 4:-4, 4:-4] - d_scaled[:, :, 4:-4, 4:-4]).abs().max()
    assert rel < 1e-3, rel


def test_signed_difference_is_asymmetric_and_zero_on_identity():
    stage = ChangeGateTARStage(8, 16).eval()
    P = torch.rand(1, 8, 16, 16)
    Q = torch.rand(1, 8, 16, 16) * 0.5
    _, forward_delta = stage(P, Q)
    _, backward_delta = stage(Q, P)
    assert forward_delta.abs().max() > 1e-4
    assert (forward_delta - backward_delta).abs().max() > 1e-4
    _, same = stage(P, P.clone())
    assert same.abs().max() == 0


def test_cross_scale_evidence_propagates_downward():
    tar = MultiScaleTAR((24, 32, 96, 320), 64, variant="cg", return_difference=True).eval()
    with torch.no_grad():
        tar.stage1.difference.weight.zero_()
        tar.stage1.bn_difference.weight.zero_()
        tar.stage1.bn_difference.bias.zero_()
        tar.stage1.beta.zero_()
    P = [torch.rand(1, c, s, s) for c, s in ((24, 64), (32, 32), (96, 16), (320, 8))]
    Q = [torch.rand(1, c, s, s) for c, s in ((24, 64), (32, 32), (96, 16), (320, 8))]
    with torch.no_grad():
        _, deltas = tar(P, Q)
    expected = torch.nn.functional.interpolate(
        deltas[1], size=deltas[0].shape[-2:], mode="bilinear", align_corners=False)
    # stage-1 delta is zeroed, so d1 must equal the upsampled, already-propagated d2
    assert torch.allclose(deltas[0], expected, atol=1e-5)
    assert [tuple(d.shape[-2:]) for d in deltas] == [(64, 64), (32, 32), (16, 16), (8, 8)]


def test_barycenter_is_idempotent_and_stays_on_the_manifold():
    tokens, dim = 5, 8
    value = torch.randn(2, 1, 3, tokens, dim)          # three queries per example
    values = value[:, :, :, :1].expand(-1, -1, -1, tokens, -1)   # all tokens identical
    weights = torch.softmax(torch.randn(2, 1, 3, tokens), dim=-1).unsqueeze(-1)
    out = minkowski_barycenter(values, weights)
    assert torch.allclose(out, value[:, :, :, 0], atol=1e-6)
    distinct = torch.randn(2, 1, 3, tokens, dim)
    uniform = torch.full((2, 1, 3, tokens, 1), 1 / tokens)
    mixed = minkowski_barycenter(distinct, uniform)
    euclidean = (distinct * (1 / tokens)).sum(-2)
    assert (mixed - euclidean).abs().max() > 1e-3
    lifted = lift(mixed)
    norm = lifted[..., :1].square() - lifted[..., 1:].square().sum(-1, keepdim=True)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_manifold_context_shares_geometry_but_not_the_readout():
    base = HyperbolicContext(grid_sizes=FLAT_GRIDS)
    manifold = ManifoldContext(grid_sizes=FLAT_GRIDS)
    base.scale_log_temperature = None if False else None
    with torch.no_grad():
        manifold.scale_log_temperature.copy_(torch.tensor([0.0, 0.3, 0.6, 0.9]))
    tau_base = base.scale_temperature(2)
    tau_scaled = manifold.scale_temperature(2)
    assert (tau_scaled - tau_base).abs().max() > 1e-3
    assert (manifold.scale_temperature(0) - manifold.scale_temperature(3)).abs().max() > 1e-3
    assert torch.allclose(base.scale_temperature(0), base.scale_temperature(3))
    # the independent prior must not move when the guide probability moves
    sources = [torch.rand(2, 64, s, s) for s in (64, 32, 16, 8)]
    with torch.no_grad():
        first = manifold.support_term(sources, None)
        manifold.guide.weight.mul_(3)
        second = manifold.support_term(sources, None)
    assert torch.allclose(first, second)
    assert first.shape == (2, 224)


def test_manifold_module_end_to_end():
    model = CASANet(module="manifold", tar_variant="cg").eval()
    image = torch.randn(2, 6, 256, 256)
    with torch.no_grad():
        pyramid = model(image[:, :3], image[:, 3:])
    assert [tuple(p.shape) for p in pyramid] == [(2, 1, 256, 256)] * 4
    assert torch.isfinite(pyramid[0]).all()
    folded = copy.deepcopy(model).switch_to_deploy().eval()
    with torch.no_grad():
        folded_pyramid = folded(image[:, :3], image[:, 3:])
    difference = max(float((a - b).abs().max()) for a, b in zip(pyramid, folded_pyramid))
    assert difference <= 2e-4, difference
    for module, tar in (("manifold", "reparam"), ("difference", "reparam"), ("manifold", "plain")):
        try:
            CASANet(module=module, tar_variant=tar)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{module} must reject tar_variant={tar}")


def test_manifold_context_reads_the_difference_stream():
    torch.manual_seed(0)
    ctx = ManifoldContext()
    ctx.eval()
    t = [torch.randn(1, 64, s, s) for s in (64, 32, 16, 8)]
    d = [torch.randn(1, 64, s, s) for s in (64, 32, 16, 8)]
    with torch.no_grad():
        proto_a = ctx.prototype_features(t, d)
        other = [x + 1.5 for x in d]
        proto_b = ctx.prototype_features(t, other)
    assert all(torch.allclose(a, b) is False for a, b in zip(proto_a, proto_b)), \
        "the prototypes ignore the difference stream"
    assert all(torch.allclose(p, channel_norm(ctx.norm_diff, q))
               for p, q in zip(proto_a, d)), "prototypes are not the normalised difference"
    try:
        ctx.prototype_features(t, None)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing difference stream must raise")

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}")
    print("[OK] run4 module checks passed")
