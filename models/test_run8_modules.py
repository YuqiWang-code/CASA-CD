"""Checks for the change/size decoupling bridge (CD2)."""
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, "models")
from model.casa_net import CASANet  # noqa: E402
from model.str_tar import ChangeDecoupleTARStage, MultiScaleTAR  # noqa: E402

SIDE = 128


def _stage(dim=64, stem_dim=0):
    torch.manual_seed(0)
    return ChangeDecoupleTARStage(dim + 8, dim, stem_dim=stem_dim).eval()


def test_illumination_normalisation_removes_a_global_gain():
    stage = _stage()
    p = torch.rand(1, 72, 32, 32)
    q = torch.rand(1, 72, 32, 32)
    _, d_ref = stage(p, q)
    _, d_gain = stage(p * 3.0, q * 3.0)
    inner = (slice(None), slice(None), slice(4, -4), slice(4, -4))
    assert (d_ref[inner] - d_gain[inner]).abs().max() < 1e-3


def test_signed_difference_is_asymmetric_and_zero_when_equal():
    stage = _stage()
    p = torch.rand(1, 72, 16, 16)
    q = torch.rand(1, 72, 16, 16)
    _, both = stage(p, q)
    _, swapped = stage(q, p)
    assert (both - swapped).abs().max() > 1e-4
    _, same = stage(p, p.clone())
    assert same.abs().max() == 0


def test_bands_reconstruct_the_change_feature():
    stage = _stage()
    p = torch.rand(1, 72, 32, 32)
    q = torch.rand(1, 72, 32, 32)
    with torch.no_grad():
        delta = stage.delta(torch.cat((F.relu(p - q), F.relu(q - p)), 1))
        delta = stage.delta_bn(delta)
        pooled3 = F.avg_pool2d(delta, 3, 1, 1)
        pooled9 = F.avg_pool2d(delta, 9, 1, 4)
        small = delta - pooled3
        medium = pooled3 - pooled9
        large = pooled9
        assert (small + medium + large - delta).abs().max() < 1e-6


def test_router_starts_uniform_and_reproduces_the_change_feature():
    stage = _stage()
    p = torch.rand(1, 72, 32, 32)
    q = torch.rand(1, 72, 32, 32)
    with torch.no_grad():
        weight = torch.softmax(stage.router(torch.zeros(1, 192, 2, 2)) * 0, 1) * 3.0
        assert torch.allclose(weight, torch.ones_like(weight), atol=1e-6)


def test_stem_detail_reaches_only_the_small_band():
    torch.manual_seed(0)
    stage = _stage(stem_dim=16)
    p = torch.rand(1, 72, 64, 64)
    q = torch.rand(1, 72, 64, 64)
    stem = torch.rand(1, 16, 128, 128)
    plain, _ = stage(p, q)
    with_stem, _ = stage(p, q, stem=stem)
    assert (plain - with_stem).abs().max() > 1e-4
    assert plain.shape == with_stem.shape


def test_stage_and_network_shapes_and_folding():
    torch.manual_seed(0)
    net = CASANet(module="none", tar_variant="cd2").eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        probs = net(x, x)
    assert len(probs) == 4 and probs[0].shape == (1, 1, 256, 256)
    assert all(torch.isfinite(p).all() for p in probs)
    import copy
    folded = copy.deepcopy(net).switch_to_deploy().eval()
    with torch.no_grad():
        ref = net.predict(x, x)
        new = folded.predict(x, x)
    assert (ref - new).abs().max() <= 2e-4
    assert ((ref > 0.5) == (new > 0.5)).float().mean() == 1.0


def test_multi_scale_bridge_returns_four_scales_and_a_difference_stream():
    torch.manual_seed(0)
    tar = MultiScaleTAR((24, 32, 96, 320), dim=64, variant="cd2",
                        return_difference=True, stem_dim=16).eval()
    feats = [(torch.randn(1, c, s, s), torch.randn(1, c, s, s))
             for c, s in ((24, 64), (32, 32), (96, 16), (320, 8))]
    with torch.no_grad():
        fused, difference = tar([f[0] for f in feats], [f[1] for f in feats],
                                stem=torch.randn(1, 16, 128, 128))
    assert [tuple(t.shape[-2:]) for t in fused] == [(64, 64), (32, 32), (16, 16), (8, 8)]
    assert len(difference) == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}")
