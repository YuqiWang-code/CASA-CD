"""Tests for the minor-goal arms: multi-scale difference pooling and difference retrieval."""
import copy
import unittest

import torch

from model.casa_net import CASANet
from model.str_tar import MultiScaleTAR, RepSPPBlock, SPPTARStage, embed_depthwise_kernel
from model.layers.difference_context import DifferenceContext
from model.layers.hyperbolic_context import HyperbolicContext


def count(model):
    return sum(p.numel() for p in model.parameters())


class RepSPPBlockTest(unittest.TestCase):
    def test_fold_matches_training_path(self):
        torch.manual_seed(0)
        block = RepSPPBlock(16).eval()
        with torch.no_grad():
            block.bn_branches[1].weight.fill_(0.7)
            block.bn_branches[2].weight.fill_(0.4)
            block.bn_branches[3].weight.fill_(0.2)
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            reference = block(x)
        folded = copy.deepcopy(block).switch_to_deploy()
        with torch.no_grad():
            result = folded(x)
        self.assertLess(float((reference - result).abs().max()), 2e-4)

    def test_kernel_embedding_keeps_the_weight_device(self):
        # A meta tensor exposes the device bug that only shows up on CUDA: the
        # padding buffer must be created on the kernel's own device.
        weight = torch.zeros(4, 1, 3, 3, device="meta")
        embedded = embed_depthwise_kernel(weight, 7)
        self.assertEqual(embedded.device, weight.device)
        self.assertEqual(tuple(embedded.shape), (4, 1, 7, 7))

    def test_summed_branches_are_centred(self):
        torch.manual_seed(1)
        block = RepSPPBlock(8)
        weight, bias = block.get_equivalent_kernel()
        target = block.max_kernel
        expected = torch.zeros_like(weight)
        expected_bias = torch.zeros_like(bias)
        for conv, bn in zip(block.dw_branches, block.bn_branches):
            k = conv.weight.shape[-1]
            offset = (target - k) // 2
            scale = (bn.weight.detach().double()
                     / torch.sqrt(bn.running_var.detach().double() + bn.eps))
            branch = conv.weight.detach().double() * scale.view(-1, 1, 1, 1)
            expected[..., offset:offset + k, offset:offset + k] += branch
            expected_bias += (bn.bias.detach().double()
                              - bn.running_mean.detach().double() * scale)
        self.assertLess(float((weight - expected).abs().max()), 1e-10)
        self.assertLess(float((bias - expected_bias).abs().max()), 1e-10)

        # Everything outside the embedded kernels stays exactly zero.
        mask = torch.zeros(target, target, dtype=torch.float64)
        for k in block.kernels:
            offset = (target - k) // 2
            mask[offset:offset + k, offset:offset + k] = 1
        self.assertEqual(float((weight * (1 - mask)).abs().max()), 0.0)
        self.assertEqual(tuple(bias.shape), (8,))

    def test_residual_and_identity_initialisation(self):
        block = RepSPPBlock(8)
        x = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            y = block(x)
        # Wider branches start switched off, so only the 1x1 identity path is active.
        self.assertEqual(float(block.bn_branches[1].weight.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(y).all())


class SPPStageTest(unittest.TestCase):
    def test_returns_fused_and_difference_streams(self):
        stage = SPPTARStage(24, 64)
        pre, post = torch.randn(2, 24, 64, 64), torch.randn(2, 24, 64, 64)
        fused, difference = stage(pre, post)
        self.assertEqual(tuple(fused.shape), (2, 64, 64, 64))
        self.assertEqual(tuple(difference.shape), (2, 64, 64, 64))
        with torch.no_grad():
            identical = stage(pre, pre)[1]
        self.assertEqual(float(identical.abs().max()), 0.0)

    def test_tar_variant_returns_difference(self):
        tar = MultiScaleTAR((24, 32, 96, 320), dim=64, variant="spp", return_difference=True)
        pre = [torch.randn(1, c, s, s) for c, s in ((24, 64), (32, 32), (96, 16), (320, 8))]
        post = [torch.randn(1, c, s, s) for c, s in ((24, 64), (32, 32), (96, 16), (320, 8))]
        fused, difference = tar(pre, post)
        self.assertEqual([tuple(x.shape) for x in fused],
                         [(1, 64, 64, 64), (1, 64, 32, 32), (1, 64, 16, 16), (1, 64, 8, 8)])
        self.assertEqual([tuple(x.shape) for x in difference], [tuple(x.shape) for x in fused])
        self.assertEqual(MultiScaleTAR((24, 32, 96, 320), dim=64, variant="spp")(pre, post)[0].shape[1], 64)


class DifferenceContextTest(unittest.TestCase):
    def test_requires_difference_stream(self):
        context = DifferenceContext()
        tar = [torch.zeros(1, 64, 64, 64), torch.zeros(1, 64, 32, 32),
               torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 8, 8)]
        with self.assertRaises(ValueError):
            context(tar)
        with self.assertRaises(ValueError):
            context(tar, difference=tar[:2])

    def test_keys_follow_the_difference_not_the_feature(self):
        torch.manual_seed(0)
        context = DifferenceContext().eval()
        tar = [torch.randn(1, 64, s, s) for s in (64, 32, 16, 8)]
        difference = [torch.randn(1, 64, s, s) for s in (64, 32, 16, 8)]
        captured = {}
        original = context.prototype_features

        def spy(normalized, difference=None):
            features = original(normalized, difference)
            captured["features"] = features
            return features

        context.prototype_features = spy
        with torch.no_grad():
            context(tar, difference)
        reference = captured["features"]
        with torch.no_grad():
            context(tar, [d.roll(shifts=1, dims=-1) for d in difference])
        changed = captured["features"]
        self.assertFalse(torch.allclose(reference[0], changed[0], atol=1e-6))
        # Layer normalisation makes the prototypes scale invariant; only the
        # spatial content of the difference may change them.
        with torch.no_grad():
            context(tar, [d * 3.0 for d in difference])
        self.assertTrue(torch.allclose(reference[0], captured["features"][0], atol=1e-5))

    def test_inherits_geometry_and_readout(self):
        context = DifferenceContext()
        self.assertIsInstance(context, HyperbolicContext)
        self.assertEqual(context.tokens_per_scale, (128, 32, 32, 32))
        self.assertEqual(context.geometry, "hyperbolic_difference")
        self.assertEqual(count(context), count(HyperbolicContext()) + 2 * 64)


class NetworkTest(unittest.TestCase):
    ARMS = {
        ("none", "reparam"): (2042324, 1933891),
        ("none", "spp"): (2113492, 1993795),
        ("difference", "spp"): (2124905, 2005208),
    }

    def test_parameter_counts_and_deploy_fidelity(self):
        torch.manual_seed(0)
        a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
        for (module, tar_variant), (trainable, deployed) in self.ARMS.items():
            model = CASANet(None, module=module, tar_variant=tar_variant).eval()
            self.assertEqual(count(model), trainable, (module, tar_variant))
            folded = copy.deepcopy(model).switch_to_deploy()
            self.assertEqual(count(folded), deployed, (module, tar_variant))
            with torch.no_grad():
                reference, deployed_output = model(a, b), folded(a, b)
            self.assertLess(max(float((x - y).abs().max()) for x, y in zip(reference, deployed_output)),
                            2e-4)

    def test_difference_requires_spp(self):
        with self.assertRaises(ValueError):
            CASANet(None, module="difference", tar_variant="reparam")

    def test_difference_context_carries_gradient_after_gate_opens(self):
        torch.manual_seed(0)
        model = CASANet(None, module="difference", tar_variant="spp").train()
        a, b = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
        target = (torch.rand(2, 1, 128, 128) > 0.9).float()
        outputs = model(a, b)
        sum(torch.nn.functional.binary_cross_entropy(x, target) for x in outputs).backward()
        self.assertGreater(float(model.context.gamma_context.grad.norm()), 0.0)
        model.zero_grad()
        with torch.no_grad():
            model.context.gamma_context.fill_(0.5)
        outputs = model(a, b)
        sum(torch.nn.functional.binary_cross_entropy(x, target) for x in outputs).backward()
        for name in ("qk.weight", "v.weight", "out.weight", "norm_diff.weight"):
            grad = dict(model.context.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.norm()), 0.0, name)
        self.assertGreater(float(model.tar.stage1.spp.bn_branches[3].weight.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()

def test_spp1_keeps_one_branch():
    block = RepSPPBlock(64, kernels=(1,))
    assert block.max_kernel == 1
    assert len(block.dw_branches) == 1
    x = torch.randn(1, 64, 32, 32)
    block.eval()
    with torch.no_grad():
        out = block(x)
    assert torch.isfinite(out).all()
    # A single 1x1 branch must fold to the identity-wide pathway only.
    weight, bias = block.get_equivalent_kernel()
    assert weight.shape == (64, 1, 1, 1)


def test_spp1_arm_is_cheaper_than_spp():
    torch.manual_seed(0)
    spp = CASANet(module="none", tar_variant="spp")
    spp1 = CASANet(module="none", tar_variant="spp1")
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count(spp1) < count(spp)
    assert count(spp1) == 2095509


def test_difference_context_accepts_plain_difference_stream():
    torch.manual_seed(0)
    model = CASANet(module="difference", tar_variant="diff")
    model.eval()
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(a, b)
    assert torch.isfinite(outputs[0]).all()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == 1955361


def test_difference_context_still_requires_a_difference_stream():
    try:
        CASANet(module="difference", tar_variant="reparam")
    except ValueError as error:
        assert "temporal difference stream" in str(error)
    else:
        raise AssertionError("the difference context accepted a fused-only TAR")

def test_nested_aggregation_matches_the_2x2_parent_mean():
    torch.manual_seed(0)
    context = HyperbolicContext(channels=8, kv_hierarchy="nested")
    assert context.tokens_per_scale == (128, 32, 8, 2), context.tokens_per_scale
    prototypes = [torch.randn(1, 2 * grid ** 2, 8) for grid in (8, 4, 2, 1)]
    aggregated = context.aggregate_nested(prototypes)
    assert len(aggregated) == 4
    # child at scale 1 was built from the scale-0 grid, so check one token by hand
    parent = prototypes[0].reshape(1, 8, 8, 2, 8)
    expected = parent[:, 0:2, 0:2, 0, :].mean(dim=(1, 2)).reshape(1, 1, 8)
    actual = aggregated[1].reshape(1, 4, 4, 2, 8)[:, 0, 0, 0, :].reshape(1, 1, 8)
    assert 0.5 * (prototypes[1].reshape(1, 4, 4, 2, 8)[:, 0, 0, 0, :].reshape(1, 1, 8)
                  + expected).allclose(actual, atol=1e-6)


def test_support_cell_is_the_mass_alias():
    cell = HyperbolicContext(channels=8, support_mode="cell")
    mass = HyperbolicContext(channels=8, support_mode="mass")
    assert cell.support_prior == mass.support_prior == "mass"
    assert cell.tokens_per_scale == mass.tokens_per_scale
    assert count(cell) == count(mass)
    for mode in ("pixel", "none"):
        assert HyperbolicContext(channels=8, support_mode=mode).support_prior == mode
    try:
        HyperbolicContext(channels=8, support_mode="bogus")
    except ValueError as error:
        assert "unknown support_mode" in str(error)
    else:
        raise AssertionError("an unknown support mode was accepted")
