"""Check grid geometry, cross-scale connectivity and manifold gradients."""
import torch
from torch.nn import functional as F

from model.layers.pyramid_context import PyramidContext
from model.layers.poincare_ops import expmap0, logmap0, pairwise_distance, weighted_centroid


def pooling_equivalence():
    """Moving K/V projections must preserve values and assignment gradients."""
    op = PyramidContext().double()
    for size, grid in zip((64, 32, 16, 8), op.grid_sizes):
        feature = torch.randn(1, 64, size, size, dtype=torch.double, requires_grad=True)
        normalized = op.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        scores = torch.randn(1, 1, size, size, dtype=torch.double, requires_grad=True)
        weights = op.assignments(scores.sigmoid(), grid)
        pooled = op.pool_features(normalized, weights, grid)
        for layer in (op.k, op.v):
            before = torch.einsum("brtp,bhrpd->bhrtd", weights,
                                  op.regions(op.project_heads(layer, normalized), grid)).flatten(2, 3)
            after = op.project_prototypes(layer, pooled)
            assert torch.allclose(before, after, atol=1e-11, rtol=1e-9)
            direction = torch.randn_like(after)
            a = torch.autograd.grad((before * direction).sum(), (feature, scores, layer.weight),
                                    retain_graph=True)
            b = torch.autograd.grad((after * direction).sum(), (feature, scores, layer.weight),
                                    retain_graph=True)
            assert a[1].abs().sum() > 0
            assert all(torch.allclose(x, y, atol=1e-10, rtol=1e-8) for x, y in zip(a, b))
    print("pool-before-project values and feature/assignment/projection gradients PASS")


def value_readout():
    """Hyperbolic matching must retrieve the unbounded Euclidean value vectors."""
    op = PyramidContext(geometry="hyperbolic").eval()
    op.gamma.data.fill_(0.1)
    op.v.weight.data.mul_(20)
    features = [torch.randn(1, 64, n, n) for n in (64, 32, 16, 8)]
    captured = []
    hook = op.out.register_forward_pre_hook(lambda layer, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad():
        op(features)
        normalized = [op.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for x in features]
        probabilities = [op.guide(x).sigmoid() for x in normalized]
        weights = [op.assignments(p, grid) for p, grid in zip(probabilities, op.grid_sizes)]
        keys = torch.cat([op.compress_hyperbolic_keys(expmap0(op.project_heads(op.k, x)), w, grid)
                          for x, w, grid in zip(normalized, weights, op.grid_sizes)], 2)
        values = []
        for x, w, grid in zip(normalized, weights, op.grid_sizes):
            prototypes = op.pool_features(x, w, grid)
            linear = F.linear(prototypes, op.v.weight[:, :, 0, 0]) / 4
            values.append(linear.reshape(1, -1, 4, 16).transpose(1, 2))
        values = torch.cat(values, 2)
        assert values.norm(dim=-1).max() > 1, "value test must exercise outside-ball content"
        for index, x in enumerate(normalized):
            query = expmap0(op.project_heads(op.q, x)).flatten(2, 3)
            expected = (op.attention(query, keys, index) @ values).transpose(-1, -2).reshape(
                1, 64, x.shape[-2], x.shape[-1])
            assert torch.allclose(captured[index], expected, atol=1e-6, rtol=1e-5)
    hook.remove()
    print("hyperbolic Q/K attention with Euclidean V and direct readout PASS")


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    sizes = (64, 32, 16, 8)
    features = [torch.randn(1, 64, n, n, requires_grad=True) for n in sizes]
    for geometry in ("euclidean", "hyperbolic"):
        op = PyramidContext(geometry=geometry)
        weights = op.assignments(torch.full((1, 1, 64, 64), 0.5), 8)
        assert weights.shape == (1, 64, 2, 64)
        assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
        # Fine-scale regions span 8x8 pixels; coarse grids span four regions per axis.
        coordinate = torch.arange(64 * 64).reshape(1, 1, 64, 64, 1).float()
        regions = op.regions(coordinate, 8)
        assert regions[0, 0, 0, 0, 0] == 0 and regions[0, 0, 1, 0, 0] == 8
        assert regions[0, 0, 8, 0, 0] == 8 * 64
        assert op.tokens_per_scale == (128, 32, 32, 32)
        constant_q = torch.zeros(1, 4, 3, 16)
        constant_k = torch.zeros(1, 4, 224, 16)
        attention = op.attention(constant_q, constant_k, 0)
        mass = torch.stack([p.sum(-1) for p in attention.split(op.tokens_per_scale, -1)], -1)
        assert torch.allclose(mass, torch.full_like(mass, .25), atol=1e-6)
        op.scale_bias.data[0, 0, 0] = torch.log(torch.tensor(2.0))
        learned = op.attention(constant_q, constant_k, 0)
        assert torch.allclose(learned[0, 0, 0, :128].sum(), torch.tensor(.4), atol=1e-6)
        op.scale_bias.data.zero_()
        outputs, logits = op(features)
        assert all(torch.equal(x, y) for x, y in zip(features, outputs))
        assert [z.shape[-2:] for z in logits] == [(n, n) for n in sizes]
        op.gamma.data.fill_(0.1)
        op.collect_diag = True
        outputs, _ = op(features)
        assert all(x.shape == y.shape for x, y in zip(features, outputs))
        assert all(d["contexts"] == 224 and d["finite"] for d in op.last_diag.values())
        for index, (original, enhanced) in enumerate(zip(features, outputs), 1):
            diag = op.last_diag[f"stage{index}"]
            ratio = (enhanced-original).square().mean().sqrt() / original.square().mean().sqrt()
            assert abs(diag['residual_to_feature_rms']-float(ratio.detach())) < 1e-6
            assert 0 < diag['query_attention_tv'] < 1
            assert all(abs(sum(head)-1) < 1e-5 for head in diag['scale_mass'])
        gradients = torch.autograd.grad(outputs[0].square().mean(), features, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), "cross-scale path absent"
        op.zero_grad()
        sum(x.square().mean() for x in outputs).backward()
        for name, parameter in op.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        print(geometry, "224 prototypes, scale prior, residual diagnostics and cross-scale gradients PASS")
    pooling_equivalence()
    value_readout()
    # Exercise the reused geometry on four independent head dimensions.
    tangent = (torch.randn(1, 4, 3, 16, dtype=torch.double) * 0.03).requires_grad_()
    assert torch.allclose(logmap0(expmap0(tangent)), tangent, atol=1e-10)
    probability = torch.softmax(torch.randn(1, 4, 2, 3, dtype=torch.double), -1)
    assert torch.autograd.gradcheck(lambda x: weighted_centroid(probability, expmap0(x)), (tangent,))
    distance = pairwise_distance(expmap0(tangent), expmap0(tangent))
    assert torch.isfinite(distance).all() and torch.allclose(distance, distance.transpose(-1, -2))
    print("head-wise Exp/Log, distance and centroid gradients PASS")


if __name__ == "__main__":
    main()
