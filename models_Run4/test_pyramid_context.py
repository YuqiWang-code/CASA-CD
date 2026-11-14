"""Compression fidelity, constant fields and cross-scale task gradients."""
import torch

from model.layers.euclidean_context import EuclideanContext
from model.layers.hyperbolic_context import HyperbolicContext
from model.layers.pyramid_context import assignments, pool_features, highpass, regions


def compression_semantics():
    feature = torch.randn(1, 64, 8, 8, dtype=torch.double, requires_grad=True)
    probability = torch.zeros(1, 1, 8, 8, dtype=torch.double)
    probability[..., 0, 0] = 1
    weights, confidence = assignments(probability, 1)
    prototype = pool_features(feature, weights, 1)
    assert weights[0, 0, 0, 0] > .9999
    assert torch.allclose(prototype[:, 0], feature[..., 0, 0], atol=3e-4)
    assert torch.equal(confidence, torch.ones_like(confidence)), "sparse confident target lost prior mass"
    for value in (0., .01, .5, .99, 1.):
        weights, confidence = assignments(torch.full_like(probability, value), 1)
        assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]))
        prototype = pool_features(feature, weights, 1)
        assert torch.allclose(prototype[:, 0], prototype[:, 1], atol=1e-12)
        expected = torch.tensor([[max(value, 1e-6), max(1-value, 1e-6)]], dtype=torch.double)
        assert torch.allclose(confidence, expected, atol=1e-12)
    logits = torch.randn_like(probability, requires_grad=True)
    weights, _ = assignments(logits.sigmoid(), 2)
    pooled = pool_features(feature, weights, 2)
    grads = torch.autograd.grad(pooled.square().sum(), (feature, logits))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    operator = EuclideanContext().double()
    dense = operator.project_queries(feature)
    expected = torch.einsum("brtp,bhrpd->bhrtd", weights, regions(dense, 2)).flatten(2, 3)
    actual = operator.pool_keys(dense, pooled, weights, 2)
    assert torch.allclose(actual, expected, atol=1e-12), "pool/linear equivalence failed"
    print("sparse targets, unsupported classes, assignment gradients and pool/linear equivalence PASS")


def contexts():
    sizes = (32,16,8,4)
    for context_type in (EuclideanContext, HyperbolicContext):
        op = context_type()
        tar = [torch.randn(1,64,n,n,requires_grad=True) for n in sizes]
        output, aux = op(tar)
        assert all(torch.equal(x,y) for x,y in zip(output,tar))
        assert len(aux['guide_logits']) == 4
        with torch.no_grad():
            op.gamma_context.fill_(.1)
            op.gamma_detail.fill_(.1)
        constant = torch.randn(1,64,1,1)
        fields = [constant.expand(1,64,n,n).clone() for n in sizes]
        output, _ = op(fields)
        assert all(torch.allclose(x,y,atol=3e-6,rtol=1e-6) for x,y in zip(output,fields)), "constant field changed"
        assert highpass(torch.ones(1,64,8,8)).abs().max() == 0
        op.collect_diag = True
        output, _ = op(tar)
        gradients = torch.autograd.grad(output[0].square().mean(), tar, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), "cross-scale gradient missing"
        sum(x.square().mean() for x in output).backward()
        for name, parameter in op.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        for i in range(4):
            diag=op.last_diag[f'stage{i+1}']
            assert diag['finite'] and diag['contexts']==224
            assert diag['context_residual_to_feature_rms']>0
            if i<2:
                assert diag['local_residual_to_feature_rms']>0
        # Identical distances and class confidence should not prefer the larger source grid.
        query=op.project_queries(torch.zeros(1,64,1,3)).flatten(2,3)
        key=op.project_queries(torch.zeros(1,64,1,224)).flatten(2,3)
        attention=op.attention(query,key,torch.ones(1,224),0)
        mass=torch.stack([p.sum(-1) for p in attention.split(op.tokens_per_scale,-1)],-1)
        assert torch.allclose(mass,torch.full_like(mass,.25),atol=1e-6)
        print(context_type.__name__, 'identity, constant preservation, all parameter/cross-scale gradients PASS')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    compression_semantics()
    contexts()
