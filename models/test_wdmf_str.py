"""Interface, gradient and deployment checks for the three WDMF-STR variants."""
import copy

import torch

from model.casa_wdmf_str_net import CASAWDMFSTRNet
from model.layers.change_context_qkv import ChangeContextQKV, compress_pair
from train_wdmf import learning_rate, loss_function


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    reference = None
    for variant in ["none", "hyperbolic", "qkv"]:
        torch.manual_seed(2333)
        model = CASAWDMFSTRNet(module=variant).eval()
        with torch.no_grad():
            features = model.encoder_fusion(*model.encoder(a))
            assert [x.shape for x in features] == [torch.Size([1, 64, n, n]) for n in [64, 32, 16, 8]]
            outputs = model(a, b)
            assert len(outputs) == 4 and all(x.shape == target.shape for x in outputs)
            if reference is None:
                reference = outputs[0]
            else:
                assert torch.equal(reference, outputs[0]), "variant common initialization changed"
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        for _ in range(3):
            optimizer.zero_grad()
            loss = loss_function(model(a, b), target)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(x).all() for x in gradients)
            optimizer.step()
        if model.stage_modules:
            assert all(abs(float(op.gamma)) > 0 for op in model.stage_modules)
            assert all(float(op.q.weight.grad.abs().sum()) > 0 for op in model.stage_modules)
        model.eval()
        deploy = copy.deepcopy(model).switch_to_deploy().eval()
        with torch.no_grad():
            before, after = model(a, b), deploy(a, b)
            delta = max(float((x-y).abs().max()) for x, y in zip(before, after))
            assert delta < 2e-4, delta
        print(f"{variant}: four-level shapes, common initialization, finite gradients, active residual and folding PASS ({delta:.2e})")
    a = torch.randn(1, 64, 8, 8)
    ca, cb = compress_pair(a, a)
    assert torch.allclose(ca, torch.nn.functional.avg_pool2d(a, 4), atol=1e-6)
    assert torch.equal(ca, cb)
    for variant in ["hyperbolic", "qkv"]:
        module = ChangeContextQKV(geometry=variant)
        with torch.no_grad():
            module.gamma.fill_(1)
        zero = torch.zeros(1, 64, 8, 8, requires_grad=True)
        sum(x.sum() for x in module(zero, zero)).backward()
        assert torch.isfinite(zero.grad).all()
    assert abs(learning_rate(5e-4, 0, 0, 40125)-5.225e-5) < 1e-12
    print("uniform context, zero-feature gradient and upstream warmup PASS")


if __name__ == "__main__":
    main()
