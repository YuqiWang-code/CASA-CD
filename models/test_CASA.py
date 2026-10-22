"""Interface, gradient and deployment checks for the three CASA variants."""
import torch

from model.casa_net import CASANet
from train_CASA import learning_rate, loss_function, make_optimizer, verify_deployment


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    reference = None
    for variant in ["none", "hyperbolic", "qkv"]:
        torch.manual_seed(2333)
        model = CASANet(module=variant).eval()
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
        optimizer = make_optimizer(model)
        module_ids = {id(p) for p in model.stage_modules.parameters()}
        grouped_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(grouped_ids) == len(set(grouped_ids)) == len(list(model.parameters()))
        for group in optimizer.param_groups:
            assert all(group["weight_decay"] == (0.0 if id(p) in module_ids else 1e-4)
                       for p in group["params"])
        for _ in range(3):
            optimizer.zero_grad()
            loss = loss_function(model(a, b), target)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(x).all() for x in gradients)
            optimizer.step()
        if model.stage_modules:
            assert all(abs(float(op.gamma)) > 0 for op in model.stage_modules)
            for op in model.stage_modules:
                projection = op.metric if hasattr(op, "metric") else op.q
                assert float(projection.weight.grad.abs().sum()) > 0
        model.eval()
        _, report = verify_deployment(model, torch.cat([a, b], 1))
        delta = max(report["per_scale_max_diff"])
        assert delta < 2e-4, delta
        print(f"{variant}: four-level shapes, common initialization, finite gradients, active residual and folding PASS ({delta:.2e})")
    assert abs(learning_rate(5e-4, 0, 0, 40125)-5.225e-5) < 1e-12
    print("upstream warmup PASS")


if __name__ == "__main__":
    main()
