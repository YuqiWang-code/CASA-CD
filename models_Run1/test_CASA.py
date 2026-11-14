"""Native pyramid, shared initialization, guided gradients and folding checks."""
import torch

from model.casa_net import CASANet
from train_CASA import training_losses, make_optimizer, verify_deployment


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    reference = None
    for variant in ("none", "hyperbolic", "qkv"):
        torch.manual_seed(2333)
        model = CASANet(module=variant).eval()
        count = sum(p.numel() for p in model.parameters())
        assert count == (2047125 if variant == "none" else 2063581), count
        with torch.no_grad():
            encoded = model.encoder(a)[1:]
            assert [tuple(x.shape[1:]) for x in encoded] == [(24,64,64),(32,32,32),(96,16,16),(320,8,8)]
            outputs, guide = model(a, b, return_aux=True)
            assert len(outputs) == len(guide) == 4
            assert all(x.shape == target.shape for x in outputs)
            if reference is None:
                reference = outputs[0]
            else:
                assert torch.equal(reference, outputs[0]), "common path initialization differs"
        model.train()
        optimizer = make_optimizer(model)
        context_ids = {id(p) for p in model.context.parameters()}
        grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
        assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
        for group in optimizer.param_groups:
            assert all(group["weight_decay"] == (0.0 if id(p) in context_ids else 1e-4) for p in group["params"])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            outputs, guide = model(a, b, return_aux=True)
            loss, main_loss, auxiliary = training_losses(outputs, guide, target)
            assert torch.allclose(loss, main_loss + .2 * auxiliary)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
        assert model.context.guide.weight.grad.abs().sum() > 0
        if variant != "none":
            assert (model.context.gamma.abs() > 0).all()
            assert model.context.q.weight.grad.abs().sum() > 0
        model.eval()
        _, report = verify_deployment(model, torch.cat((a, b), 1))
        assert max(report["per_scale_max_diff"]) < 2e-4
        assert report["binary_disagreement"] == 0
        deployed = model.switch_to_deploy()
        assert sum(p.numel() for p in deployed.parameters()) == (1935810 if variant == "none" else 1952266)
        print(variant, "parameters", count, "guide, full gradients and deployment PASS")


if __name__ == "__main__":
    main()
