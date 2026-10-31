"""Refined H/E task gradients, temporal inputs and deployment folding."""
import torch

from model.casa_net import CASANet
from train_CASA import loss_function, make_optimizer, training_losses, verify_deployment


def check_variant(variant, image_a, image_b, target):
    torch.manual_seed(2333)
    model = CASANet(module=variant, refinement_beta=1.).train()
    optimizer = make_optimizer(model)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs, auxiliary = model(image_a, image_b, return_aux=True)
        loss, main, guide = training_losses(outputs, auxiliary, target, .1, 0.)
        torch.testing.assert_close(loss, main + .1 * guide)
        loss.backward()
        assert torch.isfinite(loss), variant
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if step == 0:
            # Zero residual scales learn first, before the internal branch weights.
            assert (model.context.gamma_context.grad.abs().sum(-1) > 0).all()
            assert (model.context.gamma_detail.grad.abs().sum(-1) > 0).all()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    outputs, _ = model(image_a, image_b, return_aux=True)
    loss_function(outputs, target).backward()
    for name in ("qk", "v", "out", "gate", "local_dw", "local_out", "guide"):
        gradient = getattr(model.context, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all(), (variant, name)
        assert gradient.abs().sum() > 0, (variant, name)

    model.eval()
    model.context.collect_diag = True
    with torch.no_grad():
        prediction = model.predict(image_a, image_b)
        diagnostics = model.diagnostics()
        for stage in diagnostics.values():
            assert stage["finite"] and stage["contexts"] == 224
            assert stage["refinement_beta"] == 1.
            assert 0 < stage["compression_weight_tv"] < 1
            assert stage["prototype_refinement_to_feature_rms"] > 0
        model.context.collect_diag = False
        for first, second in ((image_b, image_a), (image_a, image_a)):
            other = model.predict(first, second)
            assert other.shape == target.shape and torch.isfinite(other).all()
        # The shared TAR uses ordered concat/signed differences, so exchange is
        # an augmented input condition rather than an exact symmetry identity.
        assert torch.isfinite(prediction).all()
    _, report = verify_deployment(model, torch.cat((image_a, image_b), 1))
    assert max(report["per_scale_max_diff"]) < 2e-4
    assert report["binary_disagreement"] == 0
    print(f"{variant} beta=1: residual startup, segmentation-only gradients, "
          f"temporal inputs, compression diagnostics and folding PASS", flush=True)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    for variant in ("hyperbolic", "matched"):
        check_variant(variant, a, b, target)


if __name__ == "__main__":
    main()
