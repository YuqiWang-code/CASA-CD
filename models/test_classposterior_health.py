"""Posterior H/E task gradients, temporal inputs and deployment folding."""
import json

import torch

from model.casa_net import CASANet
from train_CASA import loss_function, make_optimizer, training_losses, verify_deployment


def check_variant(variant, image_a, image_b, target):
    torch.manual_seed(2333)
    model = CASANet(module=variant).train()
    optimizer = make_optimizer(model)
    initial = {}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        model.context.collect_diag = step == 0
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
            for index, (stage, values) in enumerate(model.diagnostics().items()):
                initial[stage] = {key: values[key] for key in ("feature_rms", "context_rms", "retrieved_rms", "retrieved_spatial_rms", "self_value_rms", "value_difference_rms", "gate_mean")}
                initial[stage]["gamma_gradient_rms"] = float(model.context.gamma_context.grad[index].square().mean().sqrt())
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    outputs, _ = model(image_a, image_b, return_aux=True)
    loss_function(outputs, target).backward()
    for name in ("qk", "v", "out", "gate", "local_dw", "local_out", "guide"):
        gradient = getattr(model.context, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all(), (variant, name)
        assert gradient.abs().sum() > 0, (variant, name)

    task_gradients = {name: float(getattr(model.context, name).weight.grad.square().mean().sqrt())
                      for name in ("qk", "v", "out", "gate", "local_dw", "local_out", "guide")}
    gradient = model.context.raw_compression_temperature.grad
    assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0

    task_gradients["compression_temperature"] = float(gradient.abs())
    model.eval()
    model.context.collect_diag = True
    with torch.no_grad():
        prediction = model.predict(image_a, image_b)
        diagnostics = model.diagnostics()
        for stage in diagnostics.values():
            assert stage["finite"] and stage["contexts"] == 224
            assert stage["posterior_shift_abs_mean"] > 0
            assert 0 <= stage["posterior_entropy"] <= 1
            assert stage["compression_empty_fraction"] == 0
            assert stage["prototype_reassignment_to_feature_rms"] > 0
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
    print(f"{variant} class posterior: residual startup, segmentation-only gradients, "
          f"temporal inputs, compression diagnostics and folding PASS", flush=True)
    result = dict(initial=initial, task_gradient_rms=task_gradients, deploy=report)
    print(json.dumps(dict(variant=variant, **result)), flush=True)
    return result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    for variant in ("hyperbolic", "matched"):
        check_variant(variant, a, b, target)


if __name__ == "__main__":
    main()
