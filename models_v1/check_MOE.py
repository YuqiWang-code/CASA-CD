"""Validate MoE geometry, reconstruction, residual routing, task gradients and folding."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path

import torch

from model.casa_net import CASANet
from model.layers.lorentz_ops import pairwise_squared_distance
from model.layers.moe_context import (
    EXPERTS, MODEL_VARIANTS, MoEContext, RetrievalExpert, haar_merge, haar_split,
)
from model.layers.pyramid_context import channel_norm
from train_CASA import make_optimizer, source_files, training_losses, verify_deployment


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def wavelet_checks(device):
    errors = []
    for size in (4, 8, 16, 32, 64):
        x = torch.randn(2, 3, size, size, device=device, requires_grad=True)
        low, high = haar_split(x)
        restored = haar_merge(low, high)
        error = float((restored - x).detach().abs().max())
        require(error < 1e-6, f"Haar reconstruction error at {size}: {error}")
        energy = low.square().sum() + high.square().sum()
        require(torch.allclose(energy, x.square().sum(), rtol=2e-6), "Haar energy is not preserved")
        negative_low, negative_high = haar_split(-x)
        require(torch.equal(negative_low, -low) and torch.equal(negative_high, -high),
                "Haar subbands must retain signed coefficients")
        objective = torch.randn_like(x)
        gradient, = torch.autograd.grad((restored * objective).sum(), x)
        require(torch.allclose(gradient, objective, atol=3e-7, rtol=2e-6),
                "Haar inverse has an incorrect task gradient")
        errors.append(error)
    low, high = haar_split(torch.ones(1, 2, 8, 8, device=device))
    require(torch.equal(low, torch.full_like(low, 2)) and torch.count_nonzero(high) == 0,
            "A constant feature must have zero high-frequency coefficients")
    # Isolated horizontal, vertical and diagonal patterns must occupy distinct bands.
    patterns = torch.tensor([[[1., -1.], [1., -1.]], [[1., 1.], [-1., -1.]],
                             [[1., -1.], [-1., 1.]]], device=device)[:, None]
    low, high = haar_split(patterns)
    require(torch.count_nonzero(low) == 0, "Signed edge patterns leaked into the constant band")
    require(torch.equal(high[:, 0, :, 0, 0], 2 * torch.eye(3, device=device)),
            "Directional Haar bands are mixed or reordered")
    return dict(passed=True, tested_feature_sizes=[4, 8, 16, 32, 64],
                max_reconstruction_error=max(errors), signed_directional_bands=True)


def geometry_checks(device):
    expert = RetrievalExpert("hyperbolic").to(device=device, dtype=torch.float64)
    projection = torch.randn(2, 1, 12, 32, device=device, dtype=torch.float64, requires_grad=True)
    occupancy = torch.linspace(0, 1, 12, device=device, dtype=torch.float64).reshape(1, 1, 12, 1)
    point = expert.point(projection, occupancy)
    manifold = point[..., 0].square() - point[..., 1:].square().sum(-1)
    require(torch.allclose(manifold, torch.ones_like(manifold), atol=1e-12),
            "Hyperbolic query left the unit Lorentz manifold")
    expected_radius = torch.asinh(1 + occupancy[..., 0]).expand_as(manifold)
    require(torch.allclose(torch.asinh(point[..., 1:].norm(dim=-1)), expected_radius, atol=1e-12),
            "Hyperbolic radius does not correspond to guide occupancy")
    require(torch.allclose(point, expert.point(3 * projection, occupancy), atol=1e-12),
            "Projection magnitude unexpectedly changed the occupancy radius")
    distance = pairwise_squared_distance(point, point)
    oracle_cosh = point[..., :1] @ point[..., :1].transpose(-1, -2)
    oracle_cosh -= point[..., 1:] @ point[..., 1:].transpose(-1, -2)
    oracle = torch.acosh(oracle_cosh.clamp_min(1)).square()
    require(torch.allclose(distance, oracle, atol=2e-12, rtol=1e-10),
            "Hyperbolic retrieval distance disagrees with its Lorentz definition")
    gradient, = torch.autograd.grad(distance.square().mean(), projection)
    require(torch.isfinite(gradient).all() and gradient.abs().sum() > 0,
            "Hyperbolic retrieval lost finite task gradients")
    return dict(passed=True, max_manifold_error=float((manifold - 1).detach().abs().max()),
                max_distance_error=float((distance - oracle).detach().abs().max()))


def routing_checks(variant, device):
    if variant == "none":
        return dict(passed=True, routes=["tar"], residual_mode="identity")
    torch.manual_seed(2333)
    context = MoEContext(variant).to(device)
    tar = [torch.randn(2, 64, size, size, device=device) for size in (32, 16, 8, 4)]
    output, _ = context(tar)
    require(all(torch.equal(a, b) for a, b in zip(tar, output)),
            f"{variant}: zero gamma did not preserve the TAR identity")
    expected = tuple(name for name in EXPERTS if variant == name or
                     variant not in EXPERTS and variant != f"no_{name}")
    require(tuple(context.experts) == expected, f"{variant}: ablation still contains an excluded expert")
    require(context.route_names == ("tar",) + expected, f"{variant}: route names do not match experts")
    require([name for name, _ in context.named_parameters() if "gamma" in name] == ["gamma_context"],
            f"{variant}: correction has nested zero residual amplitudes")
    captured = {name: [] for name in expected}
    handles = []
    for name, expert in context.experts.items():
        def capture(module, inputs, result, name=name):
            captured[name].append(result if name == "frequency" else result[0])
        handles.append(expert.register_forward_hook(capture))
    with torch.no_grad():
        context.gamma_context.fill_(.1)
    context.collect_diag = True
    output, _ = context(tar)
    for handle in handles:
        handle.remove()
    errors = []
    for stage, (original, actual) in enumerate(zip(tar, output)):
        feature = channel_norm(context.norm, original)
        routes = context.route_weights(feature)
        require(torch.isfinite(routes).all() and (routes >= 0).all(), f"{variant}: invalid route weight")
        require(torch.allclose(routes.sum(1), torch.ones_like(routes[:, 0]), atol=2e-7),
                f"{variant}: route probability mass is not one")
        if context.routing_mode == "single":
            require(torch.count_nonzero(routes[:, 0]) == 0 and torch.all(routes[:, 1] == 1),
                    f"{variant}: single-expert control is attenuated by TAR routing")
        elif context.routing_mode == "equal":
            require(torch.all(routes == .25), "Fixed fusion must assign one quarter to each candidate")
        weighted = sum(routes[:, index:index+1] * captured[name][stage]
                       for index, name in enumerate(context.route_names) if name != "tar")
        reconstructed = original + context.gamma_context[stage][None, :, None, None] * weighted
        error = float((actual - reconstructed).detach().abs().max())
        require(error < 2e-6, f"{variant}: residual identity or expert weighting is duplicated")
        errors.append(error)
        diag = context.last_diag[f"stage{stage+1}"]
        require(diag["finite"] and abs(sum(diag["route_mass"].values()) - 1) < 1e-6,
                f"{variant}: route diagnostics disagree with forward routing")
        for name, attention in diag["expert_attention"].items():
            require(abs(attention["context_mass_mean"] - 1) < 1e-6,
                    f"{variant}/{name}: conditional retrieval unexpectedly controls adoption")
    return dict(passed=True, routes=list(context.route_names), routing_mode=context.routing_mode,
                max_residual_formula_error=max(errors))


def gradient_group_norm(module):
    return sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
               if parameter.grad is not None)


def network_checks(variant, device, reference):
    torch.manual_seed(2333)
    model = CASANet(module=variant).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(511)
    image = torch.randn(2, 6, 128, 128, device=device, generator=generator)
    target = (torch.rand(2, 1, 128, 128, device=device, generator=generator) > .8).float()
    with torch.no_grad():
        outputs, aux = model(image[:, :3], image[:, 3:], return_aux=True)
        require(len(outputs) == len(aux["guide_logits"]) == 4, "Deep supervision lost a scale")
        require(all(output.shape == target.shape and torch.isfinite(output).all() for output in outputs),
                f"{variant}: prediction shape or finiteness failed")
        prediction = model.predict(image[:, :3], image[:, 3:])
        require(torch.equal(prediction, outputs[0].sigmoid()), "predict does not use the final prediction head")
        if reference is not None:
            require(torch.equal(prediction, reference), f"{variant}: shared initial model differs from CTRL")
        else:
            reference = prediction.detach().clone()
        same = model.predict(image[:, :3], image[:, :3])
        reversed_prediction = model.predict(image[:, 3:], image[:, :3])
        require(torch.isfinite(same).all() and torch.isfinite(reversed_prediction).all(),
                f"{variant}: identical-image or temporal-exchange output is non-finite")
        exchange_difference = float((prediction - reversed_prediction).abs().mean())
    del outputs, aux, prediction, same, reversed_prediction
    model.train()
    optimizer = make_optimizer(model)
    train_losses = []
    gradients = {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs, aux = model(image[:, :3], image[:, 3:], return_aux=True)
        loss, main, guide = training_losses(outputs, aux, target)
        require(torch.isfinite(loss), f"{variant}: non-finite task loss")
        # Check the segmentation task itself reaches active experts after gamma learns.
        if step == 1 and variant != "none":
            task_parameters = [(f"expert:{name}", list(expert.parameters()))
                               for name, expert in model.context.experts.items()]
            if hasattr(model.context, "router"):
                task_parameters.append(("router", list(model.context.router.parameters())))
            flattened = [p for _, group in task_parameters for p in group]
            task_gradients = torch.autograd.grad(main, flattened, retain_graph=True, allow_unused=True)
            position = 0
            for name, group in task_parameters:
                values = task_gradients[position:position+len(group)]
                position += len(group)
                require(all(g is None or torch.isfinite(g).all() for g in values),
                        f"{variant}/{name}: non-finite segmentation gradient")
                magnitude = sum(float(g.abs().sum()) for g in values if g is not None)
                require(magnitude > 0, f"{variant}/{name}: segmentation gradient is blocked")
                gradients[name] = magnitude
        loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
                f"{variant}: non-finite model gradient")
        if variant != "none":
            require((model.context.gamma_context.grad.abs().sum(-1) > 0).all(),
                    f"{variant}: a scale cannot start learning its residual")
        optimizer.step()
        train_losses.append(float(loss.detach()))
    require(gradient_group_norm(model.encoder) > 0 and gradient_group_norm(model.tar) > 0 and
            gradient_group_norm(model.decoder) > 0, f"{variant}: model-body task gradient is blocked")
    model.eval()
    deployed, fold = verify_deployment(model, image[:1])
    require(max(fold["per_scale_max_diff"]) <= fold["tolerance"],
            f"{variant}: an auxiliary scale failed strict deployment folding")
    result = dict(passed=True, train_losses=train_losses, task_gradient_l1=gradients,
                  temporal_exchange_mean_abs_difference=exchange_difference,
                  temporal_symmetry="not imposed: shared TAR concatenates ordered A/B with a signed difference",
                  train_params=sum(p.numel() for p in model.parameters()),
                  deploy_params=sum(p.numel() for p in deployed.parameters()),
                  context_params=sum(p.numel() for p in model.context.parameters()),
                  deployment=fold, configuration=model.context_metadata())
    del model, deployed, optimizer, outputs, aux, loss, main, guide, image, target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--modules", nargs="+", choices=MODEL_VARIANTS, default=MODEL_VARIANTS)
    args = parser.parse_args()
    if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=1 for this GPU-1 exploration")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2333)
    device = torch.device(args.device)
    directory = Path(__file__).resolve().parent
    require(Path(inspect.getfile(CASANet)).resolve().is_relative_to(directory),
            "Imported model comes from the mainline instead of models_v1")
    require(all(path.startswith("models_v1/") for path in source_files()),
            "Training archive source list points outside models_v1")
    report = dict(passed=False, device=str(device), physical_gpu=1 if args.device == "cuda" else None,
                  source_directory=str(directory), wavelet=wavelet_checks(device),
                  geometry=geometry_checks(device), variants={})
    reference = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # CTRL establishes the shared initialization even when a selected subset is requested.
    if args.modules[0] != "none":
        _, reference = network_checks("none", device, None)
    for variant in args.modules:
        routing = routing_checks(variant, device)
        result, reference = network_checks(variant, device, reference)
        result["routing"] = routing
        report["variants"][variant] = result
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[CHECK] {variant} {json.dumps(result)}", flush=True)
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[CHECK] All requested variants passed", flush=True)


if __name__ == "__main__":
    main()
