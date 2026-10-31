"""Paired centered-readout probes with separate segmentation and posterior gradients."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


DATASETS = {"LEVIR": "LEVIR-CD-256", "SYSU": "SYSU-CD-256", "WHU": "WHU-CD-256"}
PRETRAINED = Path("/home/yqwang/projects/CASA-CD/pretrained_weight/mobilenet_v2-b0353104.pth")


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def paired_loader(make_loader, dataset, split, seed):
    loader = make_loader(dataset, split, batch_size=32, workers=1)
    generator = torch.Generator().manual_seed(seed)
    loader.generator = generator
    if split == "train":
        # Worker seeds and permutation use the same independent stream in every arm.
        loader.sampler.generator = generator
    return loader


def task_gradients(model, image, target, training_losses):
    """Measure segmentation gradients with auxiliary supervision excluded."""
    model.eval()
    model.zero_grad(set_to_none=True)
    captures = {"context": [], "local": []}
    posterior_nodes = []
    implementation = sys.modules[type(model.context).__module__]
    original_compress = implementation.compress_posterior

    def capture_compression(*args, **kwargs):
        output = original_compress(*args, **kwargs)
        nodes = {name: output[2][name] for name in ("posterior_logits", "posterior")}
        for node in nodes.values():
            node.retain_grad()
        posterior_nodes.append(nodes)
        return output

    implementation.compress_posterior = capture_compression

    def capture(kind):
        def hook(layer, inputs, output):
            inputs[0].retain_grad()
            output.retain_grad()
            captures[kind].append((inputs[0], output))
        return hook

    hooks = []
    if model.module in ("hyperbolic", "matched"):
        hooks.append(model.context.out.register_forward_hook(capture("context")))
        hooks.append(model.context.local_out.register_forward_hook(capture("local")))
    model.context.collect_diag = False
    try:
        outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
        _, main_loss, _ = training_losses(outputs, auxiliary, target)
        main_loss.backward()
    finally:
        implementation.compress_posterior = original_compress
        for hook in hooks:
            hook.remove()

    parameters = {
        name: dict(gradient_rms=rms(parameter.grad) if parameter.grad is not None else 0.0,
                   weight_rms=rms(parameter))
        for name, parameter in model.named_parameters()
    }
    nonfinite = [name for name, item in parameters.items()
                 if not all(math.isfinite(value) for value in item.values())]
    if nonfinite:
        raise FloatingPointError(f"non-finite task gradients or weights: {nonfinite}")
    branches = {
        kind: [dict(stage=index + 1, input_rms=rms(content), output_rms=rms(update),
                    input_gradient_rms=rms(content.grad) if content.grad is not None else 0.0,
                    output_gradient_rms=rms(update.grad) if update.grad is not None else 0.0)
               for index, (content, update) in enumerate(values)]
        for kind, values in captures.items()
    }
    required = [name for name, parameter in model.named_parameters()
                if name.startswith("context.") and name.endswith("weight")
                and parameter.ndim >= 2 and not name.startswith("context.guide.")]
    inactive = [name for name in required if parameters[name]["gradient_rms"] <= 0]
    if model.module in ("hyperbolic", "matched"):
        if inactive:
            raise RuntimeError(f"context weights without segmentation gradients: {inactive}")
        if len(branches["context"]) != 4 or len(branches["local"]) != 2:
            raise RuntimeError("expected four context and two local branch outputs")
        for branch in branches["context"] + branches["local"]:
            if not all(math.isfinite(value) for value in branch.values()):
                raise FloatingPointError("non-finite branch output or task gradient")
            if branch["input_gradient_rms"] <= 0 or branch["output_gradient_rms"] <= 0:
                raise RuntimeError("branch has no segmentation task gradient")
    body_gradients = {
        group: sum(item["gradient_rms"] for name, item in parameters.items()
                   if name.startswith(group + "."))
        for group in ("encoder", "tar", "decoder", "head")
    }
    if any(value <= 0 for value in body_gradients.values()):
        raise RuntimeError(f"body branch has no segmentation gradient: {body_gradients}")
    posterior_path = [dict(stage=index+1,
        posterior_logit_rms=rms(nodes["posterior_logits"]),
        posterior_logit_gradient_rms=rms(nodes["posterior_logits"].grad) if nodes["posterior_logits"].grad is not None else 0.,
        membership_gradient_rms=rms(nodes["posterior"].grad) if nodes["posterior"].grad is not None else 0.)
        for index, nodes in enumerate(posterior_nodes)]
    if len(posterior_path) != 4 or not all(math.isfinite(value) for row in posterior_path for value in row.values()):
        raise FloatingPointError("invalid posterior main-task gradient path")
    model.zero_grad(set_to_none=True)
    return dict(posterior_path=posterior_path, main_loss=float(main_loss.detach()), batch_size=len(image),
                loss_source="segmentation main loss only", parameters=parameters,
                branches=branches, body_gradient_rms_sum=body_gradients, finite=True,
                required_context_weights_nonzero=(not inactive) if model.module != "none" else None)



def posterior_geometry_gradients(model, image, target, guide_loss, objective):
    """Measure posterior BCE separately; its training weight is explicit."""
    model.eval()
    model.zero_grad(set_to_none=True)
    tar = model.tar(model.encoder(image[:, :3])[1:], model.encoder(image[:, 3:])[1:])
    _, auxiliary = model.context(tar, return_aux=True)
    raw = guide_loss(auxiliary["guide_logits"], target)
    posterior = guide_loss(auxiliary["posterior_logits"], target)
    parameters = dict(model.named_parameters())
    names = ("context.qk.weight", "context.raw_compression_temperature", "context.guide.weight")
    gradients = torch.autograd.grad(posterior, [parameters[name] for name in names])
    measured = {name: rms(gradient) for name, gradient in zip(names, gradients)}
    if not all(math.isfinite(value) for value in measured.values()):
        raise FloatingPointError("non-finite posterior BCE geometry gradients")
    return dict(loss_source="posterior BCE only, diagnostic backward",
                guide_objective=objective,
                posterior_training_weight=.05 if objective == "half_raw_half_posterior_area_bce" else 0.,
                raw_guide_bce=float(raw.detach()), posterior_bce=float(posterior.detach()),
                gradients=measured, finite=True,
                geometry_nonzero=all(measured[name] > 0 for name in names[:2]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--module", choices=("hyperbolic", "matched"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--retrieval-weight", type=float, default=0.)
    parser.add_argument("--compression-temperature", type=float, default=.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    if args.compression_temperature <= 0:
        parser.error("--compression-temperature must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.models_dir = args.models_dir.resolve()
    sys.path.insert(0, str(args.models_dir))
    from train_CASA import (PROTOCOL, GUIDE_OBJECTIVE, DATA_PROTOCOL, GUIDE_WEIGHT, context_decay_names,
                            guide_loss, evaluate_auxiliary,
                            evaluate, learning_rate, make_loader, make_optimizer,
                            training_losses, update_learning_rate)
    from model.casa_net import CASANet
    if GUIDE_OBJECTIVE not in ("raw_area_bce", "half_raw_half_posterior_area_bce"):
        raise ValueError("probe source must declare the raw or equal-budget posterior objective")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "history.json").exists():
        raise RuntimeError("probe output already contains a training history; use a fresh directory")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    dataset = Path("/share_datasets/CD") / DATASETS[args.dataset]
    train = paired_loader(make_loader, dataset, "train", args.seed)
    validation = paired_loader(make_loader, dataset, "val", args.seed + 1)
    model = CASANet(str(PRETRAINED), args.module, context_seed=2333).to(device)
    if args.module != "none":
        with torch.no_grad():
            model.context.raw_compression_temperature.fill_(math.log(math.expm1(args.compression_temperature)))
    optimizer = make_optimizer(model, encoder_lr_mult=1.0)
    total_steps = math.ceil(40000 / len(train)) * len(train)
    context = model.context
    manifest = dict(
        protocol=PROTOCOL, purpose="centered readout and posterior supervision paired probe", formal_result=False,
        guide_objective=GUIDE_OBJECTIVE,
        guide_component_weights=(dict(raw=.05, posterior=.05) if GUIDE_OBJECTIVE == "half_raw_half_posterior_area_bce"
                                 else dict(raw=.1, posterior=0.)),
        models_dir=str(args.models_dir), module=args.module, dataset=DATASETS[args.dataset],
        seed=args.seed, context_initialization_seed=2333, physical_gpu=1,
        epochs=args.epochs, batch_size=32, workers=1, train_batches=len(train),
        train_samples=len(train.dataset), validation_samples=len(validation.dataset),
        train_split="train", evaluation_split="val", selection="best val F1, ties update",
        scheduler_total=total_steps, requested_formal_steps=40000, base_lr=5e-4,
        encoder_lr_mult=1.0, guide_weight=.1, retrieval_weight=args.retrieval_weight, preprocessing=DATA_PROTOCOL,
        loader_randomness="independent generator shared by train sampler and worker seeding",
        train_loader_seed=args.seed, validation_loader_seed=args.seed + 1,
        pretrained=str(PRETRAINED), initialization="fresh pretrained encoder and seeded new layers",
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        context_parameters=sum(parameter.numel() for parameter in context.parameters()),
        grid_sizes=list(context.grid_sizes), tokens_per_scale=list(context.tokens_per_scale),
        qk_geometry=context.geometry, value_geometry=context.value_geometry,
        optimizer_groups=[dict(name=group["group_name"], lr_scale=group["lr_scale"],
                               weight_decay=group["weight_decay"],
                               decoupled_weight_decay=group["decoupled_weight_decay"])
                          for group in optimizer.param_groups],
        context_projection_decay_parameters=sorted(context_decay_names(model)),
        precision=dict(input="float32", amp=False, cudnn_tf32=True, matmul_tf32=False),
        checkpoint_usage="probe analysis only; formal experiments start from fresh initialization",
    )
    if args.module != "none":
        manifest["compression_temperature_init"] = args.compression_temperature
    write_json(args.output_dir / "run_manifest.json", manifest)
    history = []
    started, step, best_f1, best_epoch = time.time(), 0, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tick = time.time()
        model.train()
        totals = dict(train_loss=0.0, main_loss=0.0, guide_loss=0.0,
                      raw_guide_bce=0.0, posterior_bce=0.0, gradient_norm_mean=0.0)
        gradient_max = 0.0
        for image, target in train:
            image, target = image.to(device).float(), target.to(device).float()
            lr = learning_rate(5e-4, epoch - 1, step, total_steps)
            update_learning_rate(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
            loss, main_loss, guide = training_losses(outputs, auxiliary, target, .1, args.retrieval_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            with torch.no_grad():
                raw_bce = float(guide_loss(auxiliary["guide_logits"], target))
                posterior_bce = float(guide_loss(auxiliary["posterior_logits"], target))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                                  error_if_nonfinite=True, foreach=True)
            optimizer.step()
            totals["train_loss"] += float(loss.detach())
            totals["main_loss"] += float(main_loss.detach())
            totals["guide_loss"] += float(guide.detach())
            totals["raw_guide_bce"] += raw_bce
            totals["posterior_bce"] += posterior_bce
            norm_value = float(norm)
            totals["gradient_norm_mean"] += norm_value
            gradient_max = max(gradient_max, norm_value)
            step += 1

        model.context.collect_diag = True
        validation_loss, scores, spatial = evaluate(model, validation, device,
                                                     return_diagnostics=True)
        diagnostics = copy.deepcopy(model.diagnostics())
        validation_auxiliary = evaluate_auxiliary(model, validation, device)
        entry = dict(epoch=epoch, step=step, lr=lr,
                     **{key: value / len(train) for key, value in totals.items()},
                     gradient_norm_max=gradient_max, validation_loss=validation_loss,
                     validation_metrics=scores, spatial=spatial,
                     validation_auxiliary=validation_auxiliary, guide_objective=GUIDE_OBJECTIVE,
                     first_validation_batch_diagnostics=diagnostics,
                     epoch_seconds=time.time() - tick, elapsed_seconds=time.time() - started)
        history.append(entry)
        write_json(args.output_dir / "history.json", history)
        checkpoint = dict(state_dict=model.state_dict(), manifest=manifest, epoch=epoch,
                          step=step, validation_metrics=scores)
        torch.save(checkpoint, args.output_dir / "last_model.pth")
        if scores["F1"] >= best_f1:
            best_f1, best_epoch = float(scores["F1"]), epoch
            torch.save(checkpoint, args.output_dir / "best_model.pth")
        print(f"[PROBE-EPOCH] epoch={epoch}/{args.epochs} step={step} "
              f"main={entry['main_loss']:.6f} guide={entry['guide_loss']:.6f} "
              f"val_loss={validation_loss:.6f} F1={scores['F1']:.6f} "
              f"P={scores['precision']:.6f} R={scores['recall']:.6f} "
              f"boundary1={spatial['boundary_1px']['F1']:.6f} "
              f"raw_BCE={entry['raw_guide_bce']:.6f} post_BCE={entry['posterior_bce']:.6f} "
              f"seconds={entry['epoch_seconds']:.1f}", flush=True)

    image, target = next(iter(validation))
    gradients = task_gradients(model, image[:2].to(device).float(),
                              target[:2].to(device).float(), training_losses)
    write_json(args.output_dir / "task_gradients.json", gradients)
    posterior_gradients = posterior_geometry_gradients(model, image[:2].to(device).float(),
                                                      target[:2].to(device).float(), guide_loss, GUIDE_OBJECTIVE)
    write_json(args.output_dir / "posterior_gradients.json", posterior_gradients)
    result = dict(health_passed=True, finite_training=True, task_gradient_check=gradients["finite"],
                  posterior_gradient_check=posterior_gradients["finite"], guide_objective=GUIDE_OBJECTIVE, seed=args.seed,
                  criterion="finite training, context task gradients and separate posterior BCE gradients; no accuracy threshold",
                  protocol=PROTOCOL, module=args.module, dataset=DATASETS[args.dataset],
                  retrieval_weight=args.retrieval_weight,
                  epochs=args.epochs, steps=step, best_epoch=best_epoch, best_val_f1=best_f1,
                  last_three_mean_val_f1=sum(item["validation_metrics"]["F1"] for item in history[-3:])
                                         / len(history[-3:]),
                  elapsed_seconds=time.time() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2)
    if args.module != "none":
        result["compression_temperature_init"] = args.compression_temperature
    write_json(args.output_dir / "health_result.json", result)
    print(f"[PROBE-HEALTH-PASS] module={args.module} dataset={args.dataset} "
          f"epochs={args.epochs} best_val_F1={best_f1:.6f} "
          f"last3_val_F1={result['last_three_mean_val_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
