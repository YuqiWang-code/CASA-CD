"""Inspect real-data task learning, shared gradients and validation trends."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from train_CASA import (PROTOCOL, DATA_PROTOCOL, GUIDE_WEIGHT, learning_rate,
                        update_learning_rate, training_losses, make_loader,
                        make_optimizer, context_decay_names, evaluate)
from model.casa_net import CASANet


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def inspect(model, image, target):
    model.eval()
    model.context.collect_diag = True
    captures = {"context": [], "local": []}

    def capture(kind):
        def hook(layer, inputs, output):
            inputs[0].retain_grad()
            output.retain_grad()
            captures[kind].append((inputs[0], output))
        return hook

    hooks = []
    if model.module != "none":
        hooks.append(model.context.out.register_forward_hook(capture("context")))
        hooks.append(model.context.local_out.register_forward_hook(capture("local")))
    model.zero_grad(set_to_none=True)
    outputs, aux = model(image[:, :3], image[:, 3:], return_aux=True)
    model.context.collect_diag = False
    _, main, guide = training_losses(outputs, aux, target)
    shared = [(name, parameter) for name, parameter in model.named_parameters()
              if name.startswith("context.")]
    encoder_last = next((name, parameter) for name, parameter in reversed(list(model.named_parameters()))
                        if name.startswith("encoder.") and parameter.ndim >= 2)
    shared.append(encoder_last)
    guide_gradients = torch.autograd.grad(GUIDE_WEIGHT * guide, [p for _, p in shared],
                                         retain_graph=True, allow_unused=True)
    main.backward()
    for hook in hooks:
        hook.remove()
    report = model.diagnostics()
    for index, (content, update) in enumerate(captures["context"]):
        report[f"stage{index+1}"].update(
            branch_rms=rms(update), context_rms=rms(content),
            task_context_grad_rms=rms(content.grad), task_update_grad_rms=rms(update.grad))
    for index, (content, update) in enumerate(captures["local"]):
        report[f"stage{index+1}"].update(
            local_branch_rms=rms(update), task_local_grad_rms=rms(content.grad),
            task_local_update_grad_rms=rms(update.grad))
    gradients = {name: rms(p.grad) if p.grad is not None else 0.0
                 for name, p in model.context.named_parameters()}
    comparison = {}
    for (name, parameter), auxiliary in zip(shared, guide_gradients):
        task = parameter.grad
        task_rms = rms(task) if task is not None else 0.0
        aux_rms = rms(auxiliary) if auxiliary is not None else 0.0
        cosine = None
        if task_rms > 0 and aux_rms > 0:
            main_vector = task.detach().double().flatten()
            aux_vector = auxiliary.detach().double().flatten()
            cosine = float(main_vector.dot(aux_vector) / main_vector.norm() / aux_vector.norm())
        comparison[name] = dict(main_rms=task_rms, weighted_guide_rms=aux_rms,
            guide_to_main_ratio=aux_rms/task_rms if task_rms > 0 else None, cosine=cosine)
        if not math.isfinite(task_rms) or not math.isfinite(aux_rms):
            raise FloatingPointError(f"non-finite task/guide gradient: {name}")
    if not all(math.isfinite(value) for value in gradients.values()):
        raise FloatingPointError("non-finite context task gradient")
    if model.module != "none":
        required = [name for name, parameter in model.context.named_parameters()
                    if parameter.ndim >= 2 and name.endswith("weight") and not name.startswith("guide.")]
        inactive = [name for name in required if gradients[name] <= 0]
        if inactive:
            raise RuntimeError(f"segmentation task gradient absent: {inactive}")
        for stage in report.values():
            if not stage["finite"] or stage["task_context_grad_rms"] <= 0:
                raise RuntimeError("inactive pyramid stage")
    model.zero_grad(set_to_none=True)
    return dict(stages=report, task_grad_rms=gradients, task_vs_guide_gradients=comparison,
                projection_weight_rms={name: rms(p) for name, p in model.context.named_parameters()
                                       if p.ndim >= 2},
                sample_main_loss=float(main.detach()), sample_guide_loss=float(guide.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["none", "hyperbolic", "qkv"], required=True)
    parser.add_argument("--dataset", choices=["LEVIR-CD-256", "SYSU-CD-256"], required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2333); np.random.seed(2333); torch.manual_seed(2333); torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = Path("/share_datasets/CD") / args.dataset
    train = make_loader(dataset, "train", 32, 1)
    validation = make_loader(dataset, "val", 2, 0)
    model = CASANet(str(ROOT / "pretrained_weight/mobilenet_v2-b0353104.pth"), args.module).cuda()
    initial_weights = {name: rms(p) for name, p in model.context.named_parameters() if p.ndim >= 2}
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        image, target = next(iter(validation))
    image, target = image.cuda().float(), target.cuda().float()
    total = math.ceil(40000 / len(train)) * len(train)
    manifest = dict(protocol=PROTOCOL, module=args.module, dataset=args.dataset,
                    epochs=args.epochs, batch_size=32, scheduler_total=total,
                    guide_weight=GUIDE_WEIGHT, encoder_lr_mult=1.0, evaluation_split="val",
                    context_projection_weight_decay_mode="decoupled",
                    context_projection_weight_decay=1e-4, body_weight_decay_mode="coupled_l2",
                    optimizer_groups=[dict(name=g["group_name"], weight_decay=g["weight_decay"],
                                           decoupled_weight_decay=g["decoupled_weight_decay"])
                                      for g in optimizer.param_groups],
                    projection_decay_parameters=sorted(context_decay_names(model)),
                    initial_projection_weight_rms=initial_weights, preprocessing=DATA_PROTOCOL, seed=2333,
                    grid_sizes=list(model.context.grid_sizes),
                    contexts_per_scale=list(model.context.tokens_per_scale),
                    joint_contexts=sum(model.context.tokens_per_scale),
                    compression_power=model.context.compression_power, support_power=model.context.support_power,
                    self_reference=model.context.self_reference,
                    qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        losses, main_losses, guide_losses, norms = [], [], [], []
        for image_train, target_train in train:
            image_train, target_train = image_train.cuda().float(), target_train.cuda().float()
            update_learning_rate(optimizer, learning_rate(5e-4, epoch, step, total))
            optimizer.zero_grad(set_to_none=True)
            outputs, aux = model(image_train[:, :3], image_train[:, 3:], return_aux=True)
            loss, main_loss, guide = training_losses(outputs, aux, target_train)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"probe loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            losses.append(float(loss.detach())); main_losses.append(float(main_loss.detach()))
            guide_losses.append(float(guide.detach())); norms.append(float(norm))
            step += 1
        report = inspect(model, image, target)
        val_loss, scores, spatial = evaluate(model, make_loader(dataset, "val", 32, 1),
                                            torch.device("cuda:0"), return_diagnostics=True)
        entry = dict(epoch=epoch+1, step=step, train_loss=sum(losses)/len(losses),
                     main_loss=sum(main_losses)/len(main_losses), guide_loss=sum(guide_losses)/len(guide_losses),
                     gradient_norm_mean=sum(norms)/len(norms), gradient_norm_max=max(norms),
                     validation_loss=val_loss, validation_metrics=scores, spatial=spatial, **report)
        history.append(entry)
        (args.output_dir / "health_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
    result = dict(passed=True, **manifest, steps=step, validation_loss=val_loss, validation_metrics=scores,
                  elapsed_seconds=time.time()-started, final=history[-1],
                  main_loss_first_to_last=[history[0]["main_loss"], history[-1]["main_loss"]],
                  validation_f1_history=[item["validation_metrics"]["F1"] for item in history],
                  validation_loss_history=[item["validation_loss"] for item in history],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), manifest=manifest,
                    step=step), args.output_dir / "probe_checkpoint.pth")
    (args.output_dir / "health_result.json").write_text(json.dumps(result, indent=2))
    print(f"[PROBE-PASS] {args.module} {args.dataset} epochs={args.epochs} steps={step}", flush=True)


if __name__ == "__main__":
    main()
