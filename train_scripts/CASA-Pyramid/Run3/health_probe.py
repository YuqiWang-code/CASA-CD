"""Observe real-data losses, task gradients and validation before formal runs."""
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
from train_CASA import (PROTOCOL, DATA_PROTOCOL, learning_rate, update_learning_rate,
                        loss_function, training_losses, make_loader, make_optimizer, evaluate, NO_DECAY_PROJECTIONS)
from model.casa_net import CASANet


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def inspect(model, image, target):
    model.eval()
    model.context.collect_diag = True
    captured = []
    def capture(layer, inputs, output):
        inputs[0].retain_grad()
        captured.append((inputs[0], output))
    hook = model.context.out.register_forward_hook(capture) if model.module != "none" else None
    model.zero_grad(set_to_none=True)
    outputs, aux = model(image[:, :3], image[:, 3:], return_aux=True)
    model.context.collect_diag = False
    total, main, guide, boundary = training_losses(outputs, aux, target)
    guide_gradient = torch.autograd.grad(guide, model.context.guide.weight, retain_graph=True)[0]
    edge_gradient = torch.autograd.grad(boundary, model.context.boundary.weight, retain_graph=True)[0]
    main.backward()
    if hook is not None:
        hook.remove()
    report = model.diagnostics()
    for index, (context, update) in enumerate(captured):
        report[f"stage{index+1}"].update(
            branch_rms=rms(update), context_rms=rms(context), task_context_grad_rms=rms(context.grad))
        if model.module == "qkv":
            report[f"stage{index+1}"].update(mean_content_rms=rms(context[:, :32]),
                contrast_content_rms=rms(context[:, 32:]), mean_content_task_grad_rms=rms(context.grad[:, :32]),
                contrast_content_task_grad_rms=rms(context.grad[:, 32:]))
    gradients = {n: rms(p.grad) if p.grad is not None else 0.0
                 for n, p in model.context.named_parameters()}
    if not all(math.isfinite(x) for x in gradients.values()):
        raise FloatingPointError("non-finite context task gradient")
    if model.module != "none":
        required = [name for name, p in model.context.named_parameters()
                    if p.ndim >= 2 and name.endswith("weight") and not name.startswith(("guide.", "boundary."))]
        inactive = [name for name in required if gradients[name] <= 0]
        if inactive:
            raise RuntimeError(f"segmentation task gradient absent: {inactive}")
        for stage in report.values():
            if not stage["finite"] or stage["task_context_grad_rms"] <= 0:
                raise RuntimeError("inactive pyramid stage")
    model.zero_grad(set_to_none=True)
    return dict(stages=report, task_grad_rms=gradients,
                projection_weight_rms={n: rms(p) for n, p in model.context.named_parameters() if p.ndim >= 2}, guide_grad_rms=rms(guide_gradient),
                edge_grad_rms=rms(edge_gradient), sample_main_loss=float(main.detach()),
                sample_guide_loss=float(guide.detach()), sample_boundary_loss=float(boundary.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["none", "hyperbolic", "qkv"], required=True)
    parser.add_argument("--dataset", choices=["LEVIR-CD-256", "SYSU-CD-256"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.1)
    parser.add_argument("--validation-every-epoch", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2333); np.random.seed(2333); torch.manual_seed(2333); torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = Path('/share_datasets/CD') / args.dataset
    train = make_loader(dataset, 'train', 32, 1)
    validation = make_loader(dataset, 'val', 2, 0)
    model = CASANet(str(ROOT / 'pretrained_weight/mobilenet_v2-b0353104.pth'), args.module).cuda()
    initial_weights = {n: rms(p) for n, p in model.context.named_parameters() if p.ndim >= 2}
    optimizer = make_optimizer(model, encoder_lr_mult=args.encoder_lr_mult)
    with torch.random.fork_rng(devices=[]):
        image, target = next(iter(validation))
    image, target = image.cuda().float(), target.cuda().float()
    total = math.ceil(40000 / len(train)) * len(train)
    manifest = dict(protocol=PROTOCOL, module=args.module, dataset=args.dataset,
                    epochs=args.epochs, batch_size=32, scheduler_total=total, guide_weight=0.2,
                    boundary_weight=0.05, encoder_lr_mult=args.encoder_lr_mult, evaluation_split="val",
                    no_decay_projections=[n for n, _ in model.named_parameters() if n in NO_DECAY_PROJECTIONS], initial_projection_weight_rms=initial_weights,
                    preprocessing=DATA_PROTOCOL, seed=2333,
                    grid_sizes=list(model.context.grid_sizes),
                    contexts_per_scale=list(model.context.tokens_per_scale),
                    joint_contexts=sum(model.context.tokens_per_scale),
                    qk_geometry=model.context.geometry, value_geometry=model.context.value_geometry)
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        losses, main_losses, guide_losses, edge_losses, norms = [], [], [], [], []
        for image_train, target_train in train:
            image_train, target_train = image_train.cuda().float(), target_train.cuda().float()
            update_learning_rate(optimizer, learning_rate(5e-4, epoch, step, total))
            optimizer.zero_grad(set_to_none=True)
            outputs, aux = model(image_train[:, :3], image_train[:, 3:], return_aux=True)
            loss, main_loss, guide, edge = training_losses(outputs, aux, target_train)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'probe loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            losses.append(float(loss.detach())); main_losses.append(float(main_loss.detach()))
            guide_losses.append(float(guide.detach())); edge_losses.append(float(edge.detach())); norms.append(float(norm))
            step += 1
        report = inspect(model, image, target)
        entry = dict(epoch=epoch+1, step=step, train_loss=sum(losses)/len(losses),
                     main_loss=sum(main_losses)/len(losses), guide_loss=sum(guide_losses)/len(losses),
                     boundary_loss=sum(edge_losses)/len(losses),
                     gradient_norm_mean=sum(norms)/len(norms), gradient_norm_max=max(norms), **report)
        if args.validation_every_epoch or epoch == args.epochs - 1:
            val_loss, scores, spatial = evaluate(model, make_loader(dataset, 'val', 32, 1),
                                                torch.device('cuda:0'), return_diagnostics=True)
            entry.update(validation_loss=val_loss, validation_metrics=scores, spatial=spatial)
        history.append(entry)
        (args.output_dir / 'health_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
    result = dict(passed=True, **manifest, steps=step, validation_loss=val_loss, validation_metrics=scores,
                  elapsed_seconds=time.time()-started, final=history[-1],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), manifest=manifest,
                    step=step), args.output_dir / 'probe_checkpoint.pth')
    (args.output_dir / 'health_result.json').write_text(json.dumps(result, indent=2))
    print(f'[PROBE-PASS] {args.module} {args.dataset} epochs={args.epochs} steps={step}', flush=True)


if __name__ == '__main__':
    main()
