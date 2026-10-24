"""Observe real-data losses and segmentation gradients before formal training."""
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
from train_CASA import (PROTOCOL, learning_rate, loss_function, guide_loss,
                        training_losses, make_loader, make_optimizer, evaluate)
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

    hook = model.context.out.register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    outputs, logits = model(image[:, :3], image[:, 3:], return_aux=True)
    model.context.collect_diag = False
    main = loss_function(outputs, target)
    auxiliary = guide_loss(logits, target)
    guide_gradient = torch.autograd.grad(auxiliary, model.context.guide.weight, retain_graph=True)[0]
    main.backward()
    hook.remove()
    report = model.diagnostics()
    for index, (context, update) in enumerate(captured):
        report[f"stage{index+1}"].update(
            branch_rms=rms(update), residual_rms=rms(model.context.gamma[index] * update),
            context_rms=rms(context), task_context_grad_rms=rms(context.grad))
    gradients = {n: rms(p.grad) if p.grad is not None else 0.0
                 for n, p in model.context.named_parameters()}
    if not all(math.isfinite(x) for x in gradients.values()):
        raise FloatingPointError("non-finite context task gradient")
    for name in ("q.weight", "k.weight", "v.weight", "out.weight"):
        if gradients[name] <= 0:
            raise RuntimeError(f"segmentation task does not reach {name}")
    for stage in report.values():
        if not stage["finite"] or stage["residual_rms"] <= 0 or stage["task_context_grad_rms"] <= 0:
            raise RuntimeError("inactive pyramid stage")
    model.zero_grad(set_to_none=True)
    return dict(stages=report, task_grad_rms=gradients, guide_grad_rms=rms(guide_gradient),
                sample_main_loss=float(main.detach()), sample_guide_loss=float(auxiliary.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["hyperbolic", "qkv"], required=True)
    parser.add_argument("--dataset", choices=["LEVIR-CD-256", "SYSU-CD-256"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    dataset = Path('/share_datasets/CD') / args.dataset
    train = make_loader(dataset, 'train', 32, 1)
    validation = make_loader(dataset, 'val', 2, 0)
    model = CASANet(str(ROOT / 'pretrained_weight/mobilenet_v2-b0353104.pth'), args.module).cuda()
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        image, target = next(iter(validation))
    image, target = image.cuda().float(), target.cuda().float()
    total = math.ceil(40000 / len(train)) * len(train)
    manifest = dict(protocol=PROTOCOL, module=args.module, dataset=args.dataset,
                    epochs=args.epochs, batch_size=32, scheduler_total=total, guide_weight=0.2)
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        losses, main_losses, guide_losses, norms = [], [], [], []
        for image_train, target_train in train:
            image_train, target_train = image_train.cuda().float(), target_train.cuda().float()
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(5e-4, epoch, step, total)
            optimizer.zero_grad(set_to_none=True)
            outputs, logits = model(image_train[:, :3], image_train[:, 3:], return_aux=True)
            loss, main_loss, auxiliary = training_losses(outputs, logits, target_train)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'probe loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            main_losses.append(float(main_loss.detach()))
            guide_losses.append(float(auxiliary.detach()))
            norms.append(float(norm))
            step += 1
        report = inspect(model, image, target)
        entry = dict(epoch=epoch+1, step=step, train_loss=sum(losses)/len(losses),
                     main_loss=sum(main_losses)/len(losses), guide_loss=sum(guide_losses)/len(losses),
                     gradient_norm_mean=sum(norms)/len(norms), gradient_norm_max=max(norms), **report)
        history.append(entry)
        (args.output_dir / 'health_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
    val_loss, score = evaluate(model, make_loader(dataset, 'val', 32, 1), torch.device('cuda:0'))
    result = dict(passed=True, **manifest, steps=step, validation_loss=val_loss, validation_metrics=score,
                  elapsed_seconds=time.time()-started, final=history[-1],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), manifest=manifest,
                    step=step), args.output_dir / 'probe_checkpoint.pth')
    (args.output_dir / 'health_result.json').write_text(json.dumps(result, indent=2))
    print(f'[PROBE-PASS] {args.module} {args.dataset} epochs={args.epochs} steps={step}', flush=True)


if __name__ == '__main__':
    main()
