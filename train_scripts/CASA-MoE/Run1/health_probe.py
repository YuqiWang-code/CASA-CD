"""Three complete train epochs and validation checks before the exploration queue."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from config import DATASETS, PRETRAINED

RUNTIME = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUNTIME / 'models_v1'))
from train_CASA import (PROTOCOL, DATA_PROTOCOL, GUIDE_WEIGHT, learning_rate,
                        update_learning_rate, training_losses, make_loader,
                        make_optimizer, evaluate)
from model.casa_net import CASANet


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def inspect(model, image, target):
    model.eval()
    model.context.collect_diag = True
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            delta = output[0] if isinstance(output, tuple) else output
            delta.retain_grad()
            captured.setdefault(name, []).append(delta)
        return hook

    hooks = [expert.register_forward_hook(capture(name))
             for name, expert in model.context.experts.items()]
    model.zero_grad(set_to_none=True)
    outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
    model.context.collect_diag = False
    _, main, guide = training_losses(outputs, auxiliary, target)
    main.backward()
    for hook in hooks:
        hook.remove()
    gradients = {name: rms(parameter.grad) if parameter.grad is not None else 0.0
                 for name, parameter in model.context.named_parameters()}
    required = [name for name, parameter in model.context.named_parameters()
                if parameter.ndim >= 2 and name.endswith('weight') and not name.startswith('guide.')]
    missing = [name for name in required if gradients[name] <= 0]
    if missing or not all(math.isfinite(value) for value in gradients.values()):
        raise RuntimeError(f'absent or non-finite segmentation-task gradients: {missing}')
    expert_gradients = {}
    for name, stages in captured.items():
        expert_gradients[name] = [dict(output_rms=rms(delta),
                                     task_grad_rms=rms(delta.grad) if delta.grad is not None else 0.0)
                                 for delta in stages]
        if not stages or any(item['task_grad_rms'] <= 0 for item in expert_gradients[name]):
            raise RuntimeError(f'expert has no segmentation gradient: {name}')
    report = model.diagnostics()
    if any(not stage['finite'] for stage in report.values()):
        raise FloatingPointError('non-finite expert diagnostic')
    model.zero_grad(set_to_none=True)
    return dict(stages=report, task_grad_rms=gradients, expert_task_gradients=expert_gradients,
                sample_main_loss=float(main.detach()), sample_guide_loss=float(guide.detach()))


@torch.no_grad()
def spatial_probe(model, image, target):
    """Fixed validation images: true-label route groups and registration sensitivity."""
    model.eval()
    routes = []
    hook = model.context.router.register_forward_hook(
        lambda module, inputs, output: routes.append(output.softmax(1)))
    baseline = model.predict(image[:, :3], image[:, 3:])
    hook.remove()
    groups = {}
    for index, route in enumerate(routes):
        foreground = F.interpolate(target, size=route.shape[-2:], mode='area')
        background = 1-foreground
        groups[f'stage{index+1}'] = {
            'true_change': dict(zip(model.context.route_names,
                ((route*foreground).sum((0, 2, 3))/foreground.sum().clamp_min(1e-8)).tolist())),
            'true_background': dict(zip(model.context.route_names,
                ((route*background).sum((0, 2, 3))/background.sum().clamp_min(1e-8)).tolist())),
        }
    truth = target >= .5
    dilation = F.max_pool2d(target, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-target, 3, stride=1, padding=1)
    boundary = (dilation-erosion) > 0
    valid = torch.ones_like(truth)
    valid[..., :2, :] = valid[..., -2:, :] = False
    valid[..., :, :2] = valid[..., :, -2:] = False

    def errors(probability):
        prediction = probability >= .5
        tp = int((prediction & truth & valid).sum())
        fp = int((prediction & ~truth & valid).sum())
        fn = int((~prediction & truth & valid).sum())
        return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(1, 2*tp+fp+fn),
                    boundary_errors=int(((prediction != truth) & boundary & valid).sum()),
                    boundary_pixels=int((boundary & valid).sum()))

    shifts = {'0': errors(baseline)}
    for amount in (1, 2):
        shifted_b = F.pad(image[:, 3:, :, :-amount], (amount, 0, 0, 0), mode='replicate')
        probability = model.predict(image[:, :3], shifted_b)
        shifts[str(amount)] = dict(**errors(probability),
            prediction_mean_absolute_change=float((probability-baseline).abs()[valid].mean()))
    return dict(images=len(image), split='val', route_grouping='area_resized_ground_truth',
                route_groups=groups, b_right_shift_pixels=shifts, ignored_border_pixels=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS[:2], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2333)
    np.random.seed(2333)
    torch.manual_seed(2333)
    torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = Path('/share_datasets/CD') / args.dataset
    train = make_loader(dataset, 'train', 32, 1)
    validation = make_loader(dataset, 'val', 32, 1)
    model = CASANet(str(PRETRAINED), 'moe').cuda()
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        sample_image, sample_target = next(iter(make_loader(dataset, 'val', 2, 0)))
    sample_image, sample_target = sample_image.cuda().float(), sample_target.cuda().float()
    total = math.ceil(40000/len(train))*len(train)
    manifest = dict(protocol=PROTOCOL, runtime=str(RUNTIME), module='moe', dataset=args.dataset,
                    epochs=3, train_batches=len(train), batch_size=32, physical_gpu=1,
                    scheduler_total=total, guide_weight=GUIDE_WEIGHT, encoder_lr_mult=1.0,
                    evaluation_split='val', seed=2333, preprocessing=DATA_PROTOCOL,
                    architecture=model.context_metadata(),
                    optimizer_groups=[dict(name=group['group_name'], weight_decay=group['weight_decay'],
                                           decoupled_weight_decay=group['decoupled_weight_decay'])
                                      for group in optimizer.param_groups])
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    step, history, started = 0, [], time.time()
    for epoch in range(3):
        model.train()
        losses, main_losses, guide_losses, norms = [], [], [], []
        for image, target in train:
            image, target = image.cuda().float(), target.cuda().float()
            update_learning_rate(optimizer, learning_rate(5e-4, epoch, step, total))
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
            loss, main, guide = training_losses(outputs, auxiliary, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'probe loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'),
                                                 error_if_nonfinite=True, foreach=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            main_losses.append(float(main.detach()))
            guide_losses.append(float(guide.detach()))
            norms.append(float(norm))
            step += 1
        report = inspect(model, sample_image, sample_target)
        val_loss, scores, spatial = evaluate(model, validation, torch.device('cuda:0'), return_diagnostics=True)
        entry = dict(epoch=epoch+1, step=step, train_loss=sum(losses)/len(losses),
                     main_loss=sum(main_losses)/len(main_losses), guide_loss=sum(guide_losses)/len(guide_losses),
                     gradient_norm_mean=sum(norms)/len(norms), gradient_norm_max=max(norms),
                     validation_loss=val_loss, validation_metrics=scores, spatial=spatial, **report)
        history.append(entry)
        (args.output_dir / 'health_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True)
    if (not math.isfinite(val_loss) or scores['F1'] <= .1
            or history[-1]['main_loss'] > history[0]['main_loss']*1.02):
        raise RuntimeError('probe has no useful segmentation or its main training loss did not improve')
    spatial_sensitivity = spatial_probe(model, sample_image, sample_target)
    result = dict(passed=True, **manifest, steps=step, final=history[-1],
                  fixed_validation_spatial_probe=spatial_sensitivity,
                  elapsed_seconds=time.time()-started,
                  main_loss_first_to_last=[history[0]['main_loss'], history[-1]['main_loss']],
                  validation_f1_history=[item['validation_metrics']['F1'] for item in history],
                  validation_loss_history=[item['validation_loss'] for item in history],
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), manifest=manifest, step=step),
               args.output_dir / 'probe_checkpoint.pth')
    (args.output_dir / 'health_result.json').write_text(json.dumps(result, indent=2))
    print(f'[PROBE-PASS] moe {args.dataset} epochs=3 steps={step}', flush=True)


if __name__ == '__main__':
    main()
