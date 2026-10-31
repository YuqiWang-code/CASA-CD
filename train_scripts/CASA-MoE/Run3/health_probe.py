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
    model.context.capture = True
    model.zero_grad(set_to_none=True)
    outputs, auxiliary = model(image[:, :3], image[:, 3:], return_aux=True)
    _, main, guide = training_losses(outputs, auxiliary, target)
    captured = model.context.captured
    branch = {}
    if model.module != 'none':
        for scale, record in captured.items():
            rh, re, rw = record['residuals']
            derivatives = torch.autograd.grad(main, (rh,re,rw), retain_graph=True)
            gradients = [torch.autograd.grad(value, model.context.retrieval.qk.weight,
                         grad_outputs=derivative, retain_graph=True)[0]
                         for value, derivative in zip((rh,re), derivatives[:2])]
            branch[str(scale)] = dict(output_grad_rms=[rms(d) for d in derivatives],
                qk_he_gradient_rms=[rms(g) for g in gradients],
                qk_he_gradient_cosine=float(F.cosine_similarity(gradients[0].flatten(),gradients[1].flatten(),dim=0)))
            if any(rms(d) <= 0 for d in derivatives):
                raise RuntimeError('absent expert task gradient')
    main.backward()
    gradients = {name:rms(p.grad) if p.grad is not None else 0.0
                 for name,p in model.context.named_parameters()}
    if not all(math.isfinite(v) for v in gradients.values()):
        raise FloatingPointError('non-finite probe gradients')
    if model.module != 'none':
        required=['retrieval.qk.weight','retrieval.v.weight','wavelet.high.0.0.weight',
                  'wavelet.low.0.0.weight']
        if model.module == 'moe': required += ['router.0.4.weight']
        if any(gradients[name] <= 0 for name in required):
            raise RuntimeError('missing learned expert gradient')
    tar_input_gradients = []
    if model.module == 'moe':
        tar_input_gradients = [rms(layer[0].weight.grad[:, 64:128])
                               for layer in model.context.router]
        if any(value <= 0 for value in tar_input_gradients):
            raise RuntimeError('router TAR input has no task gradient')
    report=model.diagnostics()
    if any(not stage['finite'] for stage in report.values()):
        raise FloatingPointError('non-finite expert output')
    model.context.collect_diag=False
    model.context.capture=False
    model.context.captured={}
    model.zero_grad(set_to_none=True)
    return dict(stages=report, task_grad_rms=gradients, branch_task_gradients=branch,
                router_tar_input_grad_rms=tar_input_gradients,
                sample_main_loss=float(main.detach()), sample_guide_loss=float(guide.detach()))


@torch.no_grad()
def spatial_probe(model, image, target):
    model.eval()
    model.context.capture=True
    baseline=model.predict(image[:,:3],image[:,3:])
    groups={}
    for scale, record in model.context.captured.items():
        route=record['routes']
        truth=F.adaptive_avg_pool2d(target,route.shape[-2:])
        groups[str(scale)]={label:((route*mask).sum((0,2,3))/mask.sum().clamp_min(1e-8)).tolist()
                           for label,mask in [('change',truth),('background',1-truth)]}
    model.context.capture=False
    model.context.captured={}
    same=model.predict(image[:,:3],image[:,:3])
    swapped=model.predict(image[:,3:],image[:,:3])
    return dict(split='val', images=len(image), route_names=list(model.context.route_names),
                gt_route_groups=groups, same_pair_positive_fraction=float((same>.5).float().mean()),
                exchange_mean_absolute_diff=float((baseline-swapped).abs().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS[:2], required=True)
    parser.add_argument('--module', choices=['none','static','moe'], required=True)
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
    model = CASANet(str(PRETRAINED), args.module).cuda()
    optimizer = make_optimizer(model)
    with torch.random.fork_rng(devices=[]):
        sample_image, sample_target = next(iter(make_loader(dataset, 'val', 2, 0)))
    sample_image, sample_target = sample_image.cuda().float(), sample_target.cuda().float()
    total = math.ceil(40000/len(train))*len(train)
    manifest = dict(protocol=PROTOCOL, runtime=str(RUNTIME), module=args.module, dataset=args.dataset,
                    epochs=3, train_batches=len(train), batch_size=32, physical_gpu=0,
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
