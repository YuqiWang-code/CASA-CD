"""Two full train/val epochs, task gradients and deployment for one paired arm."""
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
from config import ROOT, RUNTIME, PROTOCOL, PRETRAINED, VARIANTS

sys.path.insert(0, str(RUNTIME / 'models'))
from model.casa_net import CASANet
from train_CASA import (make_loader, make_optimizer, training_losses, learning_rate,
                        update_learning_rate, evaluate, verify_deployment)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('dataset')
    args = p.parse_args()
    spec = VARIANTS[args.variant]
    out = ROOT / 'probes' / args.variant / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'health_result.json').exists():
        raise RuntimeError('probe already finished')
    torch.set_num_threads(4)
    random.seed(2333); np.random.seed(2333); torch.manual_seed(2333); torch.cuda.manual_seed_all(2333)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    loaders = [make_loader(Path('/share_datasets/CD') / args.dataset, split, 32, 1) for split in ('train','val')]
    for i, loader in enumerate(loaders):
        loader.generator = torch.Generator().manual_seed(2333+i)
        if i == 0: loader.sampler.generator = loader.generator
    train, val = loaders
    model = CASANet(PRETRAINED, module=spec['module'], context_seed=2333).cuda()
    optimizer = make_optimizer(model)
    total_steps = math.ceil(40000 / len(train)) * len(train)
    history, step = [], 0
    manifest = dict(protocol=PROTOCOL, variant=args.variant, **spec, epochs=2, formal_result=False,
                    seed=2333, context_seed=2333, loader_seeds=[2333,2334], batch_size=32,
                    train_samples=len(train.dataset), val_samples=len(val.dataset), scheduler_total_steps=total_steps,
                    guide_weight=.1, retrieval_weight=0., initialization='fresh MobileNetV2 pretrained',
                    preprocessing='same frozen runtime as formal queue', physical_gpu=1)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    for epoch in range(2):
        tick = time.time(); model.train(); model.context.collect_diag = False
        sums = np.zeros(3); norm_max = 0.
        for image, target in train:
            image, target = image.cuda().float(), target.cuda().float()
            update_learning_rate(optimizer, learning_rate(5e-4, epoch, step, total_steps))
            optimizer.zero_grad(set_to_none=True)
            outputs, auxiliary = model(image[:,:3], image[:,3:], return_aux=True)
            losses = training_losses(outputs, auxiliary, target, .1, 0., spec['interior_positive_weight'])
            assert all(torch.isfinite(x) for x in losses)
            losses[0].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            norm_max = max(norm_max, float(norm)); optimizer.step(); step += 1
            sums += [float(x.detach()) for x in losses]
        model.context.collect_diag = True
        loss, metrics, spatial = evaluate(model, val, torch.device('cuda'), return_diagnostics=True)
        item = dict(epoch=epoch, step=step, train_total=sums[0]/len(train), train_main=sums[1]/len(train),
                    train_guide=sums[2]/len(train), val_loss=loss, val_metrics=metrics,
                    spatial=spatial, gradient_norm_max=norm_max, diagnostics=copy.deepcopy(model.diagnostics()),
                    epoch_seconds=time.time()-tick)
        history.append(item)
        (out / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print('[PROBE-EPOCH] '+json.dumps(item), flush=True)
    model.eval(); model.context.collect_diag = False; optimizer.zero_grad(set_to_none=True)
    outputs, aux = model(image[:,:3], image[:,3:], return_aux=True)
    main_loss = training_losses(outputs, aux, target, .1, 0., spec['interior_positive_weight'])[1]
    main_loss.backward()
    gradients = {n:float(p.grad.square().mean().sqrt()) for n,p in model.named_parameters()
                 if n.startswith('context.') and p.grad is not None}
    assert all(math.isfinite(v) for v in gradients.values())
    if spec['module'] != 'none':
        required = ['qk','v','out','gate','guide','local_dw','local_out']
        assert all(gradients.get('context.'+n+'.weight',0)>0 for n in required), gradients
        for stage in history[-1]['diagnostics'].values():
            assert stage['finite'] and stage['context_residual_to_feature_rms']>0
    _, fold = verify_deployment(model, image[:1])
    report = dict(protocol=PROTOCOL, passed=True, epochs=2, step=step, **spec,
                  task_gradients=gradients, fold=fold,
                  initial_train_main=history[0]['train_main'], final_train_main=history[-1]['train_main'])
    (out / 'health_result.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
