"""Two full train/val epochs for a Run13 ablation arm."""
import argparse
import json
import math
import random
import sys
import time

import numpy as np
import torch

from config import DATA, PRETRAINED, PROTOCOL, ROOT, RUNTIME, VARIANTS, seed_for

sys.path.insert(0, str(RUNTIME / 'models'))
from model.casa_net import CASANet
from train_CASA import evaluate, learning_rate, make_loader, make_optimizer, training_losses, update_learning_rate, verify_deployment

ABLATED = {'plain': ['tar.stage1.temporal.proj.weight', 'tar.stage1.block.dw.weight',
                     'tar.stage1.block.pw.weight'],
           'reparam': []}
DECODER = {'plain': ['decoder.fuse1.fuse.weight', 'decoder.refine.dw.weight',
                     'decoder.refine.pw.weight'],
           'reparam': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=VARIANTS)
    parser.add_argument('dataset')
    args = parser.parse_args()
    module, heads, detail, tar, decoder = VARIANTS[args.variant]
    seed = seed_for(args.variant)
    out = ROOT / 'probes' / args.variant / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    train, val = [make_loader(DATA / args.dataset, split, 32, 1) for split in ('train', 'val')]
    train.generator = train.sampler.generator = torch.Generator().manual_seed(seed)
    val.generator = torch.Generator().manual_seed(seed+1)
    device = torch.device('cuda:0')
    model = CASANet(PRETRAINED, module, seed, heads, detail, 'area', tar, decoder).to(device)
    optimizer = make_optimizer(model)
    budget = math.ceil(40000 / len(train)) * len(train)
    history, step = [], 0
    for epoch in range(2):
        tick = time.monotonic()
        model.train()
        model.context.collect_diag = False
        sums = np.zeros(3)
        for image, target in train:
            image, target = image.to(device).float(), target.to(device).float()
            update_learning_rate(optimizer, learning_rate(5e-4, epoch, step, budget))
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(image[:, :3], image[:, 3:], return_aux=True)
            losses = training_losses(logits, aux, target)
            if not torch.isfinite(losses[0]):
                raise FloatingPointError(f'non-finite loss at probe step {step}')
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            step += 1
            sums += [float(x.detach()) for x in losses]
        model.context.collect_diag = True
        loss, metrics, spatial = evaluate(model, val, device, return_diagnostics=True)
        item = dict(epoch=epoch, step=step, train_total=sums[0]/len(train), train_main=sums[1]/len(train),
                    train_guide=sums[2]/len(train), val_loss=loss, val_metrics=metrics, spatial=spatial,
                    diagnostics=model.diagnostics(), seconds=time.monotonic()-tick)
        history.append(item)
        (out/'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print('[PROBE-EPOCH]', epoch, 'train', item['train_main'], 'val', loss, 'F1', metrics['F1'], flush=True)
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    logits, aux = model(image[:, :3], image[:, 3:], return_aux=True)
    training_losses(logits, aux, target)[1].backward()
    required = ABLATED[tar] + DECODER[decoder] + ['head.weight']
    gradients = {name: float(param.grad.detach().norm()) for name, param in model.named_parameters()
                 if name in required}
    if set(gradients) != set(required):
        raise RuntimeError(f'missing ablation parameters: {sorted(set(required) - set(gradients))}')
    if any(not math.isfinite(gradients[n]) or gradients[n] == 0 for n in required):
        raise RuntimeError(f'inactive probe gradients: {gradients}')
    _, fold = verify_deployment(model, image[:1])
    (out/'health_result.json').write_text(json.dumps(dict(protocol=PROTOCOL, passed=True,
        variant=args.variant, dataset=args.dataset, seed=seed, module=module, heads=heads,
        detail_head=detail, tar_variant=tar, decoder_variant=decoder, epochs=2, step=step,
        gradients=gradients, fold=fold), indent=2))


if __name__ == '__main__':
    main()
