"""Validation interventions for radial information and context correction value."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from model.casa_net import CASANet
from model.layers.lorentz_ops import lift
from train_CASA import make_loader, mask_boundary


@contextmanager
def intervention(context, mode):
    point = context.geometric_point
    distance = context.squared_distance
    gate = context.gate.forward
    gamma = context.gamma_context.detach().clone()
    if mode == 'context_off':
        with torch.no_grad():
            context.gamma_context.zero_()
    elif mode == 'fixed_radius':
        def equalize_radius(projected):
            space = point(projected)[..., 1:]
            norm = space.norm(dim=-1, keepdim=True)
            radial = torch.asinh(norm)
            positions = tuple(range(2, radial.ndim - 1))
            nonzero = (norm > 0).sum(positions, keepdim=True).clamp_min(1)
            mean_radius = radial.sum(positions, keepdim=True) / nonzero
            # Origins retain their zero vector; the full-position mean is preserved.
            return lift(F.normalize(space, dim=-1) * mean_radius.sinh())
        context.geometric_point = equalize_radius
    elif mode == 'fixed_direction':
        def radial_only(projected):
            direction = torch.zeros_like(projected)
            direction[..., 0] = projected.norm(dim=-1)
            return point(direction)
        context.geometric_point = radial_only
    elif mode == 'gate_open':
        context.gate.forward = lambda x: x.new_full((x.shape[0], context.heads, *x.shape[-2:]), 20.)
    elif mode == 'euclidean_metric':
        def euclidean(query, key):
            query, key = query[..., 1:], key[..., 1:]
            return (query.square().sum(-1, keepdim=True) + key.square().sum(-1).unsqueeze(-2)
                    - 2 * (query @ key.transpose(-1, -2))).clamp_min(0)
        context.squared_distance = euclidean
    try:
        yield
    finally:
        context.geometric_point = point
        context.squared_distance = distance
        context.gate.forward = gate
        with torch.no_grad():
            context.gamma_context.copy_(gamma)


def score(counts):
    tp, fp, fn, tn = counts
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                F1=2*tp/max(2*tp+fp+fn, 1), IoU=tp/max(tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1))


def small_foreground(target):
    masks = []
    for sample in target[:, 0].cpu().numpy().astype(np.uint8):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(sample, connectivity=8)
        selected = np.zeros_like(sample)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] <= 64:
                selected[labels == index] = 1
        masks.append(selected)
    return torch.as_tensor(np.stack(masks), device=target.device).bool()[:, None]


def analyze(model, loader, device):
    modes = ('baseline', 'context_off', 'fixed_radius', 'fixed_direction', 'gate_open', 'euclidean_metric')
    counts = {mode: [0, 0, 0, 0] for mode in modes}
    groups = {}
    captured_gate, captured_radius = [], []
    project_queries = model.context.project_queries
    def capture_queries(feature):
        query = project_queries(feature)
        captured_radius.append(torch.asinh(query[..., 1:].norm(dim=-1)).mean(1, keepdim=True))
        return query
    model.context.project_queries = capture_queries
    hooks = [model.context.gate.register_forward_hook(
        lambda module, inputs, output: captured_gate.append(output.sigmoid().mean(1, keepdim=True)))]
    diagnostics = None
    try:
        with torch.inference_mode():
            for image, target in loader:
                image, target = image.to(device).float(), target.to(device).float()
                predictions = {}
                for mode in modes:
                    captured_gate.clear()
                    captured_radius.clear()
                    model.context.collect_diag = mode == 'baseline' and diagnostics is None
                    with intervention(model.context, mode):
                        prediction = model.predict(image[:, :3], image[:, 3:]) > .5
                    if mode == 'baseline':
                        gates = [g.clone() for g in captured_gate]
                        radii = [radius.clone() for radius in captured_radius]
                        if diagnostics is None:
                            diagnostics = model.diagnostics()
                    truth = target > .5
                    batch_counts = [(prediction & truth).sum(), (prediction & ~truth).sum(),
                                    (~prediction & truth).sum(), (~prediction & ~truth).sum()]
                    counts[mode] = [a + int(b) for a, b in zip(counts[mode], batch_counts)]
                    predictions[mode] = prediction
                base, off = predictions['baseline'], predictions['context_off']
                masks = dict(true_change=truth, unchanged=~truth,
                             missed_without_context=truth & ~off,
                             false_positive_without_context=~truth & off,
                             corrected=(off != truth) & (base == truth),
                             harmed=(off == truth) & (base != truth),
                             boundary_1px=mask_boundary(target, 1).bool(),
                             small_foreground_le64=small_foreground(truth))
                for name, mask in masks.items():
                    group = groups.setdefault(name, dict(pixels=0, gate_sum=[0.]*4, radius_sum=[0.]*4))
                    group['pixels'] += int(mask.sum())
                    for index, (gate, radius) in enumerate(zip(gates, radii)):
                        gate = F.interpolate(gate, target.shape[-2:], mode='bilinear', align_corners=False)
                        radius = F.interpolate(radius, target.shape[-2:], mode='bilinear', align_corners=False)
                        group['gate_sum'][index] += float((gate * mask).sum())
                        group['radius_sum'][index] += float((radius * mask).sum())
    finally:
        model.context.project_queries = project_queries
        for hook in hooks:
            hook.remove()
        model.context.collect_diag = False
    for group in groups.values():
        count = group['pixels']
        group['mean_gate_by_stage'] = [x/count if count else None for x in group.pop('gate_sum')]
        group['mean_radius_by_stage'] = [x/count if count else None for x in group.pop('radius_sum')]
    return dict(interventions={mode: score(value) for mode, value in counts.items()},
                pixel_groups=groups, first_batch_diagnostics=diagnostics,
                radius_scope='Actual dense Lorentz queries: asinh(norm(Q_spatial)), averaged over heads at each stage; bilinear resize for pixel groups',
                intervention_definitions=dict(
                    fixed_radius='For each sample/head, set nonorigin positions to their mean intrinsic radius; origins stay at zero and the all-position mean is preserved. Each query stage and the joint key bank are equalized separately.',
                    fixed_direction='Preserve every projected norm and intrinsic radius; place all spatial directions on the first coordinate axis.'),
                interpretation='Frozen-weight sensitivity and pixel associations; not retrained ablations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CASANet(module='hyperbolic')
    model.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
    model = model.to(device).eval()
    source = make_loader(args.dataset_root, args.split, args.batch_size, 0).dataset
    # Training samples are inspected without random augmentation.
    if args.split == 'train':
        source.transform = make_loader(args.dataset_root, 'val', args.batch_size, 0).dataset.transform
    count = min(args.samples, len(source))
    indices = np.linspace(0, len(source)-1, count, dtype=int).tolist()
    loader = DataLoader(Subset(source, indices), batch_size=args.batch_size, shuffle=False)
    report = analyze(model, loader, device)
    report.update(dataset=str(args.dataset_root), split=args.split, sample_indices=indices,
                  threshold=.5, checkpoint=str(args.checkpoint), small_object_max_area=64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['interventions']))


if __name__ == '__main__':
    main()
