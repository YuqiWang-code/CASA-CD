"""Occupancy supervision retains small changes and reaches every assignment head."""
import math

import torch
from torch.nn import functional as F

from model.casa_net import CASANet
from train_CASA import ASSIGNMENT_WEIGHT, assignment_loss, loss_function, training_objective


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    target = torch.zeros(1, 1, 256, 256)
    target[..., 1, 1] = 1
    for size in (64, 32, 16, 8):
        occupancy = F.adaptive_avg_pool2d(target, (size, size))
        assert occupancy.max() == (size / 256) ** 2
        assert occupancy.sum() * (256 / size) ** 2 == 1
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0, 0] = 1
    expected = math.log(2) + 1 - (0.25 + 1e-5) / (2.25 + 1e-5)
    assert abs(float(assignment_loss([logits], mask).detach()) - expected) < 1e-6
    assert torch.allclose(assignment_loss([logits, logits], mask), assignment_loss([logits], mask))
    assignment_loss([logits], mask).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    for value in (0.0, 1.0):
        assert torch.isfinite(assignment_loss([logits], torch.full_like(mask, value)))
    a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    for mode in ('hyperbolic', 'qkv'):
        model = CASANet(module=mode).eval()
        outputs, assignments = model(a, b, return_aux=True)
        assignment_loss(assignments, target).backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert all(p.grad is None for p in model.encoder_fusion.parameters())
        for op in model.stage_modules:
            assert all(p.grad is None for name, p in op.named_parameters() if not name.startswith('assignment.'))
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in op.assignment.parameters())
        model.zero_grad(set_to_none=True)
        outputs, assignments = model(a, b, return_aux=True)
        total, main_loss, auxiliary_loss = training_objective(outputs, assignments, target)
        assert torch.allclose(total, loss_function(outputs, target) + ASSIGNMENT_WEIGHT * auxiliary_loss)
        total.backward()
        for op in model.stage_modules:
            assert op.gamma == 0
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in op.assignment.parameters())
        print(mode, 'head-only auxiliary gradients and four supervised heads at zero gamma PASS')
    print('One-pixel occupancy, analytical BCE/Dice, scale averaging and empty/full targets PASS')


if __name__ == '__main__':
    main()
