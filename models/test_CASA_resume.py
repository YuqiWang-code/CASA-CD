"""Optimizer/RNG continuation and deployment backend-state checks."""
import copy
import random

import numpy as np
import torch
from torch import nn

from train_CASA import (PROTOCOL, make_optimizer, restore_training_state,
                        rng_state, strict_fold_precision)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = "qkv"
        self.body = nn.Linear(3, 3)
        self.stage_modules = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, x):
        return self.stage_modules[0](self.body(x))


def step(model, optimizer):
    x = torch.randn(4, 3) + random.random() + float(np.random.random())
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()


def backend_state():
    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())


def main():
    torch.manual_seed(2333)
    np.random.seed(2333)
    random.seed(2333)
    model = SmallModel()
    optimizer = make_optimizer(model)
    for _ in range(4):
        step(model, optimizer)
    checkpoint = copy.deepcopy(dict(
        epoch=2, step=4, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        manifest=dict(protocol=PROTOCOL, module="qkv", actual_steps=6), rng_state=rng_state()))
    for _ in range(2):
        step(model, optimizer)
    restored = SmallModel()
    restored_optimizer = make_optimizer(restored)
    epoch, count, exact_rng = restore_training_state(checkpoint, restored, restored_optimizer, 6, 2)
    assert (epoch, count, exact_rng) == (2, 4, True)
    for _ in range(2):
        step(restored, restored_optimizer)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    assert all(int(state["step"]) == 6 for state in restored_optimizer.state.values())
    checkpoint.pop("rng_state")
    epoch, count, exact_rng = restore_training_state(checkpoint, restored, restored_optimizer, 6, 2)
    assert (epoch, count, exact_rng) == (2, 4, False)
    assert all(torch.equal(restored.state_dict()[key], value)
               for key, value in checkpoint["state_dict"].items())
    before = backend_state()
    try:
        with strict_fold_precision():
            assert backend_state() == (False, False, False, "highest")
            raise ValueError("exercise context exit")
    except ValueError:
        pass
    assert backend_state() == before
    print("Optimizer/RNG continuation, legacy checkpoint reseeding and backend restoration PASS")


if __name__ == "__main__":
    main()
