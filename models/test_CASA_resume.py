"""Optimizer/RNG continuation and deployment backend-state checks."""
import copy
import random

import numpy as np
import torch
from torch import nn

from train_CASA import (PROTOCOL, make_optimizer, restore_training_state,
                        rng_state, strict_fold_precision, update_learning_rate,
                        learning_rate, context_decay_names, NO_DECAY_PROJECTIONS)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = "qkv"
        self.encoder = nn.Linear(3, 3)
        self.body = nn.Linear(3, 3)
        self.context = nn.ModuleDict({"qk": nn.Linear(3, 3, bias=False),
                                      "k_contrast": nn.Linear(3, 3, bias=False),
                                      "v_contrast": nn.Linear(3, 3, bias=False),
                                      "v": nn.Linear(3, 3, bias=False),
                                      "out": nn.Linear(3, 3, bias=False),
                                      "guide": nn.Linear(3, 3)})

    def forward(self, x):
        x = self.encoder(x)
        return self.context["out"](self.context["qk"](self.body(x)) + self.context["v"](x)
                                   + self.context["k_contrast"](x) + self.context["v_contrast"](x)) + self.context["guide"](x)


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
    decay = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
    scales = {id(p): group["lr_scale"] for group in optimizer.param_groups for p in group["params"]}
    decay_names = context_decay_names(model)
    for name, parameter in model.named_parameters():
        expected_decay = 0.0 if name.startswith("context.") and name not in decay_names else 1e-4
        assert decay[id(parameter)] == expected_decay
        assert scales[id(parameter)] == (0.1 if name.startswith("encoder.") else 1.0)
        if name in NO_DECAY_PROJECTIONS:
            assert decay[id(parameter)] == 0.0
        elif name.endswith('.weight'):
            assert decay[id(parameter)] == 1e-4
    for count in (0, 185, 199, 200, 20000, 39999):
        rate = learning_rate(5e-4, 1, count, 40000)
        update_learning_rate(optimizer, rate)
        assert optimizer.param_groups[0]["lr"] == rate * 0.1
        assert all(group["lr"] == rate for group in optimizer.param_groups[1:])
    for _ in range(4):
        step(model, optimizer)
    checkpoint = copy.deepcopy(dict(
        epoch=2, step=4, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        manifest=dict(protocol=PROTOCOL, module="qkv", actual_steps=6, encoder_lr_mult=0.1), rng_state=rng_state()))
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
    print("Encoder LR ratio, context decay, optimizer/RNG continuation and backend restoration PASS")


if __name__ == "__main__":
    main()
