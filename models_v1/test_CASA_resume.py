"""Optimizer/RNG continuation and deployment backend-state checks."""
import copy
import random

import numpy as np
import torch
from torch import nn

from train_CASA import (PROTOCOL, make_optimizer, restore_training_state,
                        rng_state, strict_fold_precision, update_learning_rate,
                        learning_rate, context_decay_names)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = "qkv"
        self.encoder = nn.Linear(3, 3)
        self.body = nn.Linear(3, 3)
        self.context = nn.ModuleDict({"qk": nn.Linear(3, 3, bias=False),
                                      "norm": nn.LayerNorm(3),
                                      "v": nn.Linear(3, 3, bias=False),
                                      "out": nn.Linear(3, 3, bias=False),
                                      "guide": nn.Linear(3, 3)})

    def forward(self, x):
        x = self.context["norm"](self.encoder(x))
        return self.context["out"](self.context["qk"](self.body(x)) + self.context["v"](x)) + self.context["guide"](x)


def step(model, optimizer):
    x = torch.randn(4, 3) + random.random() + float(np.random.random())
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()


def backend_state():
    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())


def check_decay_step():
    model = SmallModel().double()
    optimizer = make_optimizer(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    lr, decay, eps = 5e-4, 1e-4, 1e-8
    expected_context = 1 - lr * decay
    expected_body = 1 - lr * decay / (decay + eps)
    torch.testing.assert_close(model.context["qk"].weight,
        torch.full_like(model.context["qk"].weight, expected_context), rtol=0, atol=1e-12)
    torch.testing.assert_close(model.body.weight,
        torch.full_like(model.body.weight, expected_body), rtol=0, atol=1e-12)
    torch.testing.assert_close(model.encoder.weight,
        torch.full_like(model.encoder.weight, expected_body), rtol=0, atol=1e-12)
    assert torch.equal(model.context["norm"].weight, torch.ones_like(model.context["norm"].weight))
    assert torch.equal(model.context["guide"].bias, torch.ones_like(model.context["guide"].bias))


def main():
    check_decay_step()
    torch.manual_seed(2333)
    np.random.seed(2333)
    random.seed(2333)
    model = SmallModel()
    optimizer = make_optimizer(model)
    decay = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
    scales = {id(p): group["lr_scale"] for group in optimizer.param_groups for p in group["params"]}
    decay_names = context_decay_names(model)
    decoupled = {id(p): group["decoupled_weight_decay"]
                 for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        expected_decay = 0.0 if name.startswith("context.") and name not in decay_names else 1e-4
        assert decay[id(parameter)] == expected_decay
        assert scales[id(parameter)] == 1.0
        assert decoupled[id(parameter)] == (name in decay_names)
        if name.startswith("context.norm.") or name == "context.guide.bias":
            assert decay[id(parameter)] == 0.0
        elif name.endswith(".weight"):
            assert decay[id(parameter)] == 1e-4
    for count in (0, 185, 199, 200, 20000, 39999):
        rate = learning_rate(5e-4, 1, count, 40000)
        update_learning_rate(optimizer, rate)
        assert optimizer.param_groups[0]["lr"] == rate
        assert all(group["lr"] == rate for group in optimizer.param_groups[1:])
    for _ in range(4):
        step(model, optimizer)
    checkpoint = copy.deepcopy(dict(
        epoch=2, step=4, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        manifest=dict(protocol=PROTOCOL, module="qkv", actual_steps=6, encoder_lr_mult=1.0), rng_state=rng_state()))
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
    print("Encoder LR, decoupled context/coupled body decay, optimizer/RNG continuation and backend restoration PASS")


if __name__ == "__main__":
    main()
