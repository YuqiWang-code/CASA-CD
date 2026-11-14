"""Static and model-level audit for Run15 (no training, no GPU)."""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import torch

import config as C


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == 20, len(jobs)
    assert len(set(jobs)) == len(jobs)
    for version in C.VERSIONS:
        assert C.arms(version) == C.VERSIONS[version]["arms"]
        for arm in C.arms(version):
            assert C.reservation_key(version, arm) in C.TRAINABLE_PARAMS
            assert C.train_script(version).name == "train_CASA.py"
    assert sum(len(C.arms(v)) for v in C.VERSIONS) * len(C.DATASETS) == 20
    for version in C.VERSIONS:
        assert C.VERSIONS[version]["selection"] == "val"


def check_cli():
    for version in ("Run4", "Run10"):
        text = subprocess.check_output([sys.executable, str(C.train_script(version)), "--help"],
                                       text=True, stderr=subprocess.STDOUT)
        assert "--selection-split {test,val}" in text, version
    text = subprocess.check_output([sys.executable, str(C.train_script("Run10")), "--help"],
                                   text=True, stderr=subprocess.STDOUT)
    assert "--seed" in text and "--context-seed" in text
    text = subprocess.check_output([sys.executable, str(C.train_script("Run1")), "--help"],
                                   text=True, stderr=subprocess.STDOUT)
    assert "--module {none,hyperbolic,qkv}" in text


def load_net(version, arm):
    directory = str(C.RUNTIME / C.model_dir(version))
    for name in [n for n in list(sys.modules) if n == "model" or n.startswith("model.")]:
        del sys.modules[name]
    sys.path.insert(0, directory)
    module = importlib.import_module("model.casa_net")
    return module.CASANet(module=arm), directory


def open_gate(model):
    """Zero-init residual gates hide the context gradient at step 0; open them for the audit."""
    context = getattr(model, "context", None)
    gamma = getattr(context, "gamma_context", None)
    if gamma is not None and float(gamma.detach().abs().sum()) == 0:
        with torch.no_grad():
            gamma.fill_(0.5)
        return True
    return False


def check_models():
    torch.manual_seed(0)
    for version in C.VERSIONS:
        for arm in C.arms(version):
            model, directory = load_net(version, arm)
            try:
                key = f"{version}/{arm}"
                trainable = sum(p.numel() for p in model.parameters())
                assert trainable == C.TRAINABLE_PARAMS[key], (key, trainable)
                model.train(False)
                with torch.no_grad():
                    outputs = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
                assert len(outputs) == 4 and outputs[0].shape == (2, 1, 256, 256), key
                assert all(torch.isfinite(o).all() for o in outputs), key
                deploy = model.switch_to_deploy()
                deployed = sum(p.numel() for p in model.parameters())
                assert deployed == C.DEPLOY_PARAMS[key], (key, deployed)
                with torch.no_grad():
                    folded = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
                difference = max(float((a - b).abs().max()) for a, b in zip(outputs, folded))
                assert difference <= 2e-3, (key, difference)
                model.switch_to_train() if hasattr(model, "switch_to_train") else None
            finally:
                sys.path.remove(directory)
    print("[VALIDATE] matrix, CLI, and per-version model audit passed")


def check_gradients(steps=1):
    torch.manual_seed(0)
    for version in C.VERSIONS:
        for arm in C.arms(version):
            if arm == "none":
                continue
            model, directory = load_net(version, arm)
            try:
                if open_gate(model):
                    pass
                model.train()
                images, targets = torch.rand(2, 3, 256, 256), (torch.rand(2, 1, 256, 256) > 0.9).float()
                outputs, auxiliary = model(images, images, return_aux=True)
                loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(o, targets) for o in outputs)
                loss.backward()
                context_grads = [p.grad for n, p in model.named_parameters() if n.startswith("context.")]
                assert any(g is not None and float(g.abs().sum()) > 0 for g in context_grads), (version, arm)
                head = dict(model.named_parameters())["head.weight"].grad
                assert head is not None and float(head.abs().sum()) > 0
            finally:
                sys.path.remove(directory)
    print("[VALIDATE] context and head gradients are non-zero on every geometry arm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gradients", action="store_true")
    args = parser.parse_args()
    check_matrix()
    check_cli()
    check_models()
    if not args.skip_gradients:
        check_gradients()


if __name__ == "__main__":
    main()
