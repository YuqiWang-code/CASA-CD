"""Static and model-level audit for Run16 (no training, no GPU).

Run16 replays two module-removal arms on the three archived version trees, so
the audit has to prove three things per tree: the two new switches exist on the
CLI, the swapped-in modules reproduce the published parameter counts once
folded, and the replaced layers really carry gradient (a silent no-op would
make the whole round meaningless).
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import torch

import config as C

CONTROL = dict(module="none", tar="reparam", decoder="reparam")
REQUIRED = {
    "TAR_DIFF": ("tar.stage1.temporal.proj.weight", "tar.stage4.temporal.proj.weight"),
    "DCR_SINGLE": ("decoder.proj.weight", "decoder.block.dw.weight"),
}


def check_matrix():
    jobs = C.experiments()
    assert len(jobs) == 12, len(jobs)
    assert len(set(jobs)) == len(jobs)
    assert C.arms() == ("TAR_DIFF", "DCR_SINGLE"), C.arms()
    for version in C.VERSIONS:
        for arm in C.arms():
            assert arm in C.TRAINABLE_PARAMS and arm in C.DEPLOY_PARAMS
            assert C.train_script(version).name == "train_CASA.py"
            assert C.VERSIONS[version]["selection"] == "val"
    assert len(C.experiments()) == len(C.VERSIONS) * len(C.arms()) * len(C.DATASETS)


def check_cli():
    for version in C.VERSIONS:
        text = subprocess.check_output([sys.executable, str(C.train_script(version)), "--help"],
                                       text=True, stderr=subprocess.STDOUT)
        assert "--tar-variant {reparam,plain,diff}" in text, version
        assert "--decoder-variant {reparam,plain,none}" in text, version


def load_net(version, options):
    directory = str(C.RUNTIME / C.model_dir(version))
    for name in [n for n in list(sys.modules) if n == "model" or n.startswith("model.")]:
        del sys.modules[name]
    sys.path.insert(0, directory)
    module = importlib.import_module("model.casa_net")
    return module.CASANet(module=options["module"], tar_variant=options["tar"],
                          decoder_variant=options["decoder"]), directory


def check_models():
    torch.manual_seed(0)
    for version in C.VERSIONS:
        reference = None
        for arm in C.arms():
            model, directory = load_net(version, C.ARMS[arm])
            try:
                trainable = sum(p.numel() for p in model.parameters())
                assert trainable == C.TRAINABLE_PARAMS[arm], (version, arm, trainable)
                model.train(False)
                with torch.no_grad():
                    outputs = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
                    if reference is None:
                        control, reference_directory = load_net(version, CONTROL)
                        try:
                            control.train(False)
                            with torch.no_grad():
                                reference = control(torch.rand(2, 3, 256, 256),
                                                    torch.rand(2, 3, 256, 256))
                        finally:
                            sys.path.remove(reference_directory)
                assert len(outputs) == 4 and outputs[0].shape == (2, 1, 256, 256), (version, arm)
                assert all(torch.isfinite(o).all() for o in outputs), (version, arm)
                # the swap must actually change the network at initialisation
                shift = max(float((a - b).abs().max()) for a, b in zip(outputs, reference))
                assert shift > 1e-4, (version, arm, shift)
                deploy = model.switch_to_deploy()
                deployed = sum(p.numel() for p in model.parameters())
                assert deployed == C.DEPLOY_PARAMS[arm], (version, arm, deployed)
                with torch.no_grad():
                    folded = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
                difference = max(float((a - b).abs().max()) for a, b in zip(outputs, folded))
                assert difference <= 2e-3, (version, arm, difference)
            finally:
                sys.path.remove(directory)
    print("[VALIDATE] matrix, CLI, and per-version model audit passed")


def check_gradients():
    torch.manual_seed(0)
    for version in C.VERSIONS:
        for arm in C.arms():
            model, directory = load_net(version, C.ARMS[arm])
            try:
                model.train()
                early = torch.rand(2, 3, 256, 256)
                late = torch.rand(2, 3, 256, 256)
                targets = (torch.rand(2, 1, 256, 256) > 0.9).float()
                outputs, _ = model(early, late, return_aux=True)
                loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(o, targets)
                           for o in outputs)
                loss.backward()
                named = dict(model.named_parameters())
                for name in REQUIRED[arm]:
                    grad = named[name].grad
                    assert grad is not None and float(grad.abs().sum()) > 0, (version, arm, name)
                head = named["head.weight"].grad
                assert head is not None and float(head.abs().sum()) > 0, (version, arm)
            finally:
                sys.path.remove(directory)
    print("[VALIDATE] replaced layers and head carry gradient on every version")


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


