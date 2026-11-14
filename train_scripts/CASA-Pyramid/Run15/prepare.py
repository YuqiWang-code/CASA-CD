"""CPU audit, per-version smoke test, and scheduling for Run15."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config as C

SMOKE_TRAIN = 40
SMOKE_VAL = 8
SMOKE_TEST = 8


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def source_names(dataset, split, count):
    listing = C.DATA / dataset / "list" / f"{split}.txt"
    names = [line.strip() for line in listing.read_text(encoding='utf-8').splitlines() if line.strip()]
    return names[:count]


def smoke_split(version, dataset="WHU-CD-256"):
    root = C.ROOT / "preflight" / version / "data"
    if root.exists():
        shutil.rmtree(root)
    for split, count in (("train", SMOKE_TRAIN), ("val", SMOKE_VAL), ("test", SMOKE_TEST)):
        names = source_names(dataset, split, count)
        (root / "list").mkdir(parents=True, exist_ok=True)
        (root / "list" / f"{split}.txt").write_text("\n".join(names) + "\n", encoding='utf-8')
        for folder in ("A", "B", "label"):
            target = root / folder
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                link = target / name
                if not link.exists():
                    link.symlink_to(C.DATA / dataset / folder / name)
    return root


def run_smoke(version, arm, root, output):
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    command = [sys.executable, str(C.train_script(version)),
               "--dataset-root", str(root), "--output-dir", str(output),
               "--checkpoint-dir", str(checkpoints), "--pretrained", str(C.PRETRAINED),
               *C.model_options(version, arm),
               "--batch-size", "32", "--max-steps", str(2 * SMOKE_TRAIN), "--num-workers", "0"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(C.GPUS[0]))
    result = subprocess.run(command, cwd=str(output), capture_output=True, text=True, env=environment)
    log = output / "smoke.log"
    log.write_text(result.stdout + "\n" + result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f'smoke failed for {version}/{arm}: see {log}')
    text = log.read_text(encoding='utf-8')
    if not re.search(r'=== TEST RESULTS ===.*?=== END TEST RESULTS ===', text, re.S):
        raise RuntimeError(f'smoke produced no TEST RESULTS block: {log}')
    payload = json.loads((output / "result.json").read_text(encoding='utf-8'))
    assert payload["selection_split"] == "val", payload["selection_split"]
    assert payload["independent_test"] is True, payload.get("independent_test")
    assert payload["optimizer_steps"] >= 1
    assert abs(payload["total_params_m"] - C.TRAINABLE_PARAMS[f"{version}/{arm}"] / 1e6) < 1e-6
    assert payload["deploy_binary_disagreement"] <= 1
    return dict(optimizer_steps=payload["optimizer_steps"],
                selection_split=payload["selection_split"], independent_test=payload["independent_test"],
                total_params_m=payload["total_params_m"],
                deploy_max_diff=payload["deploy_max_diff"],
                deploy_binary_disagreement=payload["deploy_binary_disagreement"])


def scheduler_budget():
    import subprocess as sp
    total = int(sp.check_output(['nvidia-smi', '-i', str(C.GPUS[0]), '--query-gpu=memory.total',
                                 '--format=csv,noheader,nounits'], text=True).strip())
    return {str(gpu): total - C.HEADROOM for gpu in C.GPUS}


def main():
    started = time.time()
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "validate.py")], check=True)
    smoke = {version: run_smoke(version, "hyperbolic", smoke_split(version),
                                C.ROOT / "preflight" / version / "smoke") for version in C.VERSIONS}
    reservations = {f"{version}/{arm}": C.DEFAULT_RESERVATIONS[arm]
                    for version in C.VERSIONS for arm in C.arms(version)}
    write_json(C.ROOT / "scheduling.json", dict(
        gpu_budget_mib=scheduler_budget(), variant_reserved_mib=reservations,
        slots_per_gpu=C.SLOTS, free_headroom_mib=C.HEADROOM))
    readiness = dict(protocol=C.PROTOCOL, passed=True, timestamp=time.strftime('%F %T'),
                     datasets=list(C.DATASETS), versions=list(C.VERSIONS),
                     experiments=[list(job) for job in C.experiments()],
                     smoke=smoke, reservations=reservations,
                     trainable_params=C.TRAINABLE_PARAMS, deploy_params=C.DEPLOY_PARAMS,
                     elapsed_seconds=round(time.time() - started, 1))
    write_json(C.ROOT / "readiness.json", readiness)
    print(f"[READY] Run15 audit and smoke tests passed in {readiness['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
