"""Run each dataset's three ablations before advancing to the next dataset."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run1")
HERE = Path(__file__).resolve().parent
DATASETS = ["SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256"]
VARIANTS = ["M0_CTRL", "M1_HYPER_QKV", "M2_CA_QKV"]
active = {}
state = {"status": "running", "completed": [], "active": {}, "failed": []}


def save():
    state["active"] = {str(p.pid): name for p, name in active.items()}
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (ROOT / "pipeline_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def stop_children():
    for process in active:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in active:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def interrupted(signum, frame):
    state["status"] = "stopped"
    stop_children()
    active.clear()
    save()
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for dataset in DATASETS:
            pending = list(VARIANTS)
            while pending or active:
                for process in list(active):
                    code = process.poll()
                    if code is None:
                        continue
                    name = active.pop(process)
                    if code != 0:
                        state["failed"].append(dict(run=name, exit_code=code))
                        raise RuntimeError(f"{name} exited {code}")
                    state["completed"].append(name)
                    print(f"[DONE] {name}", flush=True)
                    save()
                while pending and len(active) < args.parallel:
                    variant = pending.pop(0)
                    name = f"{variant}/{dataset}"
                    print(f"[START] {name} {time.strftime('%F %T')}", flush=True)
                    process = subprocess.Popen(["bash", str(HERE / "run_one.sh"), variant, dataset],
                                               start_new_session=True)
                    active[process] = name
                    save()
                    time.sleep(2)
                time.sleep(5)
        state["status"] = "complete"
        save()
        print("[COMPLETE] all 12 runs", flush=True)
    except Exception:
        state["status"] = "failed"
        stop_children()
        active.clear()
        save()
        raise


if __name__ == "__main__":
    main()
