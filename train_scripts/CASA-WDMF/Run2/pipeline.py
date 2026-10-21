"""Run SYSU, LEVIR, WHU and CDD with two total GPU0 training slots."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/home/yqwang/outputs/CASA-CD/CASA-WDMF/Run2")
HERE = Path(__file__).resolve().parent
DATASETS = ["SYSU-CD-256", "LEVIR-CD-256", "WHU-CD-256", "CDD-CD-256"]
VARIANTS = ["M1_HYPER_QKV", "M2_CA_QKV", "M0_CTRL"]
PROTOCOL = "wdmf_str_40k_modulewd0_v2"
active = {}
state = {"status": "running", "protocol": PROTOCOL, "completed": [], "active": {}, "failed": []}


def reserved_ctrl():
    path = ROOT.parent / "Run1/pipeline_status.json"
    if not path.exists():
        return 0
    prior = json.loads(path.read_text())
    for pid, name in prior.get("active", {}).items():
        if name == "M0_CTRL/SYSU-CD-256":
            command = Path(f"/proc/{pid}/cmdline")
            if command.exists() and b"train_wdmf.py" in command.read_bytes():
                return 1
    return 0


def save():
    state["active"] = {str(p.pid): name for p, name in active.items()}
    state["reserved_run1_ctrl"] = reserved_ctrl()
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
    parser.add_argument("--parallel", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    for module in ("hyperbolic", "qkv"):
        result = json.loads((ROOT / "probes" / module / "health_result.json").read_text())
        if not result["passed"] or result["protocol"] != PROTOCOL or result["steps"] < 3000:
            raise RuntimeError(f"health probe incomplete: {module}")
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
                while pending and len(active) < min(args.parallel, 2-reserved_ctrl()):
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
