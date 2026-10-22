"""Schedule Run2 across GPU0 and GPU1, with two training slots per GPU."""
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
state = {"status": "running", "protocol": PROTOCOL, "completed": [], "active": {},
         "active_gpus": {}, "failed": [], "slots_per_gpu": {"0": 2, "1": 2}}


def pending_runs(completed, running):
    occupied = set(completed) | set(running)
    return [f"{variant}/{dataset}" for dataset in DATASETS for variant in VARIANTS
            if f"{variant}/{dataset}" not in occupied]


def available_gpu(jobs):
    for gpu in (0, 1):
        if sum(job["gpu"] == gpu for job in jobs.values()) < 2:
            return gpu
    return None


def alive(pid):
    command = Path(f"/proc/{pid}/cmdline")
    return command.exists() and b"train_wdmf.py" in command.read_bytes()


def complete(name):
    path = ROOT / name / "train_log.txt"
    return path.exists() and "=== END TEST RESULTS ===" in path.read_text(errors="replace")


def placement(pid, job, adopted=False):
    (ROOT / job["name"] / "placement.json").write_text(json.dumps(dict(
        pid=pid, physical_gpu=job["gpu"], logical_gpu=0, adopted=adopted,
        observed_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))


def save():
    state["active"] = {str(pid): job["name"] for pid, job in active.items()}
    state["active_gpus"] = {str(pid): job["gpu"] for pid, job in active.items()}
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (ROOT / "pipeline_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def stop_children():
    for pid in active:
        if alive(pid):
            os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for pid, job in active.items():
        process = job["process"]
        if process is not None:
            try:
                process.wait(timeout=max(0.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
        else:
            while alive(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            if alive(pid):
                os.killpg(pid, signal.SIGKILL)


def interrupted(signum, frame):
    state["status"] = "stopped"
    stop_children()
    active.clear()
    save()
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, choices=[4], required=True)
    parser.parse_args()
    for module in ("hyperbolic", "qkv"):
        result = json.loads((ROOT / "probes" / module / "health_result.json").read_text())
        if not result["passed"] or result["protocol"] != PROTOCOL or result["steps"] < 3000:
            raise RuntimeError(f"health probe incomplete: {module}")
    prior_path = ROOT / "pipeline_status.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    state["completed"] = [name for name in pending_runs([], []) if complete(name)]
    for text_pid, name in prior.get("active", {}).items():
        pid = int(text_pid)
        if name in state["completed"]:
            continue
        if not alive(pid):
            raise RuntimeError(f"unfinished prior process is absent: {name} ({pid})")
        gpu = int(prior.get("active_gpus", {}).get(text_pid, 0))
        active[pid] = dict(name=name, gpu=gpu, process=None)
        placement(pid, active[pid], adopted=True)
        print(f"[ADOPT] {name} pid={pid} gpu={gpu}", flush=True)
    pending = pending_runs(state["completed"], [job["name"] for job in active.values()])
    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        while pending or active:
            for pid, job in list(active.items()):
                process = job["process"]
                if process is None:
                    if alive(pid):
                        continue
                    code = 0 if complete(job["name"]) else -1
                else:
                    code = process.poll()
                    if code is None:
                        continue
                name = active.pop(pid)["name"]
                if code != 0 or not complete(name):
                    state["failed"].append(dict(run=name, exit_code=code, gpu=job["gpu"]))
                    raise RuntimeError(f"{name} exited {code} without complete TEST results")
                state["completed"].append(name)
                print(f"[DONE] {name} gpu={job['gpu']}", flush=True)
                save()
            while pending:
                gpu = available_gpu(active)
                if gpu is None:
                    break
                name = pending.pop(0)
                variant, dataset = name.split("/")
                print(f"[START] {name} gpu={gpu} {time.strftime('%F %T')}", flush=True)
                process = subprocess.Popen(["bash", str(HERE / "run_one.sh"),
                                            variant, dataset, str(gpu)], start_new_session=True)
                active[process.pid] = dict(name=name, gpu=gpu, process=process)
                (ROOT / name).mkdir(parents=True, exist_ok=True)
                placement(process.pid, active[process.pid])
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
