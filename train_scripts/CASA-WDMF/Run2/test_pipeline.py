"""Check the four-slot schedule with the retained LEVIR process."""
from pipeline import available_gpu, pending_runs


def main():
    done = ["M0_CTRL/SYSU-CD-256", "M1_HYPER_QKV/SYSU-CD-256",
            "M2_CA_QKV/SYSU-CD-256", "M1_HYPER_QKV/LEVIR-CD-256",
            "M2_CA_QKV/LEVIR-CD-256"]
    kept = "M0_CTRL/LEVIR-CD-256"
    jobs = {1: dict(name=kept, gpu=0)}
    pending = pending_runs(done, [kept])
    assert len(pending) == 6 and len(set(pending)) == 6 and kept not in pending
    assignments = []
    for pid in range(2, 5):
        gpu = available_gpu(jobs)
        assignments.append(gpu)
        jobs[pid] = dict(name=pending.pop(0), gpu=gpu)
    assert assignments == [0, 1, 1]
    assert available_gpu(jobs) is None
    assert {job["name"].split("/")[1] for pid, job in jobs.items() if pid != 1} == {"WHU-CD-256"}
    jobs.pop(1)
    assert available_gpu(jobs) == 0
    assert all(sum(job["gpu"] == gpu for job in jobs.values()) <= 2 for gpu in (0, 1))
    assert all("CDD-CD-256" in name for name in pending)
    print("Retained process, unique pending runs and two slots per GPU PASS")


if __name__ == "__main__":
    main()
