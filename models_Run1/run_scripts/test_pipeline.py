"""Verify dataset priority and memory-limited GPU placement."""
from pipeline import choose_run, pending_runs


def main():
    pending = pending_runs([], [])
    assert len(pending) == len(set(pending)) == 12
    assert all('/LEVIR-' in name for name in pending[:3])
    assert all('/SYSU-' in name for name in pending[3:6])
    limits = {0: 30000, 1: 30000}
    required = {'M1_HYPER_QKV': 14000, 'M2_CA_QKV': 13000, 'M0_CTRL': 9000}
    jobs = {}
    starts = []
    while True:
        choice = choose_run(pending, jobs, limits, required)
        if choice is None:
            break
        name, gpu, memory = choice
        starts.append(name)
        pending.remove(name)
        jobs[len(jobs)] = dict(name=name, gpu=gpu, reserved_mib=memory)
        assert sum(j['reserved_mib'] for j in jobs.values() if j['gpu'] == gpu) <= limits[gpu]
    assert all('/LEVIR-' in name for name in starts[:3])
    assert {job['gpu'] for job in jobs.values()} == {0, 1}
    assert pending and len(jobs) >= 4
    blocked = ['M1_HYPER_QKV/SYSU-CD-256', 'M0_CTRL/SYSU-CD-256']
    choice = choose_run(blocked, {1: dict(gpu=0, reserved_mib=21000),
                                 2: dict(gpu=1, reserved_mib=22000)}, limits, required)
    assert choice == ('M0_CTRL/SYSU-CD-256', 0, 9000)
    remaining = pending_runs(starts, [j['name'] for j in jobs.values()])
    assert set(remaining) == set(pending)
    assert not set(remaining) & set(starts)
    print('12 unique runs, LEVIR/SYSU priority, both GPUs, memory budgets and fitting-job backfill PASS')


if __name__ == '__main__':
    main()
