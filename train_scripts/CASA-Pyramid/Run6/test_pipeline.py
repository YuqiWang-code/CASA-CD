"""Eight runs, dataset priority and capacity backfill on physical GPU0."""
from pipeline import VARIANTS, choose_run, pending_runs


def main():
    assert list(VARIANTS) == ['M1_HYPER_QKV', 'M0_CTRL']
    pending = pending_runs([], [])
    assert len(pending) == len(set(pending)) == 8
    assert all('/LEVIR-' in name for name in pending[:2])
    assert all('/SYSU-' in name for name in pending[2:4])
    limits = {0: 31000}
    required = {'M1_HYPER_QKV': 12000, 'M0_CTRL': 9000}
    jobs = {}
    started = []
    while (choice := choose_run(pending, jobs, limits, required)) is not None:
        name, gpu, memory = choice
        assert gpu == 0
        started.append(name)
        pending.remove(name)
        jobs[len(jobs)] = dict(name=name, gpu=gpu, reserved_mib=memory)
        assert sum(job['reserved_mib'] for job in jobs.values()) <= limits[0]
    assert started == ['M1_HYPER_QKV/LEVIR-CD-256', 'M0_CTRL/LEVIR-CD-256', 'M0_CTRL/SYSU-CD-256']
    assert not set(pending) & set(started)
    assert pending_runs(started, []) == pending
    print('8 unique runs, GPU0-only, dataset priority and fitting-job backfill PASS')


if __name__ == '__main__':
    main()
