"""Check matrix uniqueness, queue priority and measured GPU1-only capacity."""
from config import DATASETS, VARIANTS, experiments
from pipeline import choose_run, pending_runs


def main():
    names = experiments()
    assert len(names) == len(set(names)) == 24
    assert [sum(name.endswith(dataset) for name in names) for dataset in DATASETS] == [9, 9, 3, 3]
    assert names[:3] == [f'{variant}/LEVIR-CD-256' for variant in ('M4_MOE', 'M0_CTRL', 'M4_EQUAL')]
    requirements = {name: 14000 if mode != 'none' else 9000 for name, mode in VARIANTS.items()}
    pending, jobs, started = pending_runs([], []), {}, []
    while (choice := choose_run(pending, jobs, {1: 31000}, requirements)) is not None:
        name, gpu, reserved = choice
        assert gpu == 1
        pending.remove(name)
        started.append(name)
        jobs[len(jobs)] = dict(name=name, reserved_mib=reserved)
        assert sum(job['reserved_mib'] for job in jobs.values()) <= 31000
    assert started == names[:2]
    assert pending_runs(started, []) == pending
    assert choose_run(names, {}, {1: 31000}, requirements, free_mib=9000) is None
    assert choose_run(names, {}, {1: 31000}, requirements, free_mib=11000)[0].startswith('M0_CTRL/')
    print('24 experiments, dataset priority, GPU1-only capacity and external-memory guard PASS')


if __name__ == '__main__':
    main()
