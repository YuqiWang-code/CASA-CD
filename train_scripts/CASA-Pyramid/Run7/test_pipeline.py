"""Eight runs, dataset priority and capacity backfill on physical GPU0."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline import PROTOCOL, VARIANTS, check_selection, choose_run, pending_runs


def selection_gate():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        profile = {mode: dict(protocol=PROTOCOL, peak_reserved_mib=1000, context_overhead_mib=100)
                   for mode in VARIANTS.values()}
        (root / 'profile.json').write_text(json.dumps(profile))
        decision = dict(protocol=PROTOCOL, selected_arm='candidate', passed=False)
        (root / 'selection_decision.json').write_text(json.dumps(decision))
        try:
            check_selection(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('failed selection must not start formal training')
        decision['passed'] = True
        (root / 'selection_decision.json').write_text(json.dumps(decision))
        selected, measured = check_selection(root)
        assert selected == decision and measured == profile
        (root / 'profile.json').unlink()
        try:
            check_selection(root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('formal training requires final-model profiling')


def main():
    selection_gate()
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
    print('selection/profile gate, 8 unique runs, GPU0-only, priority and capacity backfill PASS')


if __name__ == '__main__':
    main()
