"""Check exact Run9 matrix, GPU capacity, and required probe gate."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from config import PROBES, COMPARISON, experiments
from pipeline import VARIANTS, PROTOCOL, pending_runs, choose_run, check_readiness


def main():
    tasks = pending_runs([], [])
    assert len(tasks) == len(set(tasks)) == 14
    assert tasks[:3] == ['H_REFINE/LEVIR-CD-256','E_REFINE/LEVIR-CD-256','M0_CTRL/LEVIR-CD-256']
    assert [name for name in tasks if name.startswith('E_BASE/')] == ['E_BASE/SYSU-CD-256','E_BASE/WHU-CD-256']
    assert pending_runs(tasks[:2],tasks[2:4]) == tasks[4:]
    required = {name:11200 for name in VARIANTS}
    limits = {1:25000}
    assert choose_run(tasks,{},limits,required) == (tasks[0],1,11200)
    active = {1:dict(gpu=1,reserved_mib=11200)}
    assert choose_run(tasks[1:],active,limits,required) == (tasks[1],1,11200)
    active[2] = dict(gpu=1,reserved_mib=11200)
    assert choose_run(tasks[2:],active,limits,required) is None
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        decision=dict(protocol=PROTOCOL,passed=True,phase='four_dataset_refinement',comparison=COMPARISON,
                      validation=dict(protocol=PROTOCOL,passed=True),
                      probes={f'{arm}/{dataset}':dict(health_passed=True,task_gradient_check=True) for arm,dataset in PROBES})
        (root/'readiness.json').write_text(json.dumps(decision))
        (root/'profile.json').write_text(json.dumps({key:dict(protocol=PROTOCOL,peak_reserved_mib=10438,context_overhead_mib=760,module=spec['module'],refinement_beta=spec['beta'],auxiliary_weights=dict(guide=.1,geometric_retrieval=0.)) for key,spec in VARIANTS.items()}))
        assert len(check_readiness(root)[1]) == 4
        del decision['probes']['M0_CTRL/SYSU']
        (root/'readiness.json').write_text(json.dumps(decision))
        try:
            check_readiness(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('missing CTRL probe accepted')
    print('14 unique jobs, exact E_BASE subset, GPU1 capacity and six-probe gate PASS')


if __name__=='__main__':
    main()
