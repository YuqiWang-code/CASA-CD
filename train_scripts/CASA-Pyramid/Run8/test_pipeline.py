"""Phase-one readiness and capacity scheduling."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from pipeline import DATASETS,VARIANTS,PROTOCOL,pending_runs,choose_run,check_readiness


def main():
    tasks=pending_runs([],[])
    assert len(tasks)==len(set(tasks))==4
    assert tasks[:2]==['H_CLASS/SYSU-CD-256','H_NONE/SYSU-CD-256']
    required={name:11200 for name in VARIANTS};limits={1:25000}
    first=choose_run(tasks,{},limits,required)
    assert first==('H_CLASS/SYSU-CD-256',1,11200)
    active={1:dict(gpu=1,reserved_mib=11200)}
    second=choose_run(tasks[1:],active,limits,required)
    assert second==('H_NONE/SYSU-CD-256',1,11200)
    active[2]=dict(gpu=1,reserved_mib=11200)
    assert choose_run(tasks[2:],active,limits,required) is None
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        (root/'readiness.json').write_text(json.dumps(dict(protocol=PROTOCOL,passed=True,phase='retrieval_loss',comparison='H_CLASS_vs_H_NONE')))
        (root/'profile.json').write_text(json.dumps({key:dict(protocol=PROTOCOL,peak_reserved_mib=10438,context_overhead_mib=760) for key in VARIANTS}))
        decision,profile=check_readiness(root)
        assert decision['phase']=='retrieval_loss' and len(profile)==2
    print('4 unique phase-one jobs, paired SYSU priority, GPU1 capacity and readiness PASS')


if __name__=='__main__':main()
