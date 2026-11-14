"""Check fixed SUP launch, actual centered probes and archived controls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from config import PROTOCOL, GUIDE_OBJECTIVE, TEMPERATURE, VARIANTS, experiments, training_options
from pipeline import pending_runs, choose_run, check_readiness
from probe_config import PROBES, VARIANTS as CANDIDATES
from probe import probe_command


def main():
    jobs=pending_runs([],[])
    assert len(jobs)==len(set(jobs))==12
    assert jobs[:3]==['H_POSTERIOR/LEVIR-CD-256','E_POSTERIOR/LEVIR-CD-256','M0_CTRL/LEVIR-CD-256']
    assert pending_runs(jobs[:2],jobs[2:4])==jobs[4:]
    assert choose_run(jobs,{}, {1:25000},{a:11200 for a in VARIANTS})==(jobs[0],1,11200)
    assert choose_run(jobs,{1:dict(gpu=1,reserved_mib=22400)},{1:25000},{a:11200 for a in VARIANTS}) is None
    assert TEMPERATURE==.5 and len(PROBES)==6
    for arm in VARIANTS:
        options=training_options(arm)
        if arm=='M0_CTRL':assert '--compression-temperature' not in options
        else:assert options[options.index('--compression-temperature')+1]=='0.5'
    here=Path(__file__).resolve().parent
    for arm,dataset in PROBES:
        command=probe_command(here,arm,dataset,Path('/tmp/probe'))
        assert command[command.index('--seed')+1]=='2333'
        assert command[command.index('--compression-temperature')+1]=='0.5'
    with TemporaryDirectory() as temporary:
        root=Path(temporary)
        for arm in ('H_POSTERIOR_T05','E_POSTERIOR_T05','M0_CTRL'):
            phase='neighborhood' if arm=='M0_CTRL' else 'classposterior'
            for dataset in ('LEVIR','SYSU'):
                directory=root/'selection'/phase/'probes'/arm/dataset;directory.mkdir(parents=True)
                (directory/'health_result.json').write_text(json.dumps(dict(health_passed=True,epochs=4)))
                manifest=dict(seed=2333,train_loader_seed=2333,validation_loader_seed=2334,
                              guide_weight=.1,retrieval_weight=0,evaluation_split='val',batch_size=32,
                              compression_temperature_init=.5)
                (directory/'run_manifest.json').write_text(json.dumps(manifest))
        for arm,dataset in PROBES:
            spec=CANDIDATES[arm];directory=root/'selection/centered/probes'/arm/dataset;directory.mkdir(parents=True)
            report=dict(health_passed=True,task_gradient_check=True,posterior_gradient_check=True,
                        protocol=spec['protocol'],guide_objective=spec['objective'],module=spec['module'],
                        dataset=dataset+'-CD-256',epochs=4,seed=2333,compression_temperature_init=.5,retrieval_weight=0)
            (directory/'health_result.json').write_text(json.dumps(report))
            (directory/'run_manifest.json').write_text(json.dumps(dict(guide_objective=spec['objective'],
                seed=2333,compression_temperature_init=.5)))
        (root/'validation_result.json').write_text(json.dumps(dict(protocol=PROTOCOL,passed=True)))
        profile={arm:dict(protocol=PROTOCOL,module=spec['module'],peak_reserved_mib=10438,
                 context_overhead_mib=760,auxiliary_weights=dict(guide=.1,geometric_retrieval=0.),
                 guide_objective=GUIDE_OBJECTIVE if arm!='M0_CTRL' else 'raw_area_bce',
                 compression_temperature_init=.5 if arm!='M0_CTRL' else None) for arm,spec in VARIANTS.items()}
        (root/'profile.json').write_text(json.dumps(profile))
        try:check_readiness(root)
        except FileNotFoundError:pass
        else:raise AssertionError('missing review accepted')
        review=dict(protocol=PROTOCOL,passed=True,guide_objective=GUIDE_OBJECTIVE,compression_temperature_init=.5)
        (root/'review_decision.json').write_text(json.dumps(review))
        decision,_=check_readiness(root)
        assert len(decision['probes'])==6 and len(decision['controls'])==6
        assert decision['guide_objective']==GUIDE_OBJECTIVE and decision['compression_temperature_init']==.5
        review['compression_temperature_init']=.1
        (root/'review_decision.json').write_text(json.dumps(review))
        try:check_readiness(root)
        except RuntimeError:pass
        else:raise AssertionError('obsolete temperature search selected for formal SUP')
        review['compression_temperature_init']=.5
        (root/'review_decision.json').write_text(json.dumps(review))
        profile['H_POSTERIOR']['guide_objective']='raw_area_bce'
        (root/'profile.json').write_text(json.dumps(profile))
        try:check_readiness(root)
        except RuntimeError:pass
        else:raise AssertionError('raw-only training profile accepted for formal SUP')
    print('12 GPU1 jobs, six actual centered probes, fixed .5 and supervised-profile review PASS')


if __name__=='__main__':main()
