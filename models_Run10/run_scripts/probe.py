"""Run six isolated centered probes using existing measured memory reservations."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from probe_config import ROOT, PROFILE_PATH, VARIANTS, PROBES, SEED, TEMPERATURE, valid_probe, control_references


def probe_command(here, arm, dataset, directory):
    spec=VARIANTS[arm]
    return [sys.executable,str(here/'health_probe.py'),'--models-dir',str(spec['models']),
            '--module',spec['module'],'--dataset',dataset,'--epochs','4','--seed',str(SEED),
            '--compression-temperature',str(TEMPERATURE),'--retrieval-weight','0',
            '--output-dir',str(directory)]


def main():
    controls=control_references()
    profiles=json.loads(PROFILE_PATH.read_text())
    reservations={arm:math.ceil(profiles[spec['profile']]['peak_reserved_mib']+
                               profiles[spec['profile']]['context_overhead_mib'])
                  for arm,spec in VARIANTS.items()}
    for spec in VARIANTS.values():
        if not (spec['models']/'train_CASA.py').is_file():raise FileNotFoundError(spec['models'])
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'],text=True).strip())
    budget=free-1024
    ROOT.mkdir(parents=True,exist_ok=True)
    pidfile=ROOT/'probe.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text()); command=Path(f'/proc/{pid}/cmdline')
        active_command = command.read_bytes() if command.exists() else b''
        if (str(Path(__file__).resolve()).encode() in active_command
                or b'selection/centered/scripts/runner.py' in active_command):
            raise RuntimeError(f'centered probe runner is already active: {pid}')
    pidfile.write_text(str(os.getpid()))
    schedule=dict(physical_gpu=1,max_concurrent=3,gpu_budget_mib=budget,
                  reserved_mib=reservations,profile_reference=str(PROFILE_PATH),seed=SEED,
                  compression_temperature_init=TEMPERATURE,experiments=[f'{a}/{d}' for a,d in PROBES])
    (ROOT/'scheduling.json').write_text(json.dumps(schedule,indent=2))
    active,reports=[],{}
    pending=list(PROBES)
    here=Path(__file__).resolve().parent
    def interrupted(signum,frame):raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while pending or active:
            for job in active[:]:
                code=job['process'].poll()
                if code is None:continue
                job['stream'].close();active.remove(job)
                path=job['directory']/'health_result.json'
                if code or not path.exists():raise RuntimeError(f"probe failed: {job['name']}")
                report=json.loads(path.read_text())
                if not valid_probe(report,job['arm'],job['dataset']):raise RuntimeError(f"invalid probe report: {job['name']}")
                reports[job['name']]=report
                print('[PROBE-DONE]',job['name'],flush=True)
            while pending and len(active)<3:
                used=sum(reservations[job['arm']] for job in active)
                selected=next(((a,d) for a,d in pending if used+reservations[a]<=budget),None)
                if selected is None:
                    if not active:raise RuntimeError('pending probe cannot fit physical GPU1')
                    break
                arm,dataset=selected;pending.remove(selected);name=f'{arm}/{dataset}'
                directory=ROOT/'probes'/arm/dataset;path=directory/'health_result.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if not valid_probe(report,arm,dataset):raise RuntimeError(f'invalid existing probe: {name}')
                    reports[name]=report;continue
                directory.mkdir(parents=True,exist_ok=True)
                stream=(directory/'probe.log').open('a')
                process=subprocess.Popen(probe_command(here,arm,dataset,directory),stdout=stream,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                active.append(dict(process=process,stream=stream,arm=arm,dataset=dataset,
                                   name=name,directory=directory))
                print('[PROBE-START]',name,'pid',process.pid,flush=True)
            (ROOT/'probe_status.json').write_text(json.dumps(dict(
                completed=list(reports),active={str(j['process'].pid):j['name'] for j in active},
                pending=[f'{a}/{d}' for a,d in pending]),indent=2))
            time.sleep(5)
    finally:
        for job in active:
            if job['process'].poll() is None:os.killpg(job['process'].pid,signal.SIGTERM)
        for job in active:
            try:job['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(job['process'].pid,signal.SIGKILL);job['process'].wait()
            job['stream'].close()
    (ROOT/'probe_summary.json').write_text(json.dumps(dict(
        purpose='paired centered readout and supervision probes only',formal_result=False,
        all_health_passed=True,physical_gpu=1,seed=SEED,compression_temperature_init=TEMPERATURE,
        probes=reports,controls=controls,scheduling=schedule),indent=2))
    print('[SIX-PROBES-COMPLETE] Results require paired analysis.',flush=True)


if __name__=='__main__':main()
