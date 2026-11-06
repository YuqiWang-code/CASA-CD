"""Validate the archived runtime, measure GPU memory, then run paired health probes."""
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
import signal
import subprocess
import sys
import time
from config import ROOT, RUNTIME, PROTOCOL, VARIANTS, PROBES, PRETRAINED

HERE = Path(__file__).resolve().parent


def main():
    pidfile = ROOT / 'prepare.pid'
    if pidfile.exists():
        old = Path('/proc') / pidfile.read_text().strip() / 'cmdline'
        if old.exists() and (str(HERE / 'prepare.py').encode() in old.read_bytes()
                             or (b'prepare.py' in old.read_bytes() and old.with_name('cwd').resolve() == HERE)):
            raise RuntimeError('Run11 preparation already active')
    pidfile.write_text(str(os.getpid()))
    subprocess.run([sys.executable, str(HERE/'validate.py')], check=True)
    subprocess.run([sys.executable, str(RUNTIME/'models/profile_CASA.py'),
                    '--dataset-root','/share_datasets/CD/LEVIR-CD-256','--pretrained',PRETRAINED,
                    '--interior-positive-weight','1.25','--output',str(ROOT/'profile.json')], check=True)
    profile = json.loads((ROOT/'profile.json').read_text())
    requirements = {v:math.ceil(profile[s['module']]['peak_reserved_mib'] +
                               profile[s['module']]['context_overhead_mib'] + 128) for v,s in VARIANTS.items()}
    free = lambda: int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'],text=True).strip())
    budget = free()-1024
    if max(requirements.values())>budget: raise RuntimeError('GPU1 cannot fit a measured Run11 job')
    scheduling = dict(protocol=PROTOCOL,gpu_budget_mib={'1':budget},variant_reserved_mib=requirements,
                      physical_gpu=1,free_headroom_mib=1024,profile_loss_weight=1.25)
    (ROOT/'scheduling.json').write_text(json.dumps(scheduling,indent=2))
    pending, active, completed = list(PROBES), {}, []
    def stop(signum, frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            for pid,job in list(active.items()):
                code=job['process'].poll()
                if code is None: continue
                job['stream'].close(); del active[pid]
                report=ROOT/'probes'/job['variant']/job['dataset']/'health_result.json'
                if code or not report.exists() or not json.loads(report.read_text())['passed']:
                    raise RuntimeError('probe failed: '+job['variant']+'/'+job['dataset'])
                completed.append(job['variant']+'/'+job['dataset'])
            while pending and len(active)<2:
                used=sum(requirements[j['variant']] for j in active.values())
                selected=next(((v,d) for v,d in pending if used+requirements[v]<=budget and requirements[v]<=free()-1024),None)
                if selected is None: break
                v,d=selected;pending.remove(selected)
                directory=ROOT/'probes'/v/d;directory.mkdir(parents=True,exist_ok=True)
                stream=(directory/'probe.log').open('w')
                process=subprocess.Popen([sys.executable,str(HERE/'health_probe.py'),v,d],stdout=stream,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                active[process.pid]=dict(process=process,stream=stream,variant=v,dataset=d)
                print('[PROBE-START]',v,d,process.pid,flush=True)
            (ROOT/'probe_status.json').write_text(json.dumps(dict(completed=completed,
                active={str(pid):j['variant']+'/'+j['dataset'] for pid,j in active.items()},pending=pending),indent=2))
            time.sleep(5)
        (ROOT/'readiness.json').write_text(json.dumps(dict(protocol=PROTOCOL,passed=True,
            validation='validation_result.json',probes=completed,formal_initialization='fresh pretrained'),indent=2))
        print('[READY] validation and all paired health probes passed',flush=True)
    finally:
        for job in active.values():
            if job['process'].poll() is None: os.killpg(job['process'].pid,signal.SIGTERM)
        for job in active.values():
            try:job['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job['process'].pid,signal.SIGKILL)
            job['stream'].close()


if __name__=='__main__':
    main()
