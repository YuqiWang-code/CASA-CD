"""Wait for CASA-Pyramid Run8 completion before using physical GPU1."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from config import ROOT, runtime_root

PREVIOUS = Path('/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8')
child = None
state = dict(status='waiting_for_Run8', physical_gpu=1)


def save(**fields):
    state.update(fields, updated=time.strftime('%F %T'))
    (ROOT/'handoff_status.json').write_text(json.dumps(state,indent=2))


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait()
    save(status='stopped')
    raise SystemExit(128+signum)


def run(command, phase):
    global child
    save(status=phase)
    log=(ROOT/'pipeline.log').open('a') if phase=='formal_training' else None
    child=subprocess.Popen(command,start_new_session=True,stdout=log,stderr=subprocess.STDOUT if log else None)
    save(child_pid=child.pid)
    code=child.wait()
    if log:
        log.close()
    if code:
        raise RuntimeError(f'{phase} exited {code}; preserve logs and checkpoints')
    child=None
    save(child_pid=None)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'handoff.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'handoff_status.json').exists():
        raise RuntimeError('handoff already recorded; inspect before explicit recovery')
    (ROOT/'handoff.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    save(pid=os.getpid())
    try:
        while True:
            previous=json.loads((PREVIOUS/'pipeline_status.json').read_text())
            if previous.get('failed') or previous['status'] in ('failed','stopped'):
                raise RuntimeError('Run8 stopped or failed; do not take over GPU1')
            if previous['status']=='complete' and not previous.get('active'):
                expected=[f'{v}/{d}-CD-256' for v in ('H_CLASS','H_NONE') for d in ('SYSU','WHU')]
                for name in expected:
                    text=(PREVIOUS/name/'train_log.txt').read_text()
                    if '=== TEST RESULTS ===' not in text or '=== END TEST RESULTS ===' not in text:
                        raise RuntimeError(f'Run8 missing complete test: {name}')
                alive=[]
                for file in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        cmd=file.read_bytes()
                    except (FileNotFoundError,PermissionError,ProcessLookupError):
                        continue
                    if b'/CASA-Pyramid/Run8/' in cmd and (b'train_CASA.py' in cmd or b'pipeline.py' in cmd):
                        alive.append(file.parent.name)
                if not alive:
                    break
            time.sleep(60)
        save(status='waiting_for_memory', previous_completed=previous['updated'])
        while int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'],text=True)) < 18000:
            time.sleep(60)
        scripts=runtime_root()/'train_scripts/CASA-MoE/Run2'
        run(['bash',str(scripts/'validate.sh')],'validation_and_profile')
        run(['bash',str(scripts/'probe.sh')],'real_data_probes')
        run(['python',str(scripts/'configure_schedule.py')],'configure_schedule')
        run(['python',str(scripts/'pipeline.py')],'formal_training')
        save(status='complete')
    except Exception as error:
        save(status='failed',reason=str(error))
        raise


if __name__=='__main__':
    main()
