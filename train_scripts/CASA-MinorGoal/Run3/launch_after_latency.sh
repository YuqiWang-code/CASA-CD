#!/usr/bin/env bash
# Run3 launcher: start the attribution queue once Run2 has fully released GPU1.
#
# Why: Run2's own watch_loop keeps the two 5090 slots busy, so starting Run3 now
# would either be refused by pipeline.py's memory gate or fight Run2 for slots.
# The latency probe is no longer a prerequisite (it is a secondary artefact), so
# this wrapper only waits for (a) Run2's pipeline process to be gone (its
# watch_loop only observes and never exits) and (b) >= 25000 MiB free on GPU1,
# for at most 6 hours. It then execs the queue, so launcher.pid becomes the
# queue's PID and never goes stale.
set -u

RUN3=/home/yqwang/projects/CASA-CD/train_scripts/CASA-MinorGoal/Run3
LAUNCHER_LOG="$RUN3/launcher.log"
PIDFILE="$RUN3/launcher.pid"
MIN_FREE_MIB=25000
MAX_WAIT_S=$((24 * 3600))
DEADLINE=$(( $(date +%s) + MAX_WAIT_S ))

run2_pipeline_active() {
    local pid cwd
    for pid in $(pgrep -f 'pipeline\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        case "$cwd" in
            */CASA-MinorGoal/Run2*) return 0 ;;
        esac
    done
    return 1
}

run2_progress() {
    local status=/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run2/pipeline_status.json
    if [ -f "$status" ]; then
        printf 'done=%s failed=%s' \
            "$(grep -o '"completed"' "$status" | wc -l)" \
            "$(python -c 'import json,sys;d=json.load(open(sys.argv[1]));print(len(d.get("failed",{})))' "$status" 2>/dev/null || echo 0)"
    else
        printf 'done=? failed=?'
    fi
}

echo "$$" > "$PIDFILE"
echo "[LAUNCH] $(date '+%F %T') launcher pid $$ waiting for Run2 to release GPU1 (min ${MIN_FREE_MIB} MiB, max 24 h)"
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PYTHONPATH=/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib
export PYTHONUNBUFFERED=1
cd "$RUN3" || { echo "[LAUNCH] cannot cd $RUN3"; exit 1; }

while :; do
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
        echo "[LAUNCH] $(date '+%F %T') timeout: GPU1 still below ${MIN_FREE_MIB} MiB free after 24 h"
        exit 3
    fi
    run2=0
    run2_pipeline_active && run2=1
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' \r')
    free_mib=${free_mib:-0}
    echo "[WAIT] $(date '+%F %T') run2_pipeline=$run2 $(run2_progress) gpu1_free=${free_mib}MiB"
    if [ "$run2" -eq 0 ] && [ "$free_mib" -ge "$MIN_FREE_MIB" ]; then
        break
    fi
    sleep 60
done

echo "[LAUNCH] $(date '+%F %T') Run2 released GPU1 (free ${free_mib} MiB) -> starting Run3 pipeline"
exec python pipeline.py
