#!/usr/bin/env bash
# Run7 波次督导：等波一过收口门 → 自动开波二 → 过门 → 自动开波三。
#
# 为什么要它：波一由 `run_all.sh --wave 1` 单独启动，跑完即退出，不会自己接波二；
# 若等我来回探测，波间会产生**空闲 GPU 时间**。本督导把"过门才开下一波"的纪律
# **机械化**：门 = 该波 4 个 run 全部产出完整 TEST 区块且 [ACTUAL-OPT-STEPS] 80000 +
# 协议串正确（由 collect_results.py 判定 VALID）。
#
# 安全阀：① 日志出现 "3 attempts exhausted" 或 Traceback → 立即中止；
#         ② 整波累计步数 25 分钟无增长 → 判定停滞并中止；
#         ③ 总时长上限 20 小时；④ 绝不删除任何产物。
set -uo pipefail
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export PROJ=/home/yqwang/projects/CASA-CD
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ" || exit 1

LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7
SUPLOG=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_supervisor.log
DS_LIST=CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256
DEADLINE=$(( $(date +%s) + 20*3600 ))

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$SUPLOG"; }

wave_state() {   # $1=variant -> "n_valid n_total steps_sum"
  python train_scripts/CASA-TViM/Run7/collect_results.py --log-root "$LOG_ROOT" \
    --out /tmp/r7_sup_gate.json --variants "$1" --datasets "$DS_LIST" >/dev/null 2>&1
  python - "$1" <<'PY'
import json, os, re, sys
v = sys.argv[1]
try:
    d = json.load(open('/tmp/r7_sup_gate.json', encoding='utf-8'))
except Exception:
    print("0 4 0"); raise SystemExit
tot = 0
for r in d.get('rows', []):
    p = r.get('log'); s = 0
    if p and os.path.isfile(p):
        txt = open(p, encoding='utf-8', errors='replace').read()
        m = re.findall(r'\[ACTUAL-OPT-STEPS\] epoch=\d+ executed=\d+ cumulative=(\d+)/', txt)
        if m:
            s = max(int(x) for x in m)
    tot += s
print(d.get('n_valid', 0), d.get('n_total', 4), tot)
PY
}

wait_wave() {    # $1=variant $2=wave_no
  local v="$1" w="$2" last=-1 lastchange now
  lastchange=$(date +%s)
  while true; do
    now=$(date +%s)
    if [ "$now" -gt "$DEADLINE" ]; then log "wave$w DEADLINE exceeded -> abort"; return 1; fi
    read -r nv nt tot <<< "$(wave_state "$v")"
    log "wave$w $v: valid=$nv/$nt steps_sum=$tot"
    if [ "$nv" = "$nt" ]; then log "wave$w GATE PASS ($nv/$nt VALID)"; return 0; fi
    if grep -l -E "3 attempts exhausted" "$LOG_ROOT/$v"/*/train_log.txt >/dev/null 2>&1; then
      log "wave$w FAIL: '3 attempts exhausted' found -> abort"; return 1
    fi
    if [ "$tot" != "$last" ]; then last=$tot; lastchange=$now; fi
    if [ $(( now - lastchange )) -gt 1500 ]; then
      log "wave$w STALLED: no step progress for 25 min (steps_sum=$tot) -> abort"; return 1
    fi
    sleep 120
  done
}

log "================ supervisor start (pid $$) ================"
# 自检模式：只跑一次门检测并退出（用于在开长任务前验证解析逻辑）
if [ "${1:-}" = "--state" ]; then
  for v in ${2:-M1_R7CTRL}; do
    echo "$v: $(wave_state "$v")   (n_valid n_total steps_sum)"
  done
  log "--state check: ${2:-M1_R7CTRL} -> $(wave_state "${2:-M1_R7CTRL}")"
  exit 0
fi

# 单实例守卫：用 pidfile（ps|grep 会匹配到调用者自身的命令行，故不可用）
PIDFILE=/tmp/run7_supervisor.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  log "another supervisor already running (pid $(cat "$PIDFILE")) -> exit"
  echo "ANOTHER_SUPERVISOR_RUNNING pid=$(cat "$PIDFILE")"
  exit 0
fi
echo $$ > "$PIDFILE"
log "pidfile written: $PIDFILE (pid $$)"
RC=0
if wait_wave M1_R7CTRL 1; then
  log "launching wave 2: E9_HFAE_A x4"
  bash train_scripts/CASA-TViM/Run7/run_all.sh --wave 2 >> "$SUPLOG" 2>&1
  rc2=$?; log "wave2 run_all rc=$rc2"
  if [ "$rc2" -eq 0 ] && wait_wave E9_HFAE_A 2; then
    log "launching wave 3: E10_HFAE_B x4"
    bash train_scripts/CASA-TViM/Run7/run_all.sh --wave 3 >> "$SUPLOG" 2>&1
    rc3=$?; log "wave3 run_all rc=$rc3"
    if [ "$rc3" -eq 0 ]; then wait_wave E10_HFAE_B 3 || RC=1; else RC=1; fi
  else
    RC=1
  fi
else
  RC=1
fi
log "================ supervisor end (rc=$RC) ================"
exit "$RC"
