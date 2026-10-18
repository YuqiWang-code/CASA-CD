#!/usr/bin/env bash
# STG_V2 曲线 / 吞吐 / ETA（只读）
set -u
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_V2
for d in LEVIR-CD-256 SYSU-CD-256; do
  L="$EXP/$d/train_log.txt"
  if [ ! -f "$L" ]; then echo "--- $d: 尚未开始 ---"; continue; fi
  echo "=================== $d ==================="
  st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
  echo "steps=$st  blocks=$(grep -ac '=== END TEST RESULTS ===' "$L")"
  echo "--- 前 3 个 epoch ---"; grep -a '^Epoch ' "$L" | head -3
  echo "--- 最近 5 个 epoch ---"; grep -a '^Epoch ' "$L" | tail -5
  echo "--- best / 异常 ---"
  grep -a '^\[BEST\]' "$L" | tail -1
  grep -acE '\[RETRY|\[FAIL|Traceback|CUDA out of memory' "$L" | sed 's/^/异常行数: /'
  # 吞吐与 ETA
  python3 - "$L" "$d" <<'PY'
import re, sys, datetime
log, ds = sys.argv[1], sys.argv[2]
txt = open(log, encoding="utf-8", errors="replace").read()
ts = [int(x) for x in re.findall(r"\| t=(\d+)s", txt)]
iters = {"LEVIR-CD-256": 223, "SYSU-CD-256": 375}.get(ds, 223)
if ts:
    avg = sum(ts[-20:]) / len(ts[-20:])
    per_step = avg / iters
    cur = 0
    m = re.findall(r"^\[ACTUAL-OPT-STEPS\] epoch=\d+ executed=\d+ cumulative=(\d+)", txt, re.M)
    if m: cur = int(m[-1])
    remain = max(0, 80000 - cur)
    eta_h = remain * per_step / 3600
    print(f"  平均 epoch {avg:.0f}s（最近20个） ⇒ {per_step:.3f} s/step；"
          f"已 {cur}/80000，剩余 ≈{eta_h:.1f} h")
PY
  echo
done
echo "=== 当前时间 ==="; date '+%F %T'
