#!/usr/bin/env bash
# 探索线全面探查（只读）：GPU / 等卡督导 / 训练进程 / v1 产物 / 讨论用摘要
set -u
STATE=/tmp/exp1
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
CK=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1

echo "=== TIME ==="; date '+%F %T %z'
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== GPU 占用归属 ==="
while IFS=, read -r uuid pid name mem; do
  pid=$(echo "$pid" | tr -d ' ')
  u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  echo "  pid=$pid user=${u:-?} mem=$(echo $mem | tr -d ' ') ${name:0:40}"
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)

echo
echo "=== 等卡督导 ==="
P=$(cat "$STATE/wait_v2.pid" 2>/dev/null || echo "")
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then echo "  supervisor=ALIVE pid=$P"; else echo "  supervisor=DEAD (${P:-no pidfile})"; fi
echo "  --- wait log: 行数 $(wc -l < "$STATE/wait_STG_V2.log" 2>/dev/null || echo 0) ---"
echo "  首行: $(head -1 "$STATE/wait_STG_V2.log" 2>/dev/null)"
echo "  末 3 行:"; tail -3 "$STATE/wait_STG_V2.log" 2>/dev/null | sed 's/^/    /'
echo "  occupied 次数: $(grep -c 'occupied' "$STATE/wait_STG_V2.log" 2>/dev/null || echo 0)"

echo
echo "=== 训练进程 ==="
echo "--- Exp1 train_stgeo ---"; pgrep -af "[t]rain_stgeo\.py" | cut -c1-90 || echo "  (无)"
echo "--- Run8 train.py ---";    pgrep -af "[t]rain\.py --dataset" | sed 's/ --dataset_root.*//' | head -2

echo
echo "=== Exp1 v1 产物（已完成，LEVIR + SYSU）==="
for v in STG_INJ STG_V2; do
  for d in LEVIR-CD-256 SYSU-CD-256 CDD-CD-256 WHU-CD-256; do
    L="$EXP/$v/$d/train_log.txt"
    [ -f "$L" ] || continue
    st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
    bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
    f1=$(grep -a '^Recall=' "$L" | tail -1 | sed 's/.*F1=//;s/ |.*//')
    printf '  %-8s %-13s steps=%-13s blocks=%s  F1(TEST)=%s\n' "$v" "$d" "${st:-NA}" "$bl" "${f1:-NA}"
  done
done
echo "  封存目录:"; ls -d "$EXP/STG_INJ/"*SUPERSEDED* 2>/dev/null | sed 's/^/    /'
echo "  ckpt 目录:"; ls -d "$CK"/*/*/ 2>/dev/null | sed 's/^/    /'
echo "  磁盘:"; du -sh "$CK"/* 2>/dev/null | sed 's/^/    /'

echo
echo "=== v1 正式结果（同源 TEST 区块）==="
python3 - <<'PY'
import re
base = "/home/yqwang/outputs/CASA-CD/CASA-TViM"
def last(p):
    try: txt = open(p, encoding="utf-8", errors="replace").read()
    except FileNotFoundError: return None
    b = re.findall(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", txt, re.S)
    if not b: return None
    m = re.search(r"Recall=([\d.]+) \| Precision=([\d.]+) \| OA=([\d.]+) \| F1=([\d.]+) \| IoU=([\d.]+)", b[-1])
    return dict(zip(["R","P","OA","F1","IoU"], [float(x) for x in m.groups()])) if m else None
print(f"{'arm':10s} {'dataset':14s} {'F1':>7s} {'R':>7s} {'P':>7s} {'IoU':>7s}")
for arm, path in [("CTRL_R8", f"{base}/Run8/CTRL_R8"), ("STG_INJ", f"{base}/Exploration/Exp1/STG_INJ")]:
    for ds in ["LEVIR-CD-256", "SYSU-CD-256"]:
        r = last(f"{path}/{ds}/train_log.txt")
        if r: print(f"{arm:10s} {ds:14s} {r['F1']:7.4f} {r['R']:7.4f} {r['P']:7.4f} {r['IoU']:7.4f}")
PY
echo "=== DONE ==="
