#!/usr/bin/env bash
# STG_SUP 探查：进度 / 曲线 / 机制读数 / GPU（只读）
set -u
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_SUP
echo "=== TIME ==="; date '+%F %T %z'
echo "=== GPU ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== 归属 ==="
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  printf '  pid=%s user=%s cmd=%s\n' "$p" "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" \
    "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-70)"
done
echo "=== 进程 ==="; pgrep -af "[t]rain_stgeo\.py" | sed 's/ --dataset_root.*//' | head -2
echo "=== 进度 ==="
for d in LEVIR-CD-256 SYSU-CD-256; do
  L="$EXP/$d/train_log.txt"
  if [ ! -f "$L" ]; then echo "  $d: 尚未开始"; continue; fi
  st=$(grep -a '^\[ACTUAL-OPT-STEPS\] epoch=' "$L" | tail -1 | sed 's/.*cumulative=//')
  bl=$(grep -ac '=== END TEST RESULTS ===' "$L")
  echo "  --- $d  steps=${st:-NA} blocks=$bl ---"
  echo "  前 2:"; grep -a '^Epoch ' "$L" | head -2 | sed 's/^/    /'
  echo "  近 4:"; grep -a '^Epoch ' "$L" | tail -4 | sed 's/^/    /'
  echo "  best: $(grep -a '^\[BEST\]' "$L" | tail -1)"
  echo "  异常行数: $(grep -acE '\[RETRY|\[FAIL|Traceback|CUDA out of memory' "$L")"
  echo "  机制: $(grep -a '^\[STGEO-GRAD\]' "$L" | head -1)"
  echo "        $(grep -a '^\[STGEO-LOSS\]' "$L" | tail -1)"
  echo "        aux_loss 序列(前3/后3): $(grep -a '^\[STGEO-LOSS\]' "$L" | sed 's/.*aux_loss=//;s/ .*//' | head -3 | tr '\n' ' ') ... $(grep -a '^\[STGEO-LOSS\]' "$L" | sed 's/.*aux_loss=//;s/ .*//' | tail -3 | tr '\n' ' ')"
done
echo "=== 对照（CTRL_R8 同 epoch 的 F1，仅趋势）==="
B=/home/yqwang/outputs/CASA-CD/CASA-TViM
e=$(grep -a '^Epoch ' "$EXP/LEVIR-CD-256/train_log.txt" 2>/dev/null | tail -1 | awk '{print $2}' | cut -d/ -f1)
if [ -n "${e:-}" ]; then
  for arm in "Run8/CTRL_R8" "Exploration/Exp1/STG_INJ" "Exploration/Exp1/STG_V2" "Exploration/Exp1/STG_SUP"; do
    v=$(grep -a "^Epoch ${e}/" "$B/$arm/LEVIR-CD-256/train_log.txt" 2>/dev/null | tail -1 | sed 's/.*F1=//;s/ |.*//')
    m=$(grep -a '^Epoch ' "$B/$arm/LEVIR-CD-256/train_log.txt" 2>/dev/null | head -"$e" | sed 's/.*F1=//;s/ |.*//' | sort -g | tail -1)
    printf '  epoch %-4s %-28s F1=%-8s 到该 epoch 的最大 F1=%s\n' "$e" "$arm" "${v:-NA}" "${m:-NA}"
  done
fi
