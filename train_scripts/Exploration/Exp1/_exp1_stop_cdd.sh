#!/usr/bin/env bash
# 收窄范围：停掉 Exp1 当前波（正在跑 CDD），并**改名封存**已产生的部分产物（绝不删除）。
# 只动本探索线的进程：模式均为 Exp1/train_stgeo 专属，绝不触碰 GPU0 上的 Run8。
set -u
REPO=/home/yqwang/projects/CASA-CD
EXP=/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== 1) 停 run_all 启动器与 train_stgeo（只匹配本探索线）==="
pkill -f "Exploration/Exp1/run_all\.sh" && echo "killed run_all" || echo "run_all 不在运行"
pkill -f "[t]rain_stgeo\.py" && echo "killed train_stgeo" || echo "train_stgeo 不在运行"
sleep 5
pkill -9 -f "[t]rain_stgeo\.py" 2>/dev/null || true
sleep 3

echo "=== 2) 确认只停了我们的进程 ==="
pgrep -af "[t]rain_stgeo\.py" | head -3 || echo "（无残留）"
echo "--- Run8 进程仍在？（必须仍在）---"
pgrep -af "[t]rain\.py" | head -2 || echo "（Run8 无进程，需注意）"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "=== 3) 改名封存 CDD 的部分产物 ==="
for d in "$EXP/STG_INJ/CDD-CD-256" "$CKPT/STG_INJ/CDD-CD-256"; do
  if [ -d "$d" ]; then
    base=$(basename "$d"); parent=$(dirname "$d")
    new="$parent/${base}.SUPERSEDED_${STAMP}_scopecut_cdd"
    mv "$d" "$new" && echo "[ARCHIVE] $d -> $new"
  fi
done
rm -f /tmp/exp1/run_all.pid
echo "=== DONE ==="
