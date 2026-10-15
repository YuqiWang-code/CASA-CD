#!/usr/bin/env bash
# 排卡探查（只读）：两张卡的真实占用、每张卡上的 compute 进程、哪些是本项目的。
# 用途：决定能否把四库并行铺到两张卡上（Run8 实测单 run 峰值 26.9–27.7 GB，
# 每卡只放 1 个 run 才安全；Run7 波三因同卡 2 run + 别人作业而连续 OOM 3 次）。
set -u
source /home/yqwang/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true

echo "=== $(date '+%F %T') ==="
echo "--- 每卡占用 ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "--- 每卡 compute 进程（pid / 显存 / 卡号）---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "  （uuid -> index 映射）"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader

echo "--- 本项目进程 ---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  ck=$(echo "$cmd" | sed -n 's/.*--ckpt_dir \([^ ]*\).*/\1/p')
  tv=$(echo "$ck" | sed -n 's#.*/Run8/\([^/]*\)/.*#\1#p')
  echo "  pid=$p  dataset=$ds  variant=${tv:-?}  gpu_id(CLI)=$(echo "$cmd" | sed -n 's/.*--gpu_id \([0-9]*\).*/\1/p')"
  echo "    CUDA_VISIBLE_DEVICES=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
  echo "    显存(从 nvidia-smi) :"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$p," | sed 's/^/      /'
done

echo "--- 别的用户/别的项目进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mem; do
  pid=$(echo "$pid" | tr -d ' ')
  [ -z "$pid" ] && continue
  if ! pgrep -f '^python train\.py' | grep -qx "$pid"; then
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-110)
    echo "  pid=$pid user=${owner:-?} mem=${mem} cmd=$cmd"
  fi
done

echo "--- 每卡可用余量（相对单 run 需要 28 GB）---"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | while IFS=, read -r i tot used; do
  i=$(echo "$i"|tr -d ' '); tot=$(echo "$tot"|tr -d ' '); used=$(echo "$used"|tr -d ' ')
  free=$((tot-used))
  verdict=$([ "$free" -ge 30500 ] && echo "可放 1 个 run" || echo "余量不足（需 30500MB）")
  echo "  GPU$i free=${free}MB -> $verdict"
done

echo "--- 督导状态 ---"
if [ -f /tmp/run8/supervisor.pid ] && kill -0 "$(cat /tmp/run8/supervisor.pid)" 2>/dev/null; then
  echo "  SUPERVISOR ALIVE pid=$(cat /tmp/run8/supervisor.pid)"
else
  echo "  SUPERVISOR 未运行"
fi
tail -6 /tmp/run8/supervisor.log 2>/dev/null | sed 's/^/  /'
