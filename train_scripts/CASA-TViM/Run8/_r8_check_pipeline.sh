#!/usr/bin/env bash
# 检查并行督导是否**重复**运行，并确认它能正确认出"我们自己的 GPU 占用"。
# 只 kill 重复的 `_r8_pipeline_supervisor.sh`；**绝不触碰** train.py / _r8_run_one_ds.sh。
set -u
STATE=/tmp/run8

echo "=== $(date '+%F %T') ==="
echo "--- 并行督导实例 ---"
FOUND=0
declare -a PIDS=()
for p in $(pgrep -f '_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
  echo "  pid=$p elapsed=$et"
  PIDS+=("$p"); FOUND=$((FOUND+1))
done
echo "  共 $FOUND 个"

if [ "$FOUND" -gt 1 ]; then
  # 保留**最早启动**的那个（父级），杀掉后启动的重复实例
  KEEP="${PIDS[0]}"
  echo "--- 保留最早实例 pid=$KEEP，其余 TERM ---"
  for p in "${PIDS[@]}"; do
    [ "$p" = "$KEEP" ] && continue
    kill -TERM "$p" 2>/dev/null && echo "  TERM pid=$p" || echo "  TERM pid=$p 失败"
  done
  sleep 3
  echo "$KEEP" > "$STATE/pipeline_supervisor.pid"
  echo "  pidfile 已更新为 $KEEP"
fi

echo "--- own_pids 自检：它认得出我们的两个 run 吗 ---"
MINE=$( { pgrep -f '^python train\.py' 2>/dev/null
          pgrep -f '_r8_run_one_ds\.sh|_r8_supervisor\.sh' 2>/dev/null
          for p in $(pgrep -f 'python' 2>/dev/null); do
            cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
            case "$cmd" in *"/home/yqwang/projects/CASA-CD"*) echo "$p" ;; esac
          done; } | sort -u )
echo "  own_pids = [$(echo "$MINE" | tr '\n' ' ')]"
echo "  GPU 上的 compute 进程与归属："
while IFS=, read -r pid mem; do
  pid=$(echo "${pid:-}" | tr -d ' '); [ -z "$pid" ] && continue
  if echo "$MINE" | grep -qx "$pid"; then own="OWN"; else own="FOREIGN"; fi
  echo "    pid=$pid mem=$mem -> $own"
done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)

echo "--- pipeline.log 尾部 ---"
tail -12 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
