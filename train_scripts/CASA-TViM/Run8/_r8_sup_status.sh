#!/usr/bin/env bash
# 只看督导/编排器存活（含单卡版），不依赖 _r8_state.sh 的匹配模式。
set -u
echo "=== $(date '+%F %T') ==="
for p in $(pgrep -f '_r8_pipeline_supervisor' 2>/dev/null || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
  echo "  督导 pid=$p et=$et  $(echo "$cmd" | cut -c1-95)"
done
echo "  pidfile: $(cat /tmp/run8/pipeline_single.pid 2>/dev/null || echo none)"
for p in $(pgrep -f '_r8_run_one_ds\.sh' 2>/dev/null || true); do
  echo "  编排器 pid=$p  $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-95)"
done
echo "--- GPU1 是否真的有别人 ---"
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
echo "--- 卡1 上的进程归属 ---"
MINE=$( { pgrep -f '^python train\.py' 2>/dev/null; pgrep -f '_r8_run_one_ds\.sh' 2>/dev/null; } | sort -u )
while IFS=, read -r pid mem uuid; do
  pid=$(echo "${pid:-}" | tr -d ' '); [ -z "$pid" ] && continue
  if echo "$MINE" | grep -qx "$pid"; then own=OWN; else own="FOREIGN(user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' '))"; fi
  echo "  pid=$pid mem=$mem -> $own"
done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader)
echo "--- GPU1 空闲已持续多久（看 nvidia-smi 无法给，改用我们的探测日志）---"
grep -E 'GPU1' /tmp/run8/pipeline_single.log 2>/dev/null | tail -3
