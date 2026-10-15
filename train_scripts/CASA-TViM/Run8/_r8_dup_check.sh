#!/usr/bin/env bash
# Run8 重复进程彻查（只读）：确认是否真的有第二个 run 在跑。
# 判据不看 `pgrep -f '^python train\.py'` 的行数（它会把 parent + 4 个 DataLoader worker
# 都算进来），而看 **`--ckpt_dir` 的唯一性** 与 **CUDA 进程数**。
set -u
echo "=== $(date '+%F %T') ==="

echo "--- 每个 train.py 进程：pid / ppid / ckpt_dir / 启动时刻 ---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  ck=$(echo "$cmd" | sed -n 's/.*--ckpt_dir \([^ ]*\).*/\1/p')
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
  lstart=$(ps -o lstart= -p "$p" 2>/dev/null)
  echo "  pid=$p ppid=$ppid ds=$ds et=$et"
  echo "      ckpt=$ck"
  echo "      started: $lstart"
done

echo "--- 唯一 ckpt_dir 数（应 = 正在跑的 run 数）---"
pgrep -f '^python train\.py' | while read -r p; do
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | sed -n 's/.*--ckpt_dir \([^ ]*\).*/\1/p'
done | sort -u | sed 's/^/  /'

echo "--- 父进程链（谁起的）---"
for p in $(pgrep -f '^python train\.py' || true); do
  ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  echo "  pid=$p <- ppid=$ppid: $(tr '\0' ' ' < /proc/$ppid/cmdline 2>/dev/null | cut -c1-100)"
done

echo "--- 所有 Run8 相关脚本实例 ---"
for pat in '_r8_run_one_ds\.sh' '_r8_pipeline_supervisor\.sh' '_r8_supervisor\.sh' 'run_all\.sh'; do
  echo "  [$pat]"
  for p in $(pgrep -f "$pat" || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
    case "$cmd" in *bash\ -c*) continue ;; esac
    et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
    echo "    pid=$p et=$et $(echo "$cmd" | cut -c1-100)"
  done
done

echo "--- GPU 上的 compute 进程 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo "--- CPU 负载（判断是否有重复训练在抢 CPU）---"
uptime
echo "  我们 train.py+worker 的 CPU 占用前 8："
ps -eo pid,pcpu,rss,etimes,comm --sort=-pcpu 2>/dev/null | head -9 | sed 's/^/    /'

echo "--- WHU 进度与速率 ---"
F=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/CTRL_R8/WHU-CD-256/train_log.txt
grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -3 | sed 's/^/  /'
grep -E '^Epoch ' "$F" | tail -3 | sed 's/^/  /'
