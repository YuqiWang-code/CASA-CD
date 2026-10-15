#!/usr/bin/env bash
# Run8 从「单卡串行督导」切换到「双卡并行流水线」的过渡脚本。
#
# 起因：2026-10-15 09:06 实测 GPU1 上别人的作业已下线（15 MiB / 0%），而 Run8 单 run
# 峰值 27.5 GB ⇒ 每卡只能放 1 个 run ⇒ 双卡并行可把每波 makespan 从 28.7 h 压到 14.3 h。
#
# 本脚本做三件事，**绝不触碰任何正在跑的 train.py**：
#   1. 停掉旧的单卡督导 `_r8_supervisor.sh`（只按进程名找 `_r8_supervisor.sh`；
#      注意新督导叫 `_r8_pipeline_supervisor.sh`，正则要精确，别误杀新的）；
#   2. 立即在 **GPU1** 起 `CTRL_R8 / SYSU-CD-256`（GPU1 现在空着，不用等）；
#   3. 后台拉起 `_r8_pipeline_supervisor.sh`（它会自动 skip 已完成/在跑的库，
#      等 GPU0 空出来再跑 WHU，然后依次做波二/三/四）。
set -u
REPO=/home/yqwang/projects/CASA-CD
RUN_DIR="$REPO/train_scripts/CASA-TViM/Run8"
STATE=/tmp/run8
PIDFILE="$STATE/supervisor.pid"
PIPEFILE="$STATE/pipeline_supervisor.pid"
mkdir -p "$STATE"

echo "=== Run8 双卡并行切换 $(date '+%F %T') ==="

echo "--- 1) 停旧单卡督导（精确匹配 _r8_supervisor.sh，且**必须排除 pipeline**）---"
# ⚠️ 2026-10-15 血泪：`pgrep -f '_r8_supervisor\.sh'` **会同时匹配** `_r8_pipeline_supervisor.sh`
# （正则子串匹配！）⇒ 上一版把它刚拉起的并行督导连同 fork 一起 TERM 掉，
# 导致一个编排器在写完 run_manifest.json 后被打断、留下孤儿 manifest、后续全部 ABORT。
# 正确做法：用锚定正则 `_r8_supervisor\.sh$`（cmdline 里该脚本是最后一个参数）**并且**
# 再显式排掉任何含 `pipeline` 的行。
for p in $(pgrep -f '_r8_supervisor\.sh$' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in
    *pipeline*) echo "  SKIP pid=$p（并行督导，不动）"; continue ;;
    *bash\ -c*) continue ;;
    */_r8_supervisor.sh) : ;;                       # 只认"脚本名就是 _r8_supervisor.sh"的
    *) echo "  SKIP pid=$p（非单卡督导）"; continue ;;
  esac
  echo "  TERM pid=$p  $cmd"
  kill -TERM "$p" 2>/dev/null || true
done
sleep 3
echo "  剩余单卡督导: $(pgrep -f '_r8_supervisor\.sh$' | wc -l) 个（应为 0）"

echo "--- 2) 立即在 GPU1 起 CTRL_R8/SYSU（脱离会话）---"
if [ -s "$STATE/run_CTRL_R8_SYSU-CD-256.log" ] && pgrep -f '_r8_run_one_ds.sh CTRL_R8 SYSU-CD-256' >/dev/null; then
  echo "  已在跑，跳过"
else
  setsid nohup bash "$RUN_DIR/_r8_run_one_ds.sh" CTRL_R8 SYSU-CD-256 1 \
      > "$STATE/run_CTRL_R8_SYSU-CD-256.log" 2>&1 < /dev/null &
  echo "  SYSU pid=$!  log=$STATE/run_CTRL_R8_SYSU-CD-256.log"
fi

echo "--- 3) 拉起双卡并行流水线督导（脱离会话）---"
if [ -f "$PIPEFILE" ] && kill -0 "$(cat "$PIPEFILE")" 2>/dev/null; then
  echo "  已运行 pid=$(cat "$PIPEFILE")"
else
  setsid nohup bash "$RUN_DIR/_r8_pipeline_supervisor.sh" \
      > "$STATE/pipe.stdout" 2>&1 < /dev/null &
  echo "$!" > "$PIPEFILE"
  echo "  pipeline pid=$(cat "$PIPEFILE")  log=$STATE/pipe.stdout"
fi

sleep 8
echo "--- 4) 切换后状态 ---"
echo "  GPU:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/    /'
echo "  我们的 train.py:"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  echo "    pid=$p ds=$ds CUDA_VISIBLE_DEVICES=$cvd"
done
echo "  督导:"
for p in $(pgrep -f '_r8_supervisor\.sh|_r8_pipeline_supervisor\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "    pid=$p $cmd"
done
echo "  pipeline.log 尾部:"
tail -6 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/    /' || echo "    (尚无)"
