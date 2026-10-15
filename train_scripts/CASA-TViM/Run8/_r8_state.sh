#!/usr/bin/env bash
# Run8 真实状态快照（只读）：进程 / 编排器日志 / ckpt 目录 / 日志尾部 / GPU。
# 之所以要这个：pipeline.log 的 tail 会让人误以为"两个 stage 都失败了"，
# 实际它只是把**上一轮调度器**留下的 STAGE-DONE 一起 tail 出来了。
set -u
STATE=/tmp/run8
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8

echo "=== $(date '+%F %T') ==="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo "--- 我们的 train.py（含 CUDA_VISIBLE_DEVICES）---"
for p in $(pgrep -f '^python train\.py' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([^ ]*\).*/\1/p')
  cvd=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  et=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
  echo "  pid=$p ds=$ds CUDA_VISIBLE_DEVICES=$cvd elapsed=$et"
done

echo "--- 编排器 / 督导 ---"
for p in $(pgrep -f '_r8_run_one_ds\.sh|_r8_pipeline_supervisor\.sh|_r8_supervisor\.sh|run_all\.sh' || true); do
  cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || echo "")
  case "$cmd" in *bash\ -c*) continue ;; esac
  echo "  pid=$p $(echo "$cmd" | cut -c1-110)"
done

echo "--- 每个编排器日志（真实内容，不 tail 混流）---"
for f in "$STATE"/run_*.log; do
  [ -f "$f" ] || continue
  echo "  # $f"
  grep -E '^\[(START|ABORT|RETRY|FAIL|DONE|GATE)\]' "$f" | tail -4 | sed 's/^/      /'
done

echo "--- ckpt 目录 ---"
for V in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    C="$CKPT_ROOT/$V/$DS"
    [ -d "$C" ] || continue
    echo "  $V/$DS : $(ls -A "$C" 2>/dev/null | wc -l) files  [$(ls -A "$C" 2>/dev/null | tr '\n' ' ')]"
  done
done

echo "--- 正式日志：steps / 有无 TEST 区块 ---"
for V in CTRL_R8 H1_HYP H2_EUC H3_HYP_GS; do
  for DS in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
    F="$LOG_ROOT/$V/$DS/train_log.txt"
    [ -f "$F" ] || continue
    cur=$(grep -E '^\[ACTUAL-OPT-STEPS\] epoch=' "$F" | tail -1 | sed 's/.*cumulative=//')
    tb=$(grep -c '=== END TEST RESULTS ===' "$F")
    echo "  $V/$DS steps=${cur:-0} test_blocks=$tb"
  done
done

echo "--- pipeline.log（本进程自己的行，排除上一轮遗留）---"
tail -8 "$STATE/pipeline.log" 2>/dev/null | sed 's/^/  /'
echo "--- 并行督导是否活着 ---"
if [ -f "$STATE/pipeline_supervisor.pid" ] && kill -0 "$(cat "$STATE/pipeline_supervisor.pid")" 2>/dev/null; then
  echo "  ALIVE pid=$(cat "$STATE/pipeline_supervisor.pid")"
else
  echo "  DEAD / 无 pidfile"
fi
