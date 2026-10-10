#!/usr/bin/env bash
# Run4 自动收口守候进程：等 8 个 run 全部产出完整 TEST 区块后，自动执行 post_train.sh。
#
# 目的：即使监控会话中断，第 ⑧ 步（协议审计 → 日志硬门 ×8 → 部署图对象指标 ×8 →
# 三层裁决 → README markdown）也一定会在训练结束后被产出，结果不会因为没人看着而丢失。
#
# 安全性：post_train.sh 只读 checkpoint 与日志，自身带 fail-closed 守卫
# （8 个 run 未完成 → exit 3；协议审计不过 → exit 4），因此本守候进程不可能破坏训练或伪造结果。
#
#     nohup bash train_scripts/CASA-TViM/Run4/auto_post_train.sh \
#       > /home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/auto_post_train.log 2>&1 &
set -uo pipefail

LOG_ROOT=${LOG_ROOT:-/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4}
PROJ=/home/yqwang/projects/CASA-CD
DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
INTERVAL=${INTERVAL:-600}          # 10 min
MAX_HOURS=${MAX_HOURS:-36}
MARKER="${LOG_ROOT}/AUTO_POST_TRAIN_DONE"

echo "[$(date +%H:%M:%S)] auto_post_train started (interval=${INTERVAL}s, max=${MAX_HOURS}h)"
t0=$(date +%s)
while :; do
  ready=1
  for v in M1_R4CTRL E6_FET1; do
    for d in ${DATASETS}; do
      L="${LOG_ROOT}/${v}/${d}/train_log.txt"
      if [ ! -f "${L}" ] || ! grep -q '^=== END TEST RESULTS ===' "${L}"; then
        ready=0
      fi
    done
  done
  # 额外条件：没有任何 train.py 在跑。否则最后一个 run 还在收尾（写 last.pth / 跑最终 TEST）时
  # 就开始对象指标 pass 会与其争抢 GPU，且理论上可能读到正在写入的 checkpoint。
  # 注意：`pgrep -c` 无匹配时**本身就会打印 0 并返回退出码 1**，因此不能用 `|| echo 0`——
  # 那会得到 "0\n0" 使 `-ne` 报 integer expression expected（2026-10-10 实测踩到，守候因此永不触发）。
  n_train=$(pgrep -c -f '[t]rain.py --dataset' 2>/dev/null); n_train=${n_train:-0}
  if [ "${n_train}" -ne 0 ]; then
    ready=0
  fi
  if [ "${ready}" -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all 8 runs have a complete TEST block and no trainer is running -> post_train.sh"
    source /home/yqwang/miniforge3/etc/profile.d/conda.sh
    conda activate casacd
    cd "${PROJ}"
    GPU=${GPU:-0} bash train_scripts/CASA-TViM/Run4/post_train.sh
    rc=$?
    echo "[$(date +%H:%M:%S)] post_train.sh exited ${rc}"
    {
      echo "finished_at=$(date -Is)"
      echo "post_train_rc=${rc}"
      echo "verdict=${LOG_ROOT}/verdict.json"
      echo "markdown=${LOG_ROOT}/README_SNIPPET.md"
    } > "${MARKER}"
    exit ${rc}
  fi
  if [ $(( $(date +%s) - t0 )) -gt $(( MAX_HOURS * 3600 )) ]; then
    echo "[$(date +%H:%M:%S)] giving up after ${MAX_HOURS}h (runs did not all finish)"
    exit 9
  fi
  sleep "${INTERVAL}"
done
