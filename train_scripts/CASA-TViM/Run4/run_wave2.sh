#!/usr/bin/env bash
set -uo pipefail

# Run4 第 2 波（E6_FET1）单独启动器。
# 用它的原因：第 1 波已跑完且目录非空，`run_all.sh` 的「全 16 路径必须为空」前置检查会拒绝启动；
# 本脚本只做与 run_all.sh 相同的波次编排（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发），
# 并且**先核对第 1 波 CTRL 的门禁结论**再启动。

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
cd /home/yqwang/projects/CASA-CD

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run4
LOG=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4

echo "########## 前置：第 1 波 CTRL 门禁逐 run 复核 ##########"
for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  python "${RUN_DIR}/check_run.py" \
    --log "${LOG}/M1_R4CTRL/${d}/train_log.txt" --variant M1_R4CTRL --dataset "${d}" \
    --ckpt-dir "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/M1_R4CTRL/${d}" \
    | tail -3
  rc=${PIPESTATUS[0]}
  echo "  >>> M1_R4CTRL/${d} gate_rc=${rc}"
  if [ "${rc}" -ne 0 ]; then echo "[ABORT] CTRL gate failed for ${d}; refusing to start wave 2"; exit 3; fi
done

echo "########## 前置：E6 目录必须为空 ##########"
for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
  C=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/E6_FET1/${d}
  if [ -d "${C}" ] && [ -n "$(ls -A "${C}" 2>/dev/null)" ]; then
    echo "[ABORT] non-empty E6 ckpt dir: ${C}"; exit 3
  fi
done
echo "OK"

echo "########## 并发显存 probe（batch32 x2/GPU） ##########"
_probe_fail=0
for g in 0 1; do
  for k in 1 2; do
    CUDA_VISIBLE_DEVICES=${g} python "${RUN_DIR}/gpu_concurrency_probe.py" \
      --variant E6_FET1 --device cuda:0 --batch-size 32 --steps 2 \
      > /tmp/run4_e6_probe_g${g}_${k}.log 2>&1 & _p=$!
    eval "EPID_${g}_${k}=${_p}"
  done
done
for g in 0 1; do for k in 1 2; do
  eval "p=\$EPID_${g}_${k}"; wait "$p" || _probe_fail=1
done; done
if [ "${_probe_fail}" -ne 0 ]; then
  echo "[ABORT] batch32 x2/GPU probe failed"; for f in /tmp/run4_e6_probe_g*.log; do tail -3 "$f"; done
  exit 4
fi
echo "probe OK"

echo "########## WAVE 2 (E6_FET1) ##########"
bash "${RUN_DIR}/E6_FET1/train_CDD-CD-256.sh"   > /tmp/wave2_E6_CDD.log   2>&1 & P0=$!
bash "${RUN_DIR}/E6_FET1/train_LEVIR-CD-256.sh" > /tmp/wave2_E6_LEVIR.log 2>&1 & P1=$!
bash "${RUN_DIR}/E6_FET1/train_SYSU-CD-256.sh"  > /tmp/wave2_E6_SYSU.log  2>&1 & P2=$!
bash "${RUN_DIR}/E6_FET1/train_WHU-CD-256.sh"   > /tmp/wave2_E6_WHU.log   2>&1 & P3=$!
_fail=0
wait $P0 || _fail=1
wait $P1 || _fail=1
wait $P2 || _fail=1
wait $P3 || _fail=1
if [ "${_fail}" -ne 0 ]; then echo "[WAVE-ABORT] wave 2 has failed runs"; exit 1; fi
echo "===== WAVE 2 done ====="
echo "[RUN-WAVE2] all 4 E6 runs finished"
