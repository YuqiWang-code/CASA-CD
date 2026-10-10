#!/usr/bin/env bash
set -uo pipefail

# Run4 训练后收口（方案 §8.5 / 第 ⑦⑧ 步）：
#   1) 逐 run 用 check_run.py 核对「最后一个完整 TEST 区块 + 硬门」；
#   2) 逐 run 在**部署图**上跑对象指标 pass（4 连通、原生 256²、p>0.5）；
#   3) 出 analyse/run4_fet_report.py 的三层裁决 verdict.json。
# 只读 checkpoint 与日志，不覆盖任何训练产物。

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
cd /home/yqwang/projects/CASA-CD
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
LOG_ROOT=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4
CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM
OBJ_DIR=${LOG_ROOT}/objects
mkdir -p "${OBJ_DIR}"

DATASETS="CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256"
_audit_fail=0

echo "########## 0) protocol audit (LR plan / step budget / manifest / in-block step marker) ##########"
python train_scripts/CASA-TViM/Run4/audit_protocol.py \
  --log-root "${LOG_ROOT}" --ckpt-root "${CKPT_ROOT}" --run Run4 --require-finished 1 || _audit_fail=1

echo "########## 1) per-run log gate (check_run.py) ##########"
_gate_fail=0
for v in M1_R4CTRL E6_FET1; do
  for d in ${DATASETS}; do
    echo "===== ${v} / ${d} ====="
    python train_scripts/CASA-TViM/Run4/check_run.py \
      --log "${LOG_ROOT}/${v}/${d}/train_log.txt" --variant "${v}" --dataset "${d}" \
      --ckpt-dir "${CKPT_ROOT}/Run4/${v}/${d}" || _gate_fail=1
  done
done

echo "########## 2) object metrics on the deploy graph ##########"
_obj_fail=0
for v in M1_R4CTRL E6_FET1; do
  for d in ${DATASETS}; do
    python analyse/run4_fet_report.py --mode object --variant "${v}" --dataset "${d}" --run Run4 \
      --device cuda:0 --ckpt-root "${CKPT_ROOT}" --out-dir "${OBJ_DIR}" || _obj_fail=1
  done
done

echo "########## 3) three-tier verdict ##########"
python analyse/run4_fet_report.py --mode verdict \
  --log-root "${LOG_ROOT}" --ckpt-root "${CKPT_ROOT}" \
  --object-dir "${OBJ_DIR}" --out "${LOG_ROOT}/verdict.json"
_verdict_rc=$?

echo "########## SUMMARY ##########"
echo "protocol_audit_fail=${_audit_fail} check_run_fail=${_gate_fail} object_pass_fail=${_obj_fail} verdict_rc=${_verdict_rc}"
python - <<PY
import json
v = json.load(open("${LOG_ROOT}/verdict.json"))
print("STATUS=", v["status"])
print("DECISION=", v["decision"])
PY
exit ${_verdict_rc}
