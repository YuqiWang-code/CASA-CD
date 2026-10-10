#!/usr/bin/env bash
set -uo pipefail

# Run4 训练后收口（方案 §8.5 / 第 ⑦⑧ 步）：
#   0) 协议审计（训练未结束时拒绝继续，见下方守卫）；
#   1) 逐 run 用 check_run.py 核对「最后一个完整 TEST 区块 + 硬门」；
#   2) 逐 run 在**部署图**上跑对象指标 pass（4 连通、原生 256²、p>0.5）；
#   3) 出 analyse/run4_fet_report.py 的三层裁决 verdict.json。
# 只读 checkpoint 与日志，不覆盖任何训练产物。
# 退出码：3 = 8 个 run 尚未全部产出完整 TEST 区块；4 = 协议审计未过（除非 FORCE=1）。

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

# ---------------------------------------------------------------------------
# 前置守卫：8 个 run 的日志都必须存在且含**完整的** TEST 区块，否则拒绝继续。
# 目的：防止在训练尚未结束时误跑收口，用半截 checkpoint 生成一个看起来合理的 verdict。
# 确需对部分/未完成的 run 出裁决时，显式 `FORCE=1`（此时层 A 会逐 run 记 INVALID）。
# ---------------------------------------------------------------------------
if [ "${FORCE:-0}" != "1" ]; then
  _missing=""
  for v in M1_R4CTRL E6_FET1; do
    for d in ${DATASETS}; do
      L="${LOG_ROOT}/${v}/${d}/train_log.txt"
      if [ ! -f "${L}" ]; then
        _missing="${_missing} ${v}/${d}=no_log"
      elif ! grep -q '^=== END TEST RESULTS ===' "${L}"; then
        _missing="${_missing} ${v}/${d}=no_complete_TEST_block"
      fi
    done
  done
  if [ -n "${_missing}" ]; then
    echo "[ABORT] post_train.sh refuses to run before all 8 runs have a complete TEST block:"
    for m in ${_missing}; do echo "        - ${m}"; done
    echo "        (set FORCE=1 only to produce a documented partial/INVALID verdict)"
    exit 3
  fi
fi

echo "########## 0) protocol audit (LR plan / step budget / manifest / in-block step marker) ##########"
python train_scripts/CASA-TViM/Run4/audit_protocol.py \
  --log-root "${LOG_ROOT}" --ckpt-root "${CKPT_ROOT}" --run Run4 --require-finished 1 || _audit_fail=1
if [ "${_audit_fail}" -ne 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[ABORT] protocol audit failed; a verdict built on non-compliant logs is not a result."
  echo "        (set FORCE=1 only to produce a documented partial/INVALID verdict)"
  exit 4
fi

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

echo "########## 4) README-ready markdown (rendered from verdict.json, no manual transcription) ##########"
python analyse/run4_fet_report.py --mode markdown --verdict "${LOG_ROOT}/verdict.json" \
  --out "${LOG_ROOT}/README_SNIPPET.md" || echo "[WARN] markdown render failed"

echo "########## SUMMARY ##########"
echo "protocol_audit_fail=${_audit_fail} check_run_fail=${_gate_fail} object_pass_fail=${_obj_fail} verdict_rc=${_verdict_rc}"
python - <<PY
import json
v = json.load(open("${LOG_ROOT}/verdict.json"))
print("STATUS=", v["status"])
print("DECISION=", v["decision"])
PY
exit ${_verdict_rc}
