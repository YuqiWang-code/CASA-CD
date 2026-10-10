#!/usr/bin/env bash
set -euo pipefail

# CASA-TViM Run4 E6_FET1 — fine_tap=1 / SYSU-CD-256（正式 80,000 optimizer updates，
# batch 32，seed 16，exact_max_steps=1，GPU1；同 run 断点恢复 retry 上限 3）
GPU=1
DATASET=SYSU-CD-256
VARIANT=E6_FET1
FINE_TAP=1
MAX_STEPS=80000
BATCH=32

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# selective_scan_cuda_oflex(.so) 链接 torch 动态库所需（自建 kernel 无内嵌 rpath）
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/${VARIANT}/${DATASET}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/${VARIANT}/${DATASET}

# ---------------------------------------------------------------------------
# 启动前置：正式路径不得已有内容（防止 train.py 的 auto-resume 意外继承旧权重）。
# 只有显式 RESUME_SAME_RUN=1 且 last.pth 存在（经 T11 验证的同 run 断点）才允许续跑。
# ---------------------------------------------------------------------------
for d in "${CKPT_DIR}" "${LOG_DIR}"; do
    mkdir -p "$d"
done
if [ -f "${CKPT_DIR}/last.pth" ]; then
    if [ "${RESUME_SAME_RUN:-0}" != "1" ]; then
        echo "[ABORT] ${CKPT_DIR}/last.pth already exists; set RESUME_SAME_RUN=1 only for a certified same-run resume" >&2
        exit 3
    fi
    echo "[RESUME] RESUME_SAME_RUN=1: continuing the same run from last.pth (manifest/arch/code SHA must match)"
fi
if [ "${RESUME_SAME_RUN:-0}" != "1" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    echo "[ABORT] ${CKPT_DIR} is not empty (no certified resume requested); refusing to start a new experiment here" >&2
    ls -la "${CKPT_DIR}" >&2
    exit 3
fi

cd "${MODELS}"

for attempt in $(seq 1 3); do
    # set -e 下必须先关掉 errexit，否则训练失败会直接终止脚本、拿不到 rc 做重试判定
    set +e
    python train.py \
        --dataset "${DATASET}" \
        --dataset_root "${DS_ROOT}" \
        --train_list "${DS_ROOT}/list/train.txt" \
        --test_list "${DS_ROOT}/list/test.txt" \
        --pretrained_weight_path "${PROJ}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \
        --tinyvim_pretrained_weight_path "${PROJ}/pretrained_weight/tinyvim_s_1000e.pth" \
        --ckpt_dir "${CKPT_DIR}" \
        --max_steps "${MAX_STEPS}" \
        --exact_max_steps 1 \
        --batch_size "${BATCH}" \
        --test_batch_size 16 \
        --inWidth 256 \
        --inHeight 256 \
        --model_type tiny \
        --mode baseline \
        --arch casa_tvim_str \
        --caacp 1 \
        --caacp_score_mode rank \
        --caacp_residual_mode current \
        --frh 0 \
        --fs_tar 0 \
        --fine_tap "${FINE_TAP}" \
        --rep_mode full \
        --backbone_lr_ratio 0.1 \
        --str_dim 96 \
        --lr 2e-4 \
        --lr_mode poly \
        --num_workers 4 \
        --seed 16 \
        --gpu_id 0 \
        >> "${LOG_DIR}/train_log.txt" 2>&1
    rc=$?
    set -e
    if [ ${rc} -eq 0 ]; then
        echo "[DONE] training loop exited 0 (attempt ${attempt})" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    if [ ${attempt} -eq 3 ]; then
        echo "[FAIL] 3 attempts exhausted (last rc=${rc}); leaving ckpt/log untouched" >> "${LOG_DIR}/train_log.txt"
        exit 1
    fi
    echo "[RETRY ${attempt}/3] crashed with exit ${rc}, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done

# ---------------------------------------------------------------------------
# 退出码由「日志最后一个完整 TEST 区块 + 硬门」决定，不看 python 进程返回码本身。
# ---------------------------------------------------------------------------
python "${PROJ}/train_scripts/CASA-TViM/Run4/check_run.py" \
    --log "${LOG_DIR}/train_log.txt" --variant "${VARIANT}" --dataset "${DATASET}" \
    --ckpt-dir "${CKPT_DIR}"
