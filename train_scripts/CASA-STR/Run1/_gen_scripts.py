#!/usr/bin/env python
"""CASA-STR Run1 脚本生成器（主线重构 实验方案 执行顺序）。

网格：6 变体 (attn_mode x rep_mode) x 4 数据集 = 24 个正式训练脚本。
执行顺序（Phase1 -> Phase3，每 Phase 双 GPU 并行队列）：
    Phase1: A0_BASE_PLAIN x4  + M1_CASAA_STR x4   （A0 先锁定 backbone，M1 全创新）
    Phase2: A1_CASAA_PLAIN x4 + A2_STR_ONLY x4
    Phase3: C1_FULLATTN_PLAIN x4 + C2_CONTENT_SAA_PLAIN x4
GPU 分工：GPU0 = CDD -> LEVIR；GPU1 = SYSU -> WHU。

再生成一次即可覆盖所有脚本（幂等）；.sh 全部 LF 结尾。
用法：python _gen_scripts.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]

VARIANTS = {
    "A0_BASE_PLAIN":    dict(attn="none",    rep="plain"),
    "A1_CASAA_PLAIN":   dict(attn="change",  rep="plain"),
    "A2_STR_ONLY":      dict(attn="none",    rep="full"),
    "C1_FULLATTN_PLAIN": dict(attn="full",   rep="plain"),
    "C2_CONTENT_SAA_PLAIN": dict(attn="content", rep="plain"),
    "M1_CASAA_STR":     dict(attn="change",  rep="full"),
}

# Phase1: A0 x4 + M1 x4；Phase2: A1 x4 + A2 x4；Phase3: C1 x4 + C2 x4
PHASES = {
    1: ["A0_BASE_PLAIN", "M1_CASAA_STR"],
    2: ["A1_CASAA_PLAIN", "A2_STR_ONLY"],
    3: ["C1_FULLATTN_PLAIN", "C2_CONTENT_SAA_PLAIN"],
}
GPU_SPLIT = {0: ["CDD-CD-256", "LEVIR-CD-256"], 1: ["SYSU-CD-256", "WHU-CD-256"]}

TRAIN_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 {VARIANT} — {DESC} / {DATASET}（正式 80K，GPU{GPU}；自动 retry 上限 3）
GPU={GPU}
DATASET={DATASET}
VARIANT={VARIANT}
ATTN_MODE={ATTN}
REP_MODE={REP}
MAX_STEPS=80000
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${{GPU}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${{PROJ}}/models
DS_ROOT=/share_datasets/CD/${{DATASET}}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/${{VARIANT}}/${{DATASET}}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-STR/Run1/${{VARIANT}}/${{DATASET}}

mkdir -p "${{CKPT_DIR}}" "${{LOG_DIR}}"
cd "${{MODELS}}"

for attempt in $(seq 1 3); do
    python train.py \\
        --dataset "${{DATASET}}" \\
        --dataset_root "${{DS_ROOT}}" \\
        --train_list "${{DS_ROOT}}/list/train.txt" \\
        --test_list "${{DS_ROOT}}/list/test.txt" \\
        --pretrained_weight_path "${{PROJ}}/pretrained_weight/shvit_s1.pth" \\
        --ckpt_dir "${{CKPT_DIR}}" \\
        --max_steps "${{MAX_STEPS}}" \\
        --batch_size "${{BATCH}}" \\
        --test_batch_size 16 \\
        --inWidth 256 \\
        --inHeight 256 \\
        --model_type tiny \\
        --mode baseline \\
        --arch casa_str \\
        --attn_mode "${{ATTN_MODE}}" \\
        --rep_mode "${{REP_MODE}}" \\
        --backbone_lr_ratio 0.1 \\
        --str_dim 160 \\
        --casaa_keep_ratio 0.25 \\
        --casaa_change_share 0.5 \\
        --lr 2e-4 \\
        --lr_mode poly \\
        --num_workers 4 \\
        --seed 16 \\
        --gpu_id 0 \\
        >> "${{LOG_DIR}}/train_log.txt" 2>&1
    rc=$?
    if [ ${{rc}} -eq 0 ]; then
        echo "[DONE] training+test finished successfully" >> "${{LOG_DIR}}/train_log.txt"
        break
    fi
    echo "[RETRY ${{attempt}}/3] crashed with exit ${{rc}}, resuming from last.pth in 10s" >> "${{LOG_DIR}}/train_log.txt"
    sleep 10
done
"""

QUEUE_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 Phase{PHASE} GPU{GPU} 顺序队列（前一个完成/3 次 retry 耗尽后才启动下一个）
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-STR/Run1
{SEQ}

echo "[PHASE{PHASE}-GPU{GPU}] queue finished"
"""

DRYRUN_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-STR Run1 dry-run：A0_BASE_PLAIN 短训（2 epochs，验证 LR-GROUPS 0.1x 与完整 TEST 链路）
GPU={GPU}
DATASET={DATASET}
VARIANT=_dryrun/A0_BASE_PLAIN
MAX_STEPS={MAX_STEPS}
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${{GPU}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${{PROJ}}/models
DS_ROOT=/share_datasets/CD/${{DATASET}}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-STR/Run1/${{VARIANT}}/${{DATASET}}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-STR/Run1/${{VARIANT}}/${{DATASET}}

mkdir -p "${{CKPT_DIR}}" "${{LOG_DIR}}"
cd "${{MODELS}}"

python train.py \\
    --dataset "${{DATASET}}" \\
    --dataset_root "${{DS_ROOT}}" \\
    --train_list "${{DS_ROOT}}/list/train.txt" \\
    --test_list "${{DS_ROOT}}/list/test.txt" \\
    --pretrained_weight_path "${{PROJ}}/pretrained_weight/shvit_s1.pth" \\
    --ckpt_dir "${{CKPT_DIR}}" \\
    --max_steps "${{MAX_STEPS}}" \\
    --batch_size "${{BATCH}}" \\
    --test_batch_size 16 \\
    --inWidth 256 \\
    --inHeight 256 \\
    --model_type tiny \\
    --mode baseline \\
    --arch casa_str \\
    --attn_mode none \\
    --rep_mode plain \\
    --backbone_lr_ratio 0.1 \\
    --str_dim 160 \\
    --lr 2e-4 \\
    --lr_mode poly \\
    --num_workers 4 \\
    --seed 16 \\
    --gpu_id 0 \\
    >> "${{LOG_DIR}}/train_log.txt" 2>&1
"""

VARIANT_DESC = {
    "A0_BASE_PLAIN": "基线（no attn, plain decoder）",
    "A1_CASAA_PLAIN": "创新一消融（change CASAA, plain）",
    "A2_STR_ONLY": "创新二消融（no attn, full TAR/DCR rep）",
    "C1_FULLATTN_PLAIN": "对照（full attention, plain）",
    "C2_CONTENT_SAA_PLAIN": "对照（content SAA, plain）",
    "M1_CASAA_STR": "主实验（change CASAA + full TAR/DCR rep）",
}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, HERE))


def main():
    # 24 正式训练脚本
    for variant, cfg in VARIANTS.items():
        for ds in DATASETS:
            gpu = 0 if ds in GPU_SPLIT[0] else 1
            script = TRAIN_TEMPLATE.format(
                VARIANT=variant, DESC=VARIANT_DESC[variant],
                DATASET=ds, GPU=gpu, ATTN=cfg["attn"], REP=cfg["rep"])
            write(os.path.join(HERE, variant, f"train_{ds}.sh"), script)

    # Phase 队列（每 GPU 一个顺序队列）
    for phase, variants in PHASES.items():
        for gpu, dss in GPU_SPLIT.items():
            seq = []
            for variant in variants:
                for ds in dss:
                    seq.append(f"bash \"${{RUN_DIR}}/{variant}/train_{ds}.sh\"")
            script = QUEUE_TEMPLATE.format(PHASE=phase, GPU=gpu, SEQ="\n".join(seq))
            write(os.path.join(HERE, f"run_queue_phase{phase}_gpu{gpu}.sh"), script)

    # dry-run（短训 2 epochs 验证 LR-GROUPS/TEST 链路；iters/epoch: CDD≈625）
    for gpu, dss in GPU_SPLIT.items():
        ds = dss[0]  # 每 GPU 用第一个数据集做 dry-run
        steps = 750   # CDD 625 iters/epoch -> 2 epochs；覆盖 iter 0/99/198/199/200/201
        script = DRYRUN_TEMPLATE.format(GPU=gpu, DATASET=ds, MAX_STEPS=steps)
        write(os.path.join(HERE, f"dryrun_a0_gpu{gpu}.sh"), script)


if __name__ == "__main__":
    main()
