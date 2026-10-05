#!/usr/bin/env python
"""CASA-TViM Run3 脚本生成器（SYSU 小目标瓶颈定向：RA-CAACP + FS-TAR）。

按 Run2 复盘文档（docs/temporary/CASA-CD_Run2复盘_SYSU小目标瓶颈与下一轮结构实验设计_2026-10-05.md）
第一批：2 变体 x 4 数据集 = 8 个正式 80K（batch 32，seed 16，协议与 Run1/Run2 逐项一致）。
变体（单变量，各改一个创新）：
    E4_RA_CAACP: caacp=1 rank current->avg_anchor  （创新一升级：residual 锚定 c_avg，0 参数）
    E5_FS_TAR:   caacp=1 rank fs_tar=1             （创新二升级：stage1 TemporalRepFine3x3，+73,728 deploy）

波次：run_all.sh 一个脚本串完 2 波（不分步）：
    Wave1 E4_RA_CAACP x4   Wave2 E5_FS_TAR x4
每波 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
GPU_OF = {0: ["CDD-CD-256", "LEVIR-CD-256"], 1: ["SYSU-CD-256", "WHU-CD-256"]}

VARIANTS = {
    "E4_RA_CAACP": dict(caacp=1, score="rank", residual="avg_anchor", frh=0, fs_tar=0),
    "E5_FS_TAR":   dict(caacp=1, score="rank", residual="current", frh=0, fs_tar=1),
}
WAVES = [["E4_RA_CAACP"], ["E5_FS_TAR"]]

TRAIN_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run3 {VARIANT} — caacp={CAACP} score={SCORE} residual={RESIDUAL} fs_tar={FS_TAR} / {DATASET}（正式 80K，batch 32，GPU{GPU}；自动 retry 上限 3）
GPU={GPU}
DATASET={DATASET}
VARIANT={VARIANT}
CAACP={CAACP}
SCORE={SCORE}
RESIDUAL={RESIDUAL}
FS_TAR={FS_TAR}
MAX_STEPS=80000
BATCH=32

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

export CUDA_VISIBLE_DEVICES=${{GPU}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# selective_scan_cuda_oflex(.so) 链接 torch 动态库所需（自建 kernel 无内嵌 rpath）
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${{LD_LIBRARY_PATH:-}}

PROJ=/home/yqwang/projects/CASA-CD
MODELS=${{PROJ}}/models
DS_ROOT=/share_datasets/CD/${{DATASET}}
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run3/${{VARIANT}}/${{DATASET}}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run3/${{VARIANT}}/${{DATASET}}

mkdir -p "${{CKPT_DIR}}" "${{LOG_DIR}}"
cd "${{MODELS}}"

for attempt in $(seq 1 3); do
    python train.py \\
        --dataset "${{DATASET}}" \\
        --dataset_root "${{DS_ROOT}}" \\
        --train_list "${{DS_ROOT}}/list/train.txt" \\
        --test_list "${{DS_ROOT}}/list/test.txt" \\
        --pretrained_weight_path "${{PROJ}}/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth" \\
        --tinyvim_pretrained_weight_path "${{PROJ}}/pretrained_weight/tinyvim_s_1000e.pth" \\
        --ckpt_dir "${{CKPT_DIR}}" \\
        --max_steps "${{MAX_STEPS}}" \\
        --batch_size "${{BATCH}}" \\
        --test_batch_size 16 \\
        --inWidth 256 \\
        --inHeight 256 \\
        --model_type tiny \\
        --mode baseline \\
        --arch casa_tvim_str \\
        --caacp "${{CAACP}}" \\
        --caacp_score_mode "${{SCORE}}" \\
        --caacp_residual_mode "${{RESIDUAL}}" \\
        --frh 0 \\
        --fs_tar "${{FS_TAR}}" \\
        --rep_mode full \\
        --backbone_lr_ratio 0.1 \\
        --str_dim 96 \\
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

RUN_ALL_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run3 全训练：2 波 x 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
# 一个脚本串完全部 8 个 80K（不分步）；每波 wait 全部完成后进入下一波。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run3
{WAVES}

echo "[RUN-ALL] all 8 runs finished"
"""


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, HERE))


def build_waves(waves_list, first_wave=1):
    waves = []
    for wi, wave in enumerate(waves_list):
        i = first_wave + wi
        lines = [f"echo \"===== WAVE {i} =====\""]
        pids = []
        for j, variant in enumerate(wave):
            for k, ds in enumerate(DATASETS):
                tag = f"P{j}_{k}"
                pids.append(tag)
                lines.append(f"bash \"${{RUN_DIR}}/{variant}/train_{ds}.sh\" > /tmp/wave{i}_{tag}.log 2>&1 & {tag}=$!")
        lines.append("_wave_fail=0")
        for tag in pids:
            lines.append(f"wait ${tag} || _wave_fail=1")
        lines.append(f"if [ \"$_wave_fail\" -ne 0 ]; then echo \"[WAVE-ABORT] wave {i} has failed runs; aborting\"; exit 1; fi")
        lines.append(f"echo \"===== WAVE {i} done =====\"")
        waves.append("\n".join(lines))
    return waves


def main():
    for variant, cfg in VARIANTS.items():
        for ds in DATASETS:
            gpu = 0 if ds in GPU_OF[0] else 1
            script = TRAIN_TEMPLATE.format(
                VARIANT=variant, CAACP=cfg["caacp"], SCORE=cfg["score"],
                RESIDUAL=cfg["residual"], FS_TAR=cfg["fs_tar"],
                DATASET=ds, GPU=gpu)
            write(os.path.join(HERE, variant, f"train_{ds}.sh"), script)

    run_all = RUN_ALL_TEMPLATE.format(WAVES="\n\n".join(build_waves(WAVES, 1)))
    write(os.path.join(HERE, "run_all.sh"), run_all)


if __name__ == "__main__":
    main()
