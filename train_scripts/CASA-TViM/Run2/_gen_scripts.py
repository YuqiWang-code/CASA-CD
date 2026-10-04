#!/usr/bin/env python
"""CASA-TViM Run2 脚本生成器（LEVIR/SYSU 定向提升：CP-CAACP + FRH）。

网格：3 变体 x 4 数据集 = 12 个正式 80K（batch 32，seed 16，其余协议与 Run1 完全一致）。
变体（Run2 调研文档执行顺序 E1 -> E2 -> E3）：
    E1_CP_CAACP: caacp=1 score=cp frh=0   （首选一：w ~ 1+s*r 置信度保留 pooling，0 新参数）
    E2_FRH:      caacp=1 score=rank frh=1 （首选二：128^2 可折叠细粒度头，deploy +768 params）
    E3_CP_FRH:   caacp=1 score=cp frh=1   （组合，仅当 E1/E2 各自有效）

波次：run_all.sh 一个脚本串完 3 波（不分步）：
    Wave1 E1_CP_CAACP x4   Wave2 E2_FRH x4   Wave3 E3_CP_FRH x4
每波 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发）。

再生成一次即可覆盖所有脚本（幂等）；.sh 全部 LF 结尾。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
GPU_OF = {0: ["CDD-CD-256", "LEVIR-CD-256"], 1: ["SYSU-CD-256", "WHU-CD-256"]}

VARIANTS = {
    "E1_CP_CAACP": dict(caacp=1, score="cp", frh=0),
    "E2_FRH":      dict(caacp=1, score="rank", frh=1),
    "E3_CP_FRH":   dict(caacp=1, score="cp", frh=1),
}
WAVES = [["E1_CP_CAACP"], ["E2_FRH"], ["E3_CP_FRH"]]

TRAIN_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run2 {VARIANT} — caacp={CAACP} score={SCORE} frh={FRH} / {DATASET}（正式 80K，batch 32，GPU{GPU}；自动 retry 上限 3）
GPU={GPU}
DATASET={DATASET}
VARIANT={VARIANT}
CAACP={CAACP}
SCORE={SCORE}
FRH={FRH}
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
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run2/${{VARIANT}}/${{DATASET}}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run2/${{VARIANT}}/${{DATASET}}

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
        --frh "${{FRH}}" \\
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

# CASA-TViM Run2 全训练：3 波 x 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
# 一个脚本串完全部 12 个 80K（不分步）；每波 wait 全部完成后进入下一波。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run2
{WAVES}

echo "[RUN-ALL] all 12 runs finished"
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
        # 任一 job 3 次 retry 后仍失败 -> 整波中止（防止滑入后续波次导致显存过载）。
        # 注意：wait 必须传 $tag（数值 pid）；直接传变量名会被 bash 当作 job 名。
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
                VARIANT=variant, CAACP=cfg["caacp"], SCORE=cfg["score"], FRH=cfg["frh"],
                DATASET=ds, GPU=gpu)
            write(os.path.join(HERE, variant, f"train_{ds}.sh"), script)

    run_all = RUN_ALL_TEMPLATE.format(WAVES="\n\n".join(build_waves(WAVES, 1)))
    write(os.path.join(HERE, "run_all.sh"), run_all)


if __name__ == "__main__":
    main()
