#!/usr/bin/env python
"""CASA-TViM Run4 脚本生成器（R4-FET1：1/4 细尺度证据 1×1 可折叠旁路）。

方案来源：docs/temporary/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md §5/§6/§8。

变体（唯一结构变量 = fine_tap）：
    M1_R4CTRL : fine_tap=0  —— 同期唯一变量对照（必须与 E6 同协议、同 80K 严格步数）
    E6_FET1   : fine_tap=1  —— 唯一正式主实验（+13,921 train / +13,920 deploy 参数）

波次：先 M1_R4CTRL 四库全部跑完，再 E6_FET1 四库；每波 GPU0=CDD+LEVIR、GPU1=SYSU+WHU。
两个变体共用 `--exact_max_steps 1`（poly 分母=80000，精确 80,000 optimizer updates）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
GPU_OF = {0: ["CDD-CD-256", "LEVIR-CD-256"], 1: ["SYSU-CD-256", "WHU-CD-256"]}

VARIANTS = {
    "M1_R4CTRL": dict(fine_tap=0),
    "E6_FET1":   dict(fine_tap=1),
}
WAVES = [["M1_R4CTRL"], ["E6_FET1"]]

TRAIN_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

# CASA-TViM Run4 {VARIANT} — fine_tap={FINE_TAP} / {DATASET}（正式 80,000 optimizer updates，
# batch 32，seed 16，exact_max_steps=1，GPU{GPU}；同 run 断点恢复 retry 上限 3）
GPU={GPU}
DATASET={DATASET}
VARIANT={VARIANT}
FINE_TAP={FINE_TAP}
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
CKPT_DIR=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/${{VARIANT}}/${{DATASET}}
LOG_DIR=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/${{VARIANT}}/${{DATASET}}

# ---------------------------------------------------------------------------
# 启动前置：正式路径不得已有内容（防止 train.py 的 auto-resume 意外继承旧权重）。
# 只有显式 RESUME_SAME_RUN=1 且 last.pth 存在（经 T11 验证的同 run 断点）才允许续跑。
# ---------------------------------------------------------------------------
for d in "${{CKPT_DIR}}" "${{LOG_DIR}}"; do
    mkdir -p "$d"
done
if [ -f "${{CKPT_DIR}}/last.pth" ]; then
    if [ "${{RESUME_SAME_RUN:-0}}" != "1" ]; then
        echo "[ABORT] ${{CKPT_DIR}}/last.pth already exists; set RESUME_SAME_RUN=1 only for a certified same-run resume" >&2
        exit 3
    fi
    echo "[RESUME] RESUME_SAME_RUN=1: continuing the same run from last.pth (manifest/arch/code SHA must match)"
fi
if [ "${{RESUME_SAME_RUN:-0}}" != "1" ] && [ -n "$(ls -A "${{CKPT_DIR}}" 2>/dev/null)" ]; then
    echo "[ABORT] ${{CKPT_DIR}} is not empty (no certified resume requested); refusing to start a new experiment here" >&2
    ls -la "${{CKPT_DIR}}" >&2
    exit 3
fi

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
        --exact_max_steps 1 \\
        --batch_size "${{BATCH}}" \\
        --test_batch_size 16 \\
        --inWidth 256 \\
        --inHeight 256 \\
        --model_type tiny \\
        --mode baseline \\
        --arch casa_tvim_str \\
        --caacp 1 \\
        --caacp_score_mode rank \\
        --caacp_residual_mode current \\
        --frh 0 \\
        --fs_tar 0 \\
        --fine_tap "${{FINE_TAP}}" \\
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
        echo "[DONE] training loop exited 0 (attempt ${{attempt}})" >> "${{LOG_DIR}}/train_log.txt"
        break
    fi
    if [ ${{attempt}} -eq 3 ]; then
        echo "[FAIL] 3 attempts exhausted (last rc=${{rc}}); leaving ckpt/log untouched" >> "${{LOG_DIR}}/train_log.txt"
        exit 1
    fi
    echo "[RETRY ${{attempt}}/3] crashed with exit ${{rc}}, resuming from last.pth in 10s" >> "${{LOG_DIR}}/train_log.txt"
    sleep 10
done

# ---------------------------------------------------------------------------
# 退出码由「日志最后一个完整 TEST 区块 + 硬门」决定，不看 python 进程返回码本身。
# ---------------------------------------------------------------------------
python "${{PROJ}}/train_scripts/CASA-TViM/Run4/check_run.py" \\
    --log "${{LOG_DIR}}/train_log.txt" --variant "${{VARIANT}}" --dataset "${{DATASET}}" \\
    --ckpt-dir "${{CKPT_DIR}}"
"""

RUN_ALL_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail

# CASA-TViM Run4 全训练：2 波 x 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发、batch 32）。
# 先 M1_R4CTRL（同期对照）四库全部结束，再 E6_FET1（唯一正式主实验）四库；某波失败立即中止，不启动下一波。
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd

RUN_DIR=/home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run4
PROJ=/home/yqwang/projects/CASA-CD

# --- 预检 1：所有正式 ckpt/log 路径必须不存在或严格空目录（除非 RESUME_SAME_RUN=1） ---
if [ "${{RESUME_SAME_RUN:-0}}" != "1" ]; then
    for v in M1_R4CTRL E6_FET1; do
        for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
            ck=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run4/${{v}}/${{d}}
            lg=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4/${{v}}/${{d}}
            if [ -d "${{ck}}" ] && [ -n "$(ls -A "${{ck}}" 2>/dev/null)" ]; then
                echo "[ABORT] non-empty ckpt dir: ${{ck}}"; exit 3
            fi
            if [ -d "${{lg}}" ] && [ -n "$(ls -A "${{lg}}" 2>/dev/null)" ]; then
                echo "[ABORT] non-empty log dir: ${{lg}}"; exit 3
            fi
        done
    done
else
    echo "[RESUME_SAME_RUN=1] skipping the empty-dir precheck (certified same-run resume only)"
fi

# --- 预检 2：每 GPU 并发 2 个 batch32 进程的显存 probe（R8）。任一 OOM 停止并发，不改 batch/协议 ---
echo "[PREFLIGHT] batch32 x2 per GPU concurrency memory probe"
_probe_fail=0
for g in 0 1; do
    for k in 1 2; do
        CUDA_VISIBLE_DEVICES=${{g}} python "${{RUN_DIR}}/gpu_concurrency_probe.py" \\
            --variant E6_FET1 --device cuda:0 --batch-size 32 --steps 2 \\
            > /tmp/run4_probe_g${{g}}_${{k}}.log 2>&1 & _p=$!
        eval "PID_${{g}}_${{k}}=$_p"
    done
done
for g in 0 1; do
    for k in 1 2; do
        eval "p=\\$PID_${{g}}_${{k}}"
        wait "$p" || _probe_fail=1
    done
done
if [ "${{_probe_fail}}" -ne 0 ]; then
    echo "[ABORT] batch32 x2/GPU probe failed (likely OOM). Keep batch32; switch to same-GPU serial + cross-GPU parallel; do NOT change the training protocol."
    for f in /tmp/run4_probe_g*.log; do echo "--- $f"; tail -5 "$f"; done
    exit 4
fi
echo "[PREFLIGHT] concurrency probe OK"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

{WAVES}

echo "[RUN-ALL] all 8 runs finished"
"""


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.chmod(path, mode)
    print("wrote", os.path.relpath(path, HERE))


def build_waves(waves_list, first_wave=1):
    waves = []
    for wi, wave in enumerate(waves_list):
        i = first_wave + wi
        lines = [f'echo "===== WAVE {i} ====="']
        pids = []
        for j, variant in enumerate(wave):
            for k, ds in enumerate(DATASETS):
                tag = f"P{j}_{k}"
                pids.append(tag)
                lines.append(f'bash "${{RUN_DIR}}/{variant}/train_{ds}.sh" > /tmp/wave{i}_{tag}.log 2>&1 & {tag}=$!')
        lines.append("_wave_fail=0")
        for tag in pids:
            lines.append(f"wait ${tag} || _wave_fail=1")
        lines.append(f'if [ "$_wave_fail" -ne 0 ]; then echo "[WAVE-ABORT] wave {i} has failed runs; aborting"; exit 1; fi')
        lines.append(f'echo "===== WAVE {i} done ====="')
        waves.append("\n".join(lines))
    return waves


def main():
    for variant, cfg in VARIANTS.items():
        for ds in DATASETS:
            gpu = 0 if ds in GPU_OF[0] else 1
            script = TRAIN_TEMPLATE.format(
                VARIANT=variant, FINE_TAP=cfg["fine_tap"], DATASET=ds, GPU=gpu)
            write(os.path.join(HERE, variant, f"train_{ds}.sh"), script, mode=0o755)

    run_all = RUN_ALL_TEMPLATE.format(WAVES="\n\n".join(build_waves(WAVES, 1)))
    write(os.path.join(HERE, "run_all.sh"), run_all, mode=0o755)


if __name__ == "__main__":
    main()
