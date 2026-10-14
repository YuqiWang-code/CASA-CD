#!/usr/bin/env bash
# Run8 启动前最后一道防呆：用**真实 train.py + 真实数据 + 真实日志格式**跑 3 步，
# 只验证「脚本能不能把 train.py 正确拉起来并进到训练循环」。
#
# 为什么需要它（2026-10-15 两次真实翻车）：
#   ① `run_all.sh` 由 `setsid nohup` 拉起 ⇒ 不继承 conda 激活 ⇒ `python: command not found`（rc=127）；
#   ② `train.py` 的 `--arch` choices 忘了加 `casa_vmamba_str` ⇒ argparse 直接 rc=2。
# 这两类错误都只在**四个库各失败 3 次之后**才被发现，白烧掉十几分钟与一次波次尝试。
# 本脚本把它压缩成一次 30 秒的检查。
#
# 只读保证：ckpt 写到 `/tmp/run8_preflight_<ds>`，用 `--max_steps 3`；
# `test_best()` 在找不到 `best_F1=*.pth` 时会直接跳过（不会跑完整 test 集）。
# 不触碰 outputs/ 与 checkpoints/ 下任何正式路径。
set -u
CONDA_SH=/home/yqwang/miniforge3/etc/profile.d/conda.sh
REPO=/home/yqwang/projects/CASA-CD
CA=/home/yqwang/miniforge3/envs/casacd
MB=/home/yqwang/miniforge3/envs/mamba_base
DATA=/share_datasets/CD
PRET=$REPO/pretrained_weight

source "$CONDA_SH" && conda activate casacd
export PATH="$CA/bin:${PATH}"
export PYTHONPATH="$MB/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CA/lib/python3.10/site-packages/torch/lib:$CA/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS=${1:-CDD-CD-256}
DS_ROOT="$DATA/$DS"
CKPT="/tmp/run8_preflight_$DS"
rm -rf "$CKPT" 2>/dev/null
mkdir -p "$CKPT"

echo "########## Run8 preflight $(date '+%F %T')  ds=$DS ##########"
cd "$REPO/models" || exit 1
CUDA_VISIBLE_DEVICES=0 timeout 900 python train.py \
  --dataset "$DS" \
  --dataset_root "$DS_ROOT" \
  --train_list "$DS_ROOT/list/train.txt" \
  --test_list "$DS_ROOT/list/test.txt" \
  --pretrained_weight_path "$PRET/deit_tiny_patch16_224-a1311bcf.pth" \
  --vmamba_pretrained_weight_path "$PRET/vssm_tiny_0230_ckpt_epoch_262.pth" \
  --ckpt_dir "$CKPT" \
  --max_steps 3 \
  --exact_max_steps 1 \
  --batch_size 32 \
  --test_batch_size 16 \
  --inWidth 256 --inHeight 256 \
  --model_type tiny --mode baseline --arch casa_vmamba_str \
  --hqkv 1 --hqkv_geometry hyperbolic --hqkv_branch both \
  --hqkv_inject_mode dcr1 \
  --hqkv_rank 32 --hqkv_lite_dim 192 \
  --hqkv_detail_window 2 --hqkv_global_window 4 \
  --hqkv_attn_stages 1,2,3 \
  --hqkv_r0 0.5 --hqkv_a 0.5 \
  --hgsl 1 --hgsl_detach_score 1 --hgsl_lambda 0.2 \
  --fs_tar 0 \
  --rep_mode full --backbone_lr_ratio 0.1 --str_dim 192 \
  --lr 2e-4 --lr_mode poly --num_workers 4 --seed 16 --gpu_id 0 \
  2>&1 | tee /tmp/run8_preflight_$DS.log | grep -E \
  '^\['"'"'(CONFIG|HQKV|PARAM-BREAKDOWN|PRETRAIN-LOAD|MANIFEST|LR-GROUPS|MAX-STEPS|EXACT-STEPS)'"'"'\]|^\[ACTUAL-OPT-STEPS\]|Epoch |Error|error|Traceback|FAIL' | head -30
RC=${PIPESTATUS[0]}
echo "########## preflight rc=$RC (0=PASS) ##########"
echo "--- 关键断言 ---"
L="/tmp/run8_preflight_$DS.log"
for k in "worst_diff=0.0" "gamma_zero_init=True" "ACTUAL-OPT-STEPS" "PROTOCOL-VERSION run8_vmamba_hqkv_exact80k_v1"; do
  if grep -q "$k" "$L"; then echo "  [OK]   $k"; else echo "  [MISS] $k"; fi
done
exit $RC
