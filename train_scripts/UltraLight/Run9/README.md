# Run9 — B4-OPRE（Frozen B4 Semantic + Overlap Patch Re-Embedding）

> 决策依据：`docs/temporary/CASA-CD_Run9_可执行预注册方案.md`。
> 新硬目标：SYSU≥85 / LEVIR≥92.5 / WHU≥95 / CDD≥98（IoU 同方向换算），<3M 参数、
> ≤2.0G FLOPs、创新性/故事性/可解释性/轻量化。
> Run9 主方案：保留 B4 语义源，**零新增 encoder 参数**地复用冻结 DeiT patch kernel
> 做 stride=8 重叠重嵌入（O-PRE 192×32×32 局部 evidence），由 B4 语义门控注入
> 极轻量 PixelShuffle head（总 ≈2.037M / trainable ≈60K）。

## 阶段与 gate（预注册，逐 gate 启动）

1. **R9-D0（零训练，四数据集）—— 已执行 → FAIL**：
   P0/B4/O-PRE/proxy 的像素与边界带 PR-AUC。结果：**G0=0/4、G1=0/4、
   G2a=3/4、G2b=1/4、G3=False → [R9-D0-GATE] FAIL**。
   关键数字（边界带 lift）：O-PRE vs P0：CDD +0.0154、LEVIR **−0.0094**、
   SYSU +0.0042、WHU **−0.0160**；proxy vs B4：CDD +0.0171、LEVIR **−0.0154**、
   SYSU +0.0038、WHU **−0.0361**——采样 lattice 变密不带来边界互补，
   H9 四数据集一致证伪。
   **按 §8.5/Stop-0：Run9 停止，0 个 80K；不调 stride/padding/score 救场。**
   记录：`docs/temporary/CASA-CD_Run9_R9-D0结果与B4-OPRE路线终止.md`。
2. R9-1 B4_OPRE / SYSU 80K：**未启动**（D0 gate 前置失败）。OPREHead 代码
   已实现并通过 T-R9 smoke（2,036,945 参数 / trainable 60,113 / FLOPs 1.5270G /
   对称 0.00 / 零新增 encoder 参数），作为被 gate 否决的候选存档。
3. R9-2/3/4 与 R9-A1：**未启动**。

## 训练协议（沿用）

BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、seed 16、
test-as-val；ViT4 全冻结（含共享 O-PRE kernel）；只用 GPU1；**自动 retry 上限 3 次**。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run9/<VARIANT>/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run9/<VARIANT>/<DATASET>/train_log.txt`

## 启动顺序（GPU1 专属）

```bash
cd /home/yqwang/projects/CASA-CD
# 0) R9-D0（零训练；读 [R9-D0-GATE]）
nohup bash train_scripts/UltraLight/Run9/audit_R9_D0_OPRE.sh > /dev/null 2>&1 &
# 1) R9-1 SYSU（D0 PASS 后）
bash train_scripts/UltraLight/Run9/dryrun_R9_1_SYSU.sh
nohup bash train_scripts/UltraLight/Run9/train_R9_1_B4_OPRE_SYSU.sh > /dev/null 2>&1 &
# 2) LEVIR → WHU → CDD（逐 gate）→ 全过后 R9_A1_NOGATE
```

代码入口：
- `models/model/opre_head.py`（OPREHead：pair 36,960 + 语义 expansion + O-PRE
  local_proj 6,208 + gate 33 + 重建 = 60,113；总 2,036,945）
- `analyse/run9_overlap_reembedding_audit.py`（R9-D0）
- CLI：`--vit_depth 4 --detail_mode opre --head_mode opre_spe [--opre_gate 1|0]`
- smoke：`python smoke_test.py --mode run9_opre`（T-R9-0..9）

## 绝不做（gate 失败时）

改 stride/padding、换 score 重刷、加 edge/frequency/detail 模块、解冻 patch embed、
改 loss/LR/steps、换 seed、多 phase ensemble（§8.6 / §12 Stop-0..4）。
