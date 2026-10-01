# Run8 — B4-SPE 最终路线（Change-Sensitive B4 Semantic Source + Sub-patch Expansion）

> 决策依据：`docs/temporary/CASA-CD_Run8_B4-SPE最终路线与预注册.md`。
> Run7 四数据集 depth gate 已证 B4 是稳健语义源（B2 截断被否）；Run8 只做最后一
> 次最小因果检验：**删除独立 detail 与旧 FI/decoder 后，B4 能否被 ~0.054M 的
> 对称 sub-patch head 独立重建成 256×256**。不再搜新模块；每步预注册 gate。

## 阶段与 gate（预注册，每步人工读 gate）

1. **R8-D0（零训练，四数据集）—— 已执行 → FAIL**：
   M1 像素 PR-AUC / M2 边界带 PR-AUC / M3 分桶 / M4 fixed patch-unembedding。
   结果：C1（G1_pixel）=2/4、**C2（G2_boundary）=1/4**、C3=4/4 → **[R8-D0-GATE] FAIL**。
   关键数字（边界带 lift，B4−B12）：CDD −0.0006、LEVIR +0.0164、SYSU +0.0171、
   WHU +0.0377——B4 的 token 级优势在边界带几乎消失（ViT-CoMer 的 inner-patch
   limitation 实证）。
   **按 §4.5：永久停止 B4-only/no-detail dense reconstruction 路线；不启动
   R8-1/2/3/4；进入论文分析型收尾（R4-1 82.77 为最强已验证轻量化结构，
   Run1-8 为 budget-allocation / negative-evidence study）。**
   记录：`docs/temporary/CASA-CD_Run8_R8-D0结果与B4-only路线终止.md`。
2. R8-1 B4_SPE / SYSU 80K：**未启动**（D0 gate 前置失败）。B4-SPE head 代码
   已实现并通过 T-R8 smoke（2,030,704 参数 / trainable 53,872 / FLOPs 1.2186G /
   严格时间对称），作为被 gate 否决的最终候选存档，不进入训练。
3. R8-2/3/4（LEVIR/WHU/CDD）：**未启动**。

## 训练协议（沿用）

BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、seed 16、
test-as-val；ViT4 全冻结（corrected DeiT loader）；只用 GPU1。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run8/R8_1_B4_SPE/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run8/R8_1_B4_SPE/<DATASET>/train_log.txt`

## 启动顺序（GPU1 专属）

```bash
cd /home/yqwang/projects/CASA-CD
# 0) R8-D0 四数据集 dense recoverability audit（零训练；读 [R8-D0-GATE]）
nohup bash train_scripts/UltraLight/Run8/audit_R8_D0_B4_DENSE.sh > /dev/null 2>&1 &
# 1) R8-1 SYSU（D0 PASS 后）
bash train_scripts/UltraLight/Run8/dryrun_R8_1_SYSU.sh
nohup bash train_scripts/UltraLight/Run8/train_R8_1_B4_SPE_SYSU.sh > /dev/null 2>&1 &
# 2) LEVIR → WHU → CDD（逐 gate）
```

代码入口：
- `models/model/b4_spe_head.py`（B4SPEHead：pair 36,960 + expansion 16,912 = 53,872；
  总 2,030,704）
- `analyse/run8_b4_dense_recoverability_audit.py`（R8-D0）
- CLI：`--vit_depth 4 --detail_mode none_b4 --head_mode b4_spe`
- smoke：`python smoke_test.py --mode run8_b4_spe`（T-R8）

## 绝不做（gate 失败时）

改 D0 阈值、换上采样、加 edge/frequency/detail 模块、adaptive depth 救场、
head width/gate sweep、ResNet/Mobile 回流、改 loss/LR/steps。
