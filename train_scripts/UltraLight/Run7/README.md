# Run7 — CSDP-CD：Change-Sensitive Depth-Pyramid（语义源深度替换）

> 决策依据：`docs/temporary/CASA-CD_Run7_CSDP方案与预注册.md`；
> 文献依据：`docs/参考文献/文献索引.md`（FDAM/LaViT/ResCLIP/ViT-CoMer/LiFT/EoMT 等）。
> Run6 已按 R6-D0 gate 收束。Run7 主推：推翻「最深 ViT feature 最好」，
> 用 DeiT-Tiny 前 2 个 block（B1+B2）作为终止语义源，删除后续深度与独立 detail
> encoder，用 CSDP head（B1+B2 对称差分 + PixelShuffle 金字塔）做全分辨率预测。

## 阶段与 gate（预注册，每步人工读 gate）

1. **R7-D0（零训练，四数据集）—— 已执行 → FAIL**：CDD/LEVIR/SYSU/WHU 完整 test
   的 P0/B1-B4/B12 token change ranking。结果：C1=3/4（CDD 差 0.0016）、
   **C2=2/4（CDD、WHU 未过）**、C3=4/4 → **[R7-D0-GATE] FAIL**。
   按预注册规则：**CSDP-CD 停止，不启动 R7-0/R7-1 任何 80K，不改阈值、
   不改 B2→B3 救**。B2≈B4 只对 SYSU/LEVIR 成立；CDD 需 B4、WHU 需 B3/B4。
   记录：`docs/temporary/CASA-CD_Run7_R7-D0结果与CSDP停止.md`。
2. R7-0 VIT2_OLDHEAD / SYSU 80K：**未启动（D0 gate 前置失败）**。
3. R7-1 CSDP / SYSU 80K：**未启动**（head 代码已实现并通过 smoke，
   1,177,936 参数 / 0.673G / 时间交换对称，留作资产）。
4. R7-1 CSDP / LEVIR、CDD、WHU：**未启动**。

## 下一步（供重新预注册）

四数据集 depth 曲线结论：B4 是稳健最优（CDD/SYSU 第一、LEVIR/WHU 第二）、
full12 全面劣化、B2 只在 SYSU/LEVIR 够用 →「optimal semantic depth 是
dataset-dependent」。候选：保持 B4 语义源轻量化 head / depth-adaptive 多源聚合 /
以 Run4 已验证组合收尾，待决策。

## 训练协议（沿用）

BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、seed 16、
test-as-val；ViT 一律冻结（corrected DeiT loader）；只用 GPU1。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run7/<VARIANT>/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run7/<VARIANT>/<DATASET>/train_log.txt`

## 启动顺序（GPU1 专属）

```bash
cd /home/yqwang/projects/CASA-CD
# 0) R7-D0 四数据集审计（零训练；读 [R7-D0-GATE]）
nohup bash train_scripts/UltraLight/Run7/audit_R7_D0_DEPTH_4DS.sh > /dev/null 2>&1 &
# 1) R7-0（D0 PASS 后）
bash train_scripts/UltraLight/Run7/dryrun_R7_0_SYSU.sh
nohup bash train_scripts/UltraLight/Run7/train_R7_0_VIT2_OLDHEAD_SYSU.sh > /dev/null 2>&1 &
# 2) R7-1 SYSU（R7-0 PASS 后）
bash train_scripts/UltraLight/Run7/dryrun_R7_1_SYSU.sh
nohup bash train_scripts/UltraLight/Run7/train_R7_1_CSDP_SYSU.sh > /dev/null 2>&1 &
# 3) LEVIR → CDD/WHU（逐 gate）
```

代码入口：
- `models/model/depth_pyramid_head.py`（CSDP head，90,832 参数）
- `analyse/run7_depth_source_audit.py`（R7-D0）
- CLI：`--vit_depth 2 --detail_mode depth_pyramid --head_mode csdp`
- smoke：`python smoke_test.py --mode run7_csdp`（T-R7）

## 绝不做（gate 失败时）

改阈值、B2→B3 救、回 detail branch、解冻 ViT、head width/gate 调参、
加 loss、延长 steps、多 seed。
