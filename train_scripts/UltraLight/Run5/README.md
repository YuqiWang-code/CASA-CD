# Run5 — 成熟预训练 MicroDetail + SGDP（唯一主推，两个 SYSU 80K 决定）

> 决策依据：`docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md`。
> Run4 已按 Stop-3 收束（不再救 Light/PSD detail，不启动旧 SABI/DFPD）。
> Run5：保留已过 gate 的 TinyViT4-192，detail 换成 MobileNetV3-Small features 0-3
> （ImageNet 完整预训练链条，10,488 参数），head 从 3.44M 的旧 FI+decoder 换成
> SGDP（Semantic-Guided Difference Pyramid，115,267 参数）。
> **每个阶段有 gate，前一个不过不进入下一个；不做自动连跑队列。**

## 阶段与 gate（预注册）

1. **R5-D0（无训练，已执行 → FAIL）**：H1/H2/H3 postmortem + MobileDetail raw gate：
   D4 PR-AUC≥0.50、D8 PR-AUC≥0.52、D4 Top32 prec≥0.46、D8 Top32 prec≥0.48。
   **实际：D4 PR 0.4831（差 0.017），其余 3 项过 → [MOBILE-GATE] FAIL →
   按 §12 候选② 永久停止，R5-1/R5-2/R5-3 一律不启动。**
   记录：`docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md`。
2. R5-1 MOBILEDETAIL_OLDHEAD / SYSU 80K：**未启动（gate 前置失败）**。
3. R5-2 MOBILEDETAIL_SGDP / SYSU 80K：**未启动**。
4. R5-3 / LEVIR：**未启动**。

## 下一步（Run6，需重新预注册）

按 §12/§48：Run5 候选② 永久停止后，下一轮**只讨论 ViT semantic 参数预算重分配**
（更浅/更窄 ViT 腾预算、或去掉独立 detail、或换 semantic 源）；先做零训练
raw gate（可复用 `run5_postmortem_and_mobile_audit.py` 的 D 部分模式）再决定 80K。

## 训练协议（沿用）

BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、seed 16、
test-as-val；ViT4 冻结；只用 GPU1。

## 路径

- 权重：`pretrained_weight/mobilenet_v3_small-047dcff4.pth`（ImageNet-1K V1 官方）
- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run5/<VARIANT>/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run5/<VARIANT>/<DATASET>/train_log.txt`

## 启动顺序（GPU1 专属，每步人工读 gate）

```bash
cd /home/yqwang/projects/CASA-CD
# 0) R5-D0 audit（无训练；读 [MOBILE-GATE] 行）
nohup bash train_scripts/UltraLight/Run5/audit_R5_D0_SYSU.sh > /dev/null 2>&1 &
# 1) R5-1 dry run → 80K
bash train_scripts/UltraLight/Run5/dryrun_R5_1_SYSU.sh
nohup bash train_scripts/UltraLight/Run5/train_R5_1_MOBILEDETAIL_OLDHEAD_SYSU.sh > /dev/null 2>&1 &
# 2) R5-2 dry run → 80K（R5-1 PASS 后）
bash train_scripts/UltraLight/Run5/dryrun_R5_2_SYSU.sh
nohup bash train_scripts/UltraLight/Run5/train_R5_2_MOBILEDETAIL_SGDP_SYSU.sh > /dev/null 2>&1 &
# 3) R5-3 LEVIR（R5-2 全过后）
nohup bash train_scripts/UltraLight/Run5/train_R5_3_MOBILEDETAIL_SGDP_LEVIR.sh > /dev/null 2>&1 &
```

代码入口：
- `models/model/mobile_detail.py`（MobileDetail-P3，10,488 参数，exact 继承）
- `models/model/sgdp_head.py`（SGDP，115,267 参数）
- `analyse/run5_postmortem_and_mobile_audit.py`（R5-D0）
- CLI：`--detail_mode mobile_p3 --head_mode legacy|sgdp --mobile_pretrained_weight_path <pth>`
- smoke：`python smoke_test.py --mode all --mobile_pretrained_weight_path <pth>`（T-R5 系列）

## 绝不做（gate 失败时的禁项）

不换 features0-4/0-5、不换 MobileNetV2/EfficientNet/ShuffleNet、不做 Mobile 冻结/小 LR、
不做 SGDP width/gate/attention 变体、不加 loss、不延长 steps。
