# CASAA Run3 — A4 Detail-guided 可部署 v2（先证据、后 80K）

> 决策依据：`docs/temporary/CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md`。
> **顺序铁律：Audit Gate 不通过就不启动 80K；SYSU 通过后才跑 LEVIR；无自动连跑队列。**

## 方法（A4, router=detail_fused）

- change score：`s = 0.5·rank(s_v) + 0.5·rank(s_d)`
  - `s_v` = ViT late-stage 双时相 cosine（同 Run1 A2）；
  - `s_d` = **detail branch 1/8 特征（resnet.layer3 输出 32×32×256）**
    → AvgPool2d(2) → 16×16（与 ViT token 同网格）→ `1−cos(d̄1,d̄2)`；
  - 两者各自 per-image ordinal rank 归一化到 [0,1] 后 1:1 相加（无融合超参）。
- 其余与 A2/A3 完全一致：blocks 8-11、N=256、K=64、Kc=32、Kb=32、
  keep_ratio 0.25、change_share 0.50、共享确定性 density-peak 背景聚类、
  qkv/proj 预训练原位继承、**零新增参数/零新 loss**、冻结 ViT（`--freeze_vit 1`）、
  BCE+Dice、poly、80000 steps、batch 16、256×256、seed 16、test-as-val。
- 实现要点：detail branch 只前向一次（特征复用于 decoder）；routing 全程 no-grad
  （detail score 不反传，但 detail branch 仍经原 decoder 路径正常训练）。

## 已有对照（Run2，不重跑）

| ID | Router | LEVIR F1 | SYSU F1 |
|---|---|---:|---:|
| A1 | content-only（冻结 ViT） | 91.84 | 82.04 |
| A3 | Oracle GT（诊断上界） | 91.88 | 83.52 |

## Audit 结果（2026-09-29，A1_SAA_FROZEN checkpoint）

- **fused 未通过 SYSU gate**：PR-AUC +0.014（门槛 +0.03）、Spearman −0.052（要求不下降）；
  ViT cue 在稀释 detail cue（D >> F）。
- **detail-only 明显优于 ViT-only**：PR-AUC 0.4649→0.5236（+0.059）、
  Top32 precision 0.3945→0.6252（+0.231）、coverage 0.2444→0.4577（+0.213）、
  ROC 0.6848→0.7259（+0.041）；LEVIR 上同样明显更好（coverage 0.2181→0.5875）。
- → 按决策树 §34/§8：**只做 1 个 SYSU detail-only 80K 救援实验（A4_DETAIL_FROZEN）**；
  不再跑 fused。判据不变：F1 ≥ 82.34（+0.30）且 IoU 同向为通过；≤ 82.19 失败 →
  停止 router 迭代转主线二。

## 执行顺序（Stage Gate）

1. **Stage 0 代码冒烟**：`smoke_test.py --mode all`（含新增 A4 测试）。
2. **Stage 1 Router Audit（不训练）**：
   ```bash
   bash audit_A4_SYSU.sh   # primary: Run2 A1_SAA_FROZEN checkpoint
   bash audit_A4_LEVIR.sh
   ```
   **SYSU Audit Gate**（A4 进入 80K 的最低条件，fused vs ViT-only）：
   `PR-AUC ≥ +0.03` 且（`Top32 precision` 或 `coverage ≥ +0.05`）且 `Spearman 不下降`。
   不通过 → 按决策树：detail-only 明显更好则最多做 1 个 SYSU detail-only 救援 run；
   否则停止 CASAA router 迭代，转主线二。
3. **Stage 2 dry run**：`bash dryrun_A4_SYSU.sh`（真实 SYSU ~60 steps，
   检查 loss 有限、ViT checksum 不变、无 shape 错误）。
4. **Stage 3**：只启动 `train_A4_DETAIL_FUSED_FROZEN_SYSU-CD-256.sh`（80K）。
5. **Stage 4**：SYSU 判据通过（F1 ≥ 82.34 且 IoU 同向）才启动 LEVIR（保护判据 ≥ 91.69）。

## 成败判据

- SYSU 通过：A4−A1 ≥ +0.30（82.34），IoU 同向（最好 Recall 改善）；强通过 +0.50；
  失败 ≤ +0.15（82.19）→ 停止 fused router。
- LEVIR：只要求不降超 0.15（91.69）——验证新 cue 不在稀疏/零变化场景制造伪变化。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/<dataset>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASAA/Run3/A4_DETAIL_FUSED_FROZEN/<dataset>/train_log.txt`
- Audit 报告：`/home/yqwang/outputs/CASA-CD/CASAA/Run3/AUDIT/<dataset>/router_audit.txt`

## GPU

只用 **GPU1**（GPU0 让给其它项目），每卡同时最多一个任务。
