# CASA-CD Run7：CSDP-CD —— Change-Sensitive Depth-Pyramid 可执行预注册方案

> **日期**：2026-10-01
> **依据**：2024–2026 网络调研结论（FDAM/LaViT/ResCLIP/ViT-CoMer/LiFT/EoMT 等）
> 与本仓库证据链（R6-D0：B2 0.6008 ≈ B4 0.6127 > B12 0.5143，SYSU）。
> 调研文献见 `docs/参考文献/文献索引.md`；参考代码见 `others/`。
> **固定协议**：BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、
> seed 16、test-as-val、GPU1、ViT 一律冻结（corrected DeiT loader）。
> **结果纪律**：只认 train_log.txt 最后一个完整 TEST RESULTS 区块。

---

## 0. 一句话预注册摘要

> **Change-Sensitive Intermediate Semantic Source Replacement（CSDP）**：推翻
> 「最深 ViT feature 默认最好」，用变化判别最强的中浅层 DeiT token 作为终止语义源
> （DeiT-Tiny 前 2 个 block，B1+B2），砍掉后续 Transformer 深度与独立 detail
> encoder，用极轻量 cross-depth token head 完成全分辨率变化预测。
> 先四数据集零训练 gate（R7-D0），再两个 SYSU 80K（R7-0 因果 / R7-1 终模型），
> 任一 gate 不过立即停止，不改阈值、不换 B3 救。

## 1. 三个可证伪假设

- **H1 Change-sensitive depth saturation**：四数据集上 `B2 ≈ B4 > B12`（token
  change PR-AUC）不是 SYSU 特例。
- **H2 Intermediate semantic source 能取代 deep semantic source**：同 ResNet/FI/
  decoder 下，仅 Depth4→Depth2，最终 F1 不明显下降。
- **H3 B1+B2 可支撑无独立 detail 的 dense BCD**：换成 CSDP head 后 F1 仍达标。
  （若 H1/H2 成立、H3 失败 → shallow ViT 是好 semantic source 但缺 intra-patch
  detail，机制仍有价值、<3M 终模型未完成。）

## 2. R7-D0：四数据集零训练深度审计（先做，不训练）

新增 `analyse/run7_depth_source_audit.py`：CDD/LEVIR/SYSU/WHU 完整 test，用
frozen depth-12 corrected DeiT 提取 P0/B1-B12 的 1−cos change score（同一权重；
R6-D0 已证明其与 R4-1/R4-0 冻结 checkpoint 的 ViT 逐位一致）。SYSU 附 R4-1
ResNet 1/8 对照自检（0.6535/0.5948 ±0.005），越界 → [AUDIT-INVALID]。

**预注册 gate（三项各自需 ≥3/4 数据集）**：
```text
C1: PR-AUC(B2) >= PR-AUC(B12) + 0.02
C2: PR-AUC(B2) >= PR-AUC(B4)  - 0.03
C3: Top32 precision(B2) >= Top32 precision(B4) - 0.05
```
任一 <3/4 → **CSDP-CD 停止**（不改阈值、不改 B2→B3）。

## 3. R7-0：Source Causality（vit_depth 4→2）

结构 = R4-1 完全一致（ResNet detail + 原 FI + 原 decoder，frozen），唯一变量：
`--vit_depth 4 → 2`。SYSU 80K。

```text
Reference R4-1: F1 82.77 / IoU 70.61
Gate: F1 >= 82.47（drop <= 0.30）且 IoU drop <= 0.35（IoU >= 70.26）
```
FAIL → 「token ranking 好 ≠ 最终 dense 表示好」，方向停止（不实现 CSDP head）。

## 4. R7-1：最终模型（ViT2 + CSDP head）

删除 ResNet18 / FeatureInjector / legacy decoder / P0 / CASAA router。
保留：PatchEmbed → Block1 → Block2（frozen，1,087,104 参数）→ CSDP head。

### CSDP head（`models/model/depth_pyramid_head.py`，预注册定稿）

对 l∈{1,2}：
```text
D_l = P_l([ |T_A^l − T_B^l| , (T_A^l + T_B^l)/2 ])      # concat 384 → Linear(384→96)
Z = D_1 + D_2                                          # B×256×96 → reshape 96×16×16
```
Sub-patch Expansion（每级：1×1 通道扩张 + BN + ReLU → PixelShuffle×2 → DW 3×3 + BN + ReLU）：
```text
96×16×16 → conv1×1 96→128 → PS2 → 32×32×32 → DW3×3(32)
        → conv1×1 32→64  → PS2 → 64×64×16 → DW3×3(16)
        → conv1×1 16→64  → PS2 → 128×128×16 → DW3×3(16)
        → conv1×1 16→16  → PS2 → 256×256×4  → DW3×3(4) → conv1×1 4→1 → sigmoid
```
B1/B2 均过 final LN（与 R6-D0 审计口径一致）。T1/T2 共享权重、head 只依赖
[|Δ|, mean] → 严格时间交换对称。

### 参数预算（设计值，正式以代码打印为准）

```text
ViT2 (patch_embed+pos+mask+2 blocks+norm) = 1,087,104
CSDP head（P1/P2 73,920 + 各 stage 16,912） ≈    90,832
TOTAL                                      ≈ 1,177,936 ≈ 1.178M
```

### R7-1 预注册 gate（SYSU 80K）

```text
F1 >= 82.30 且 IoU >= 69.92（沿用最终候选绝对地板）
effective params <= 1.50M（目标；硬门槛 <3M）
FLOPs <= 2.00G
```
FAIL → 主方案终止（H3 证伪）；不回 detail、不改 loss、不做 sweep。

## 5. R7-2：LEVIR 跨场景 gate（R7-1 PASS 后才跑）

```text
F1 >= 91.50 且 IoU >= 84.33；params/FLOPs 同上
```
FAIL → 不补 CDD/WHU，不宣称跨数据集轻量 SOTA。

## 6. R7-3/4：CDD / WHU 定稿覆盖（LEVIR PASS 后）

```text
CDD: F1 >= 97.25（baseline 97.75，drop<=0.50）且 IoU >= 94.65
WHU: F1 >= 94.34（baseline 94.84，drop<=0.50）且 IoU >= 89.29
```
只做最终覆盖，不再改结构；失败只记录局限。

## 7. 停止规则总表

| Gate | 失败动作 | 永久禁止 |
|---|---|---|
| R7-D0 | CSDP 停止 | 改阈值、B2→B3 救、回 detail branch |
| R7-0 | 不实现 CSDP head | depth2 继续调参、解冻 ViT |
| R7-1 | 主方案终止 | head 调 width/gate、加 loss、回 ResNet/FI |
| R7-2 | 不补 CDD/WHU | LEVIR 特调 |
| R7-3/4 | 记录局限 | 单数据集结构修补 |

## 8. 论文机制故事（gate 全过才成立）

统一主题：**Change Detection does not need uniform computation across either
tokens or depth.**
- 第一层（token，CASAA/A1 留存）：Full Query + K/V 压缩 25% 基本无损；
- 第二层（depth，CSDP 主贡献）：change-discriminative representation 在
  shallow/intermediate depth 已形成（B2≈B4>B12，跨四数据集），继续加深反而
  损伤 localized change evidence → 按 task-specific change separability 截断
  pretrained plain ViT，11.75M → ~1.18M 是机制的结论而非目的。
- 外部机制支撑：FDAM（frequency vanishing）、LaViT（attention saturation）、
  ResCLIP（intermediate localization）、ViT-CoMer（inner-patch 交互不足）。
  文献只作 motivation，因果以本仓库四数据集审计 + 80K 为准。

## 9. 立即执行顺序

1. 先只写 `analyse/run7_depth_source_audit.py`，GPU1 跑四数据集 → 读 [R7-D0-GATE]。
2. PASS → R7-0（vit_depth=2 脚本 + dryrun + 80K）。
3. R7-0 PASS → 实现 `depth_pyramid_head.py` + `detail_mode=depth_pyramid` +
   `head_mode=csdp` + T-R7 smoke（exact 继承 / 参数 1,177,936 / 对称 <1e-6 /
   冻结 checksum / FLOPs ≤2.0G）→ dryrun → R7-1 SYSU 80K。
4. R7-1 PASS → R7-2 LEVIR → CDD/WHU。
5. 每步后更新 README/Excel/快照/推送。
