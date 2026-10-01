# CASA-CD Run7：R7-D0 四数据集审计结果与 CSDP-CD 停止记录

> **日期**：2026-10-01
> **方案**：`docs/temporary/CASA-CD_Run7_CSDP方案与预注册.md`（含四数据集预注册 gate）
> **执行事实**：只写了 `analyse/run7_depth_source_audit.py`（未启动任何训练），在 GPU1
> 用 frozen depth-12 corrected DeiT 对 CDD/LEVIR/SYSU/WHU 完整 test 集零训练前向，
> 提取 P0/B1-B12 的 1−cos token change score。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自完整 test 集确定性前向。

---

## 0. 结论先行

```text
[R7-D0-GATE] FAIL
  C1 (B2 >= B12+0.02) : 3/4  ✓（CDD 差 0.0016 未过）
  C2 (B2 >= B4 -0.03) : 2/4  ✗（CDD、WHU 未过）→ 判定 FAIL
  C3 (prec B2 >= prec B4 -0.05) : 4/4 ✓
```

按预注册规则（方案 §2/§7）：

> **整个 CSDP-CD 停止。不改 threshold。不改 B2→B3 去救。不启动 R7-0/R7-1。**

H1「change-sensitive depth saturation 跨数据集成立（B2≈B4>B12）」**被证伪**：
该模式只在 SYSU、LEVIR 成立；在 CDD、WHU 上 B4（CDD）或 B3（WHU）明显强于 B2。

## 1. 审计有效性

- 冻结 ViT 与 corrected DeiT 初始化逐位一致（checksum ✓，depth 0-11 + pos_embed）；
- SYSU 对照自检精确复现 R4-1 ResNet 1/8：PR-AUC **0.6535** / Top32 **0.5948** ✓。

## 2. 四数据集 per-source PR-AUC（token change ranking，1−cos）

| source | CDD | LEVIR | SYSU | WHU |
|---|---:|---:|---:|---:|
| P0 | 0.2816 | 0.1998 | 0.3711 | 0.0818 |
| B1 | 0.3638 | 0.2694 | 0.5349 | 0.1825 |
| **B2** | **0.4164** | **0.2947** | **0.6008** | **0.2660** |
| B3 | 0.4524 | 0.3017 | 0.6108 | **0.3508** |
| **B4** | **0.4594** | 0.3009 | 0.6127 | 0.3449 |
| B12 | 0.3980 | 0.2465 | 0.5143 | 0.1693 |

Top32 precision：

| source | CDD | LEVIR | SYSU | WHU |
|---|---:|---:|---:|---:|
| B2 | 0.3620 | 0.1705 | 0.5721 | 0.1078 |
| B4 | 0.3515 | 0.1685 | 0.5535 | 0.1088 |
| B12 | 0.3253 | 0.1431 | 0.4886 | 0.0788 |

## 3. 解读：depth 现象是数据集相关的，不是普适机制

- **SYSU / LEVIR**：B2 已进入平台（B2≈B4，且 B2>B12）→ 支持 shallow saturation；
- **CDD**：B2 < B4 差 0.043（B4 为四深度最优）——CDD 的抗伪变化任务需要更深语义；
- **WHU**：B2 < B3/B4 差 ~0.08——稀疏小建筑变化在更深的 block 才成形
  （且 WHU 全深度绝对值都低：B4 仅 0.345，对比 SYSU 0.613）。
- 结论：**最优 change-sensitive depth 不是单一常量**——它随数据集的
  「变化稀疏度/语义依赖」变化（SYSU 密集地表变化浅层即够；CDD 抗伪变化与
  WHU 稀疏建筑需要 B3/B4）。「一刀切截断到 B2」的 CSDP 主线因此不成立；
  B12（full12 终 token）在所有数据集都劣于 B2-B4，**「最深未必最好」部分仍成立**，
  但「B2 最优」被否。

## 4. 停止执行内容（严格遵守）

- ❌ 不启动 R7-0（VIT2_OLDHEAD）与 R7-1（CSDP）任何 80K；
- ❌ 不改 gate 阈值、不把 B2 换成 B3 救场、不回 detail branch、不解冻 ViT；
- CSDP head 代码（`models/model/depth_pyramid_head.py`，已 smoke：1,177,936 参数、
  0.673G、严格时间交换对称）**留作资产**，不进入训练。

## 5. 本轮净收益（进入研究记录的结论）

1. **四数据集 token-level depth 曲线**（上表）：这是本课题目前最完整的
   「change-sensitive depth」证据，可直接用于论文分析章节——说明
   optimal semantic depth 是 task/dataset dependent，并支持
   「depth-4（B4）是四数据集稳健最优（CDD/SYSU 第一、LEVIR/WHU 第二），
   full12 全面劣化」。
2. **B4 ≈ 四数据集稳妥点**：现有 R4-1（ViT4+ResNet+legacy）正是 B4 路线，
   其 82.77 SYSU 是经过验证的；后续若做轻量化，语义源应保持 B3/B4 级别，
   而不是 B2。
3. 文献证据（FDAM/LaViT/ResCLIP 的 frequency vanishing / attention saturation /
   intermediate localization）对本现象的解释力保留——它解释「深于 B4 为什么变差」，
   但不支持「B2 就够」。
4. `analyse/run7_depth_source_audit.py` 成为四数据集 depth audit 的标准工具。

## 6. 下一步决策叉（供下一步调研/预注册）

CSDP-CD（B2 截断）被四数据集 gate 否决后，候选（均未预注册，不作数）：
1. **保持 B4 语义源**（R4-1 已验证 82.77），把轻量化重心放回 head——但 Run5
   SGDP / Run6 PTPR 均已在「B4 + 轻量 head」方向上被 raw gate 拦下，需要新证据
   才可重启；
2. **depth-adaptive / dataset-adaptive source**（把「最优 depth 依赖数据集」本身
   变成机制：多 depth token 的轻量可学习聚合，而不是硬截断）——但这又接近
   RFL-CDNet 的 intermediate fusion 思路，需论证与它的差异；
3. 回到 Run4 证据最强的资产组合（ViT4 + ResNet detail + legacy head，82.77），
   以「训练协议级轻量化」（如 A1 K=64 压缩 + 冻结）作为论文主线收尾，
   把 depth/detail 负结果作为系统性分析。
   该方向与本仓库全部正面证据兼容，风险最低，但创新性弱于 2。

建议下一步把本节原样带给 GPT 做最终路线决策。
