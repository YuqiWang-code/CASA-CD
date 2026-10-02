# CASA-CD Run11（TASS）：TASS-D0 审计结果与 Run11 终止记录

> **日期**：2026-10
> **方案**：`docs/temporary/CASA-CD_下一步方案_Run11_TASS_设计与预注册.md`（按 §7 / §15 执行）
> **执行事实**：只写了 `analyse/run11_tass_source_gate.py`（参数自由 raw-RGB 差分代理 +
> Rfuse 融合 + 边界带 PR-AUC），在 GPU1 用 frozen depth-4 corrected DeiT 对
> CDD/LEVIR/SYSU/WHU 完整 test 集零训练前向；**未实现 TASS 模型、未跑任何 80K**。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自完整 test 集确定性前向。

---

## 0. 结论先行

```text
[TASS-D0-GATE] FAIL
  G0 (审计有效性)                            : PASS（B4 边界带 PR-AUC 四数据集与 SF-D0
                                               4 位小数逐位复现；ViT4 checksum；样本数一致）
  G1 (PRbnd(Rfuse) >= PRbnd(B4)+0.020,
      SYSU 必过且 >=2/4)                      : 1/4（仅 SYSU +0.0309 过；CDD −0.0075 /
                                               LEVIR −0.0409 / WHU −0.0297）
  G2 (PRpix(Rfuse) >= PRpix(B4)-0.010,
      SYSU 必过且 >=3/4)                      : 2/4（SYSU/WHU 过；CDD −0.0346 / LEVIR −0.0032）
```

按预注册 §7.4 / §11-F1：

> **Run11 STOP。不实现 TASS、不跑任何 80K、不换 proxy/权重/宽度刷 gate。**
> 按 §16 锁定 fallback：不再开 Run12「换 stem / 加 edge / 调宽度」救援线，
> **下一轮直接进入分析型论文收尾设计（方向 c）**。

**H11 的前提（原始像素网格在 1/4–1/16 上普遍含有相对 B4 互补的 boundary evidence）
被四数据集证伪——raw pixel evidence 只在 SYSU 成立，在建筑/伪变化数据集上被
radiometric pseudo-change 主导。第九轮负结果。**

## 1. 审计有效性（G0 全过）

- 冻结 ViT4 与 corrected DeiT 初始化逐位一致（checksum ✓）；
- 样本数：四个数据集 list/test.txt 与两个 loader 逐一相等；
- **B4-only 边界带 PR-AUC 与 SF-D0 记录 4 位小数逐位一致**：
  CDD 0.5216 / LEVIR 0.5369 / SYSU 0.6113 / WHU 0.5718。

> 注：第一版审计曾因「raw uint8 域先 Scale 再手动归一化」与参考管线
> （Normalize→Scale 的 float32 域缩放）存在 cv2 定点插值舍入差异而 AUDIT-INVALID；
> 改为双 loader（raw loader 供代理 + 参考管线 loader 供 ViT 路径）后逐位复现，
> 属于审计口径修复，不涉及阈值改动。

## 2. 四数据集结果（rank 分数重采样 256×256）

| 数据集 | B4-only 像素 / 边界 | Rfuse 像素 / 边界 | Rspatial 像素 | 边界 lift（G1） | 像素差（G2） |
|---|---|---|---|---|---|
| CDD | 0.2173 / 0.5216 | 0.1727 / 0.5141 | 0.1379 | **−0.0075** ✗ | **−0.0346** ✗ |
| LEVIR | 0.0908 / 0.5369 | 0.0776 / 0.4960 | 0.0558 | **−0.0409** ✗ | **−0.0032** ✗ |
| SYSU | 0.4691 / 0.6113 | 0.4726 / 0.6423 | 0.4119 | **+0.0309** ✓ | +0.0035 ✓ |
| WHU | 0.0837 / 0.5718 | 0.0751 / 0.5421 | 0.0624 | **−0.0297** ✗ | +0.0014 ✓ |

## 3. 解读：raw pixel evidence 是 dataset-dependent 的，且只在 SYSU 互补

- **G1=1/4 是干净证伪**：参数自由的 raw-RGB 通道均值差在 1/4–1/16 尺度上，
  只有变化最密集的 SYSU（中位变化 patch 50，通用地表变化）能提供相对 B4 的
  边界带互补证据（+0.0309）；三个建筑数据集（CDD/LEVIR/WHU，变化稀疏 + 强辐射
  伪变化）上 Rfuse 边界带反而低于 B4-only（−0.0075 / −0.0409 / −0.0297）。
- **G2=2/4 进一步说明**：CDD 上 Rfuse 的**全像素**证据被 raw 差分显著拉低
  （−0.0346）——raw 差分的伪变化噪声不只是「边界带无用」，而是整体稀释语义证据。
- **与八轮证据链完全同向**：R9 已证「改变采样 lattice 不提供互补证据」；本审计证
  「未训练的 raw 像素差分也不提供跨数据集互补证据」。项目反复得到同一结论：
  **dense evidence 必须被任务训练过，但训练前的 raw feasibility 在三个建筑数据集
  上不成立**——在「先过 raw gate」的项目纪律下，这条路径到此为止。
- **SYSU +0.0309 的例外如实记录**：它在预注册口径下不足以放行（要求 ≥2/4 且
  SYSU 必过，SYSU 过但只有 1/4），且 G2 的 CDD/LEVIR 双失败独立阻止放行。

## 4. 停止执行内容（严格遵守）

- ❌ 不实现 `tass_stem.py / str_tass_fusion.py`；
- ❌ 不跑 smoke/dry run/任何 80K；
- ❌ 不改 0.5/0.5 融合权重、不换 Sobel/Laplacian 代理、不调宽度、不加 edge branch；
- ❌ 不开 Run12「换 stem 救援线」（§16 锁定）。

## 5. 课题状态更新（九轮负结果后的证据总结）

```text
Run3   CASAA deployable router      ：ranking↑ 未转 F1（证伪）
Run4   自定义轻量 detail ×3         ：端到端 gate 三连败（证伪）
Run5   MobileNet 预训练 prefix      ：raw gate FAIL（证伪）
Run6   P0 重建                      ：raw gate FAIL（证伪）
Run7   B2 深度截断                   ：四数据集 gate FAIL（证伪）
Run8   B4-only dense 重建           ：boundary gate FAIL（证伪）
Run9   O-PRE 采样格变密             ：G0/G1 全 0/4（证伪）
Run10  多深度 token 金字塔(融合线)  ：SF-D0 G1 0/4（证伪）
Run11  raw 空间证据(TASS-D0)        ：G1 1/4、G2 2/4（证伪，本记录）
```

- 九轮全部在 **0–1 个正式 80K** 内被预注册 gate 拦截（正式训练总消耗仍为 1 个 R4-2d）。
- 证据链现覆盖：token routing、detail 结构、patch 重建、深度截断、单/多深度 dense
  重建、采样格、可训练折叠解码器接口、未训练 raw 像素差分——**冻结 plain ViT 框架内
  「零/极小参数补回 dense 边界」的路径与「未训练 raw 证据跨数据集可行性」均已系统排除**。
- **按方案 §16：下一轮直接进入分析型论文收尾设计（方向 c）**，核心材料已齐：
  CASAA ranking≠F1；depth truncation（B4 robust/B12 worse）；token ranking ≠ dense
  boundary reconstruction；denser lattice ≠ complementary；depth pyramid ≠ sub-patch
  evidence；untrained raw diff ≠ cross-dataset complementarity——形成完整的
  negative-evidence / budget-allocation study；主结果 = R4-1 ViT4+ResNet+旧头 82.77。

## 6. 结果记录

- 审计报告：`/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run2_TASS/TASS_D0_SOURCE_GATE/audit_report.txt`
- 明细 JSON：同目录 `tass_d0_results.json`
- 代码：`analyse/run11_tass_source_gate.py` + `train_scripts/STR-Fusion/Run2_TASS/`（gate 工具存档）

---

## 7. 用户决策附录（2026-10，训练照常执行）

> 用户指令：**取消预注册 gate 拦截；服务器两张 GPU 均可用；必须训练**；硬目标远未
> 达成、论文为时尚早。

据此修订执行状态：

- **TASS-D0 FAIL 保留为已知负证据记录，不再作为停止条件**；
- Run11 训练照常执行：TASS 已实现（`models/model/tass_stem.py` +
  `models/model/str_tass_fusion.py`），C0_TOKEN（GPU1）与 M1_TASS（GPU0）按
  SYSU→LEVIR→WHU→CDD 从头 80K 双卡启动；
- **结果解释口径**：由于 gate 未过，M1 相对 C0 的任何增益都必须在「TASS-D0 曾为负
  （raw 空间证据仅在 SYSU 互补）」的背景下陈述；本文档 §0–§5 的 gate 结论不回改。
