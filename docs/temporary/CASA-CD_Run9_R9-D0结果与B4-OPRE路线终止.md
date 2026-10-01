# CASA-CD Run9：R9-D0 审计结果与 B4-OPRE 路线终止记录

> **日期**：2026-10-02
> **方案**：`docs/temporary/CASA-CD_Run9_可执行预注册方案.md`（按 §16 执行）
> **执行事实**：只写了 `analyse/run9_overlap_reembedding_audit.py`，在 GPU1 用 frozen
> depth-12 corrected DeiT 对 CDD/LEVIR/SYSU/WHU 完整 test 集零训练前向；无任何 80K。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自完整 test 集确定性前向。

---

## 0. 结论先行

```text
[R9-D0-GATE] FAIL
  G0 (PRbnd(OPRE) >= PRbnd(P0) + 0.03) : 0/4
  G1 (PRbnd(proxy) >= PRbnd(B4) + 0.02) : 0/4
  G2 (G2a=3/4, G2b=1/4)
  G3 (SYSU Δbnd>=+0.03 且 Δpix>=+0.015)  : False（Δbnd +0.0038 / Δpix +0.0025）
```

按预注册 §8.5 / §12 Stop-0：

> **Run9 B4-OPRE 停止；不实现训练模型，不启动任何 80K。**
> 不把 stride 8 改成 4/6/12、不改 reflect pad、不换 score 重刷、不加多 phase
> ensemble、不放宽 G0–G3、不训练「看看端到端会不会救回来」。

**H9（改变采样 lattice 能提供与 B4 互补的局部几何 evidence）被四数据集一致证伪。**

## 1. 审计有效性

- 冻结 ViT 与 corrected DeiT 逐位一致（checksum ✓，depth 0-11 + pos_embed）；
- SYSU 对照自检精确复现 R4-1 ResNet 1/8：PR-AUC **0.6535** / Top32 **0.5948** ✓。

## 2. 四数据集结果（rank 分数 bilinear → 256×256）

| 数据集 | P0 pix/bnd | B4 pix/bnd | O-PRE pix/bnd | proxy pix/bnd | O-PRE vs P0 bnd lift | proxy vs B4 bnd lift |
|---|---|---|---|---|---|---|
| CDD | 0.1824 / 0.5334 | 0.2173 / 0.5216 | 0.1836 / 0.5488 | **0.2406 / 0.5387** | +0.0154 | +0.0171 |
| LEVIR | 0.0554 / 0.4857 | 0.0908 / 0.5369 | 0.0531 / **0.4763** | 0.0880 / 0.5215 | **−0.0094** | **−0.0154** |
| SYSU | 0.3460 / 0.5898 | 0.4691 / 0.6113 | 0.3332 / 0.5940 | 0.4716 / 0.6151 | +0.0042 | +0.0038 |
| WHU | 0.0364 / 0.4704 | 0.0837 / 0.5718 | 0.0359 / **0.4544** | 0.0672 / 0.5357 | **−0.0160** | **−0.0361** |

分桶与冗余度诊断（不参与 gate）：
- Spearman(OPRE, B4)：CDD +0.2665（低冗余，但低冗余 ≠ 互补——G1 已证）；
- bucket-OPRE 的 Top8 hit 在 1-16 组明显低于 P0/B4（CDD：0.268 vs 0.551/0.429）——
  O-PRE 对小变化的定位甚至弱于 canonical P0。

## 3. 解读：采样几何假设被否——第七轮负结果

- **G0=0/4 是最强的证伪**：同一预训练 patch kernel 换到 stride=8 重叠 lattice 后，
  边界带排序能力没有系统性提升（CDD/SYSU 微升 ≤0.015，LEVIR/WHU 反而下降）。
  「固定 non-overlap patchification 是 dense recovery 的主瓶颈」这一命题不成立——
  bottleneck 不在 patch 采样的 lattice 密度。
- **G1=0/4**：O-PRE 与 B4 的固定 proxy 融合不能带来 0.02 的边界 lift，
  在 LEVIR/WHU 反而稀释 B4（−0.015 / −0.036）——与 Run6 的「P0+B4 融合稀释」
  现象同类，只是这次换了采样格。
- 结合 Run4（1/2 浅层特征不弱但深层崩坏）、Run6（P0 弱）、Run8（B4-only 边界不足）、
  Run9（lattice 变密无用）——完整证据链现在指向：**plain ViT 的 patch 投影与
  token 表征在冻结条件下无法通过「重采样/重建」方式补回 dense 边界信息**；
  补 dense evidence 的可行路径需要的是**训练过的、任务适配的空间表征**
  （ResNet detail 之所以有效，或许正因为它在 CD 任务上被训练），而不是冻结
  先验的几何复用。

## 4. 停止执行内容（严格遵守）

- ❌ 不实现/不训练 R9-1..R9-4 与 R9-A1——OPREHead 代码已实现并通过 T-R9 全套 smoke
  （总 2,036,945 参数 / trainable 60,113 / FLOPs 1.5270G ≤2.0G / 时间交换对称
  0.00 / 无 duplicate patch kernel / DeiT 逐位继承），作「被 gate 否决的候选」
  存档，不进入训练；
- ❌ 不改 stride/padding、不换 score 函数重刷 gate、不加 edge/frequency/detail
  模块、不解冻 patch embed、不改 loss/LR/steps。

## 5. 课题状态更新（七轮负结果后的证据总结）

```text
Run3  CASAA router          ：ranking↑ 未转 F1（证伪）
Run4  自定义轻量 detail ×3   ：端到端 gate 三连败（证伪）
Run5  MobileNet 预训练 prefix：raw gate FAIL（证伪）
Run6  canonical P0 重建      ：raw gate FAIL（证伪）
Run7  B2 深度截断            ：四数据集 gate FAIL（证伪）
Run8  B4-only 无 detail      ：dense boundary gate FAIL（证伪）
Run9  O-PRE 采样格变密       ：G0/G1 全 0/4（证伪）
```

- 全部七轮都在 **0-1 个正式 80K** 内被预注册 gate 拦截（正式训练总消耗仍为 1 个
  R4-2d），方法纪律一致：raw ranking 只做筛选。
- 新硬目标（SYSU 85 / LEVIR 92.5 / WHU 95 / CDD 98，<3M）**远未达成**，且当前
  证据表明冻结-pretrained ViT 框架内的「零参数/极小参数 dense 重建」路径已被
  系统排除。下一步若继续冲击目标，需要新的机制假设（例如：允许任务适配的可训练
  spatial head 承担更大预算、重新审视「冻结 ViT」约束、或换 semantic backbone），
  并且必须重新预注册。
