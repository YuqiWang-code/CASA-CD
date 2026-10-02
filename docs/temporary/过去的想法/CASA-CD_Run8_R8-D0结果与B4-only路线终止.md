# CASA-CD Run8：R8-D0 审计结果与 B4-only 路线终止记录（课题实验阶段收束）

> **日期**：2026-10-02
> **方案**：`docs/temporary/CASA-CD_Run8_B4-SPE最终路线与预注册.md`（按 §15 执行）
> **执行事实**：只写了 `analyse/run8_b4_dense_recoverability_audit.py`，在 GPU1 用 frozen
> depth-12 corrected DeiT 对 CDD/LEVIR/SYSU/WHU 完整 test 集零训练前向；无任何 80K。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自完整 test 集确定性前向。

---

## 0. 结论先行

```text
[R8-D0-GATE] FAIL
  C1 G1_pixel   (B4 >= B12+0.03): 2/4  （SYSU +0.0644 ✓、WHU +0.0316 ✓；
                                          CDD +0.0162、LEVIR +0.0209 未达）
  C2 G2_boundary(B4 >= B12+0.02): 1/4  （仅 WHU +0.0377 ✓；
                                          CDD −0.0006、LEVIR +0.0164、SYSU +0.0171 未达）
  C3（无数据集 B4 降 >0.01）:     4/4  ✓
```

按预注册 §4.5：

> **永久停止「B4-only / no-detail dense reconstruction」路线。**
> 不改上采样方式、不调 head width、不加 edge/frequency/detail 模块、不回 Mobile/ResNet。
> 直接进入论文分析型收尾：R4-1 作为最强已验证轻量化结构，Run1–8 作为
> budget-allocation / negative-evidence study；明确 `<3M` 终模型目标未被验证达成。

即：**R8-1（B4-SPE）不启动**。B4-SPE head 代码已实现（`models/model/b4_spe_head.py`，
2,030,704 参数 / trainable 53,872，含 T-R8 smoke 定义），作为「被 D0 gate 否决的
最终候选」存档，不进入训练。

## 1. 审计有效性

- 冻结 ViT 与 corrected DeiT 逐位一致（checksum ✓，depth 0-11 + pos_embed）；
- SYSU 对照自检精确复现 R4-1 ResNet 1/8：PR-AUC **0.6535** / Top32 **0.5948** ✓。

## 2. 四数据集 dense recoverability 结果（rank(B4/B12) bilinear → 256×256）

| 数据集 | M1 像素 PR-AUC B4 / B12（lift） | M2 边界带 PR-AUC B4 / B12（lift） | M4 unembed B4 / B12（诊断） |
|---|---|---|---|
| CDD | 0.2173 / 0.2011（+0.0162） | **0.5216 / 0.5222（−0.0006）** | 0.1503 / 0.1402 |
| LEVIR | 0.0908 / 0.0699（+0.0209） | 0.5369 / 0.5205（+0.0164） | 0.0649 / 0.0598 |
| SYSU | 0.4691 / 0.4047（+0.0644） | 0.6113 / 0.5943（+0.0171） | 0.2858 / 0.2721 |
| WHU | 0.0837 / 0.0521（+0.0316） | **0.5718 / 0.5341（+0.0377）** | 0.0599 / 0.0503 |

分桶诊断（Top8 hit / Top32 coverage，B4 vs B12）：
- 1-16 changed patches：CDD 0.429/0.330 vs 0.470/0.347（B12 反而略好）；
  LEVIR 0.596/0.482 vs 0.502/0.409；SYSU 0.664/0.478 vs 0.593/0.427；
  WHU 0.610/0.553 vs 0.559/0.447。
- 17-64 / >64：B4 普遍略优（与 M1 一致）。

## 3. 解读：最后一个假设也被干净证伪

- **「B4 的 16×16 change evidence 足以在无 detail 时支撑像素级重建」不成立。**
  B4 在 token 级强于 B12（R7-D0：0.61 vs 0.51），但在**像素域**优势大幅缩小
  （SYSU +0.064、其余 +0.02~0.03），在**边界带**几乎消失（3/4 数据集 lift
  <0.02，CDD 甚至为负）。
- 这正是 ViT-CoMer 预警的 inner-patch limitation：plain ViT 的 token 变化证据
  在 patch 内部（尤其是真实变化边界 ±4px 邻域）无法被 bilinear 投影保留排序
  能力——**ranking quality ≠ detail-free dense reconstruction ability**。
- 结合 Run3 的旧结论（router ranking 提高 60% 未转化为 F1），课题形成了一条
  完整的方法论纪律：**raw ranking 只做候选筛选，永远不能当作端到端性能的
  替代品**——本课题全部五轮结构搜索都坚持了这一纪律，最后一轮也不例外。

## 4. 停止执行内容（§4.5 / §10，严格遵守）

- ❌ 不启动 R8-1/2/3/4 任何 80K；
- ❌ 不改 D0 阈值、不换上采样、不加 edge/frequency/detail 模块、
  不做 adaptive depth 救场、不回 Mobile/ResNet、不改 loss/LR/steps。

## 5. 课题实验阶段总结（供论文分析型收尾使用）

### 5.1 最终立场（evidence-backed）

| 资产 | 数值 | 用途 |
|---|---|---|
| R4-0 健康 full12 frozen | 83.14 / 71.14 | 健康参考上界 |
| **R4-1 ViT4 + ResNet + 旧 head（最强已验证轻量化结构）** | **82.77 / 70.61** | 论文收尾主结果（8.195M） |
| A1 content 压缩（冻结 ViT，K=64） | ≈baseline（LEVIR −0.01 / SYSU +0.02） | token 冗余 analysis |
| 四数据集 token depth 曲线 | B4 稳健最优、B12 全面劣化、B2 不普适 | depth 冗余 analysis |
| R8-D0 dense recoverability | B4 边界带 lift <0.02（3/4 数据集） | detail-free 重建的边界 |

### 5.2 五轮负结果链（budget allocation study）

```text
Run3   CASAA deployable router：ranking 提升未转 F1（detail Top32 0.39→0.63，F1 −0.12）
Run4   Light32/48/PSD 自定义轻量 detail：端到端 gate 三连败（82.30/82.32/82.02）
Run5   MobileNetV3 预训练 prefix：raw gate FAIL（D4 0.4831<0.50）
Run6   PatchEmbed token（P0）重建：raw gate FAIL（P0 0.3711<0.44）
Run7   B2 深度截断（CSDP）：四数据集 gate FAIL（C2=2/4）
Run8   B4-only 无 detail 重建：dense recoverability gate FAIL（边界带 C2=1/4）
```

### 5.3 论文可成立的机制结论（均已过预注册验证）

1. **Depth redundancy**：12→4 block 仅 −0.37 F1（省 3.56M）；四数据集 B12 全面
   劣于 B2-B4——「最深语义不最适合 BCD」成立，但「统一浅截断（B2）」不成立，
   最优 depth 是 dataset-dependent。
2. **Token redundancy**：Full-Q + K/V 压缩 25% 基本无损；deployable
   change-aware routing 未带来收益。
3. **Detail-branch / dense-reconstruction 负证据**：<0.25M 独立 detail 与
   detail-free token 重建在 ChangeViT 式框架内均未达严格 gate。
4. 统一叙事：**Change Detection does not need uniform computation across
   either tokens or depth** —— 前半句（token）有 A1 支持，后半句（depth）有
   R4-0→R4-1 与四数据集曲线支持；「如何利用非均匀性重建细节」是开放问题。

### 5.4 明确不成立的结论（论文中必须写清）

- ❌ `<3M` 极轻量终模型：**未被验证达成**（R4-1 8.195M 是最强已验证轻量结构）；
- ❌ CASAA change router 提升精度；
- ❌ 任何单 seed 结果宣称统计显著。

## 6. 立即执行顺序（已完成）

- [x] 只新增 `run8_b4_dense_recoverability_audit.py`，GPU1 四数据集跑完
- [x] [R8-D0-GATE] FAIL → 按规则不启动 R8-1
- [x] B4-SPE head 代码存档（实现 + smoke 定义，未训练）
- [x] 本记录 + README + 快照 + 推送 GitHub
