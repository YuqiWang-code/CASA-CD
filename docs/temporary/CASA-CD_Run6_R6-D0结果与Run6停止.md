# CASA-CD Run6：R6-D0 审计结果与 Run6 停止记录（P0/Fuse gate FAIL）

> **日期**：2026-10-01
> **方案**：`docs/temporary/CASA-CD_Run6_ViT语义预算重分配_可执行预注册方案.md`（按 §18 执行）
> **执行事实**：只写了 `analyse/run6_semantic_token_audit.py`（未动模型代码），在 GPU1 用
> R4-1/R4-0 best checkpoint + corrected DeiT 对完整 SYSU test（4000 对）零训练前向。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自完整 test 集确定性前向。

---

## 0. 结论先行

```text
[R6-D0-GATE] FAIL
  G1 P0   : PR-AUC 0.3711 < 0.44  FAIL（Top32 prec 0.4524 >= 0.40 过）
  G2 B4   : PASS（PR 0.6127 / prec 0.5535 / Spearman +0.4255）
  G3 Fuse : PR-AUC 0.4753 < 0.50  FAIL（Spearman 0.2427 < 0.25 亦不过；prec 0.5533 过）
```

按预注册 §4.6 / §9：

> **不实现 / 不训练 R6-1；永久停止「ViT4 token-only + token reconstruction」路线。**
> 后续唯一允许方向：新的 semantic source 重构（候选 3），另开下一轮重新预注册。
> 禁止：再找 detail branch、放宽 D0 threshold、改 score 重跑、width/depth sweep。

即：PTPR（`token_reconstructor.py`）**不实现**，R6-1/2/3/4 全部不启动——R6-D0 再次
以零训练成本淘汰了一条路线。

## 1. 审计有效性（必须先立）

- **冻结 ViT checksum**：R4-1 的 patch_embed/blocks0-3/norm/pos_embed 与 corrected
  DeiT 初始化逐位一致 ✓（§4.2）。
- **口径自检（§4.5）**：同 run 复现 R4-1 ResNet 1/8 —— PR-AUC **0.6535**、
  Top32 prec **0.5948**，与 R4-D0 历史值逐位一致 → 审计有效，非口径漂移。

## 2. 全部 per-source 指标（完整 SYSU test 4000 对）

| source | PR-AUC | Spearman | Top32 prec | Top32 coverage |
|---|---:|---:|---:|---:|
| P0（PatchEmbed raw，pos 前） | **0.3711** | +0.1162 | 0.4524 | 0.3031 |
| B1（block0+LN） | 0.5349 | +0.3207 | 0.5456 | 0.3828 |
| B2（block1+LN） | 0.6008 | +0.4092 | 0.5721 | 0.4203 |
| B3（block2+LN） | 0.6108 | +0.4263 | 0.5570 | 0.3904 |
| B4（block3+LN，= V4） | **0.6127** | +0.4255 | 0.5535 | 0.3834 |
| B12（R4-0 full12 终 token，诊断） | 0.5143 | +0.3290 | 0.4886 | 0.3274 |
| FUSE = 0.5R(P0)+0.5R(B4) | 0.4753 | +0.2427 | 0.5533 | 0.3872 |

分桶诊断（§4.7，不参与 gate）：

| source | 1-16（Top8 hit / Top32 cov） | 17-64 | >64 |
|---|---:|---:|---:|
| P0 | 0.621 / 0.453 | 0.809 / 0.351 | 0.949 / 0.185 |
| B2 | 0.744 / 0.566 | 0.955 / 0.513 | 0.996 / 0.257 |
| B4 | 0.664 / 0.478 | 0.934 / 0.467 | 0.996 / 0.253 |
| B12 | 0.593 / 0.427 | 0.902 / 0.393 | 0.996 / 0.213 |
| FUSE | 0.733 / 0.525 | 0.920 / 0.466 | 0.991 / 0.242 |

## 3. 可证伪假设的裁决

- **H6-A（PatchEmbed token 仍含可利用的局部 change evidence）：证伪。**
  P0 PR-AUC 0.3711，远低于 0.44 地板——16×16 patch 的 192 通道在进入任何
  transformer block 之前，1−cos 变化判别力只略高于随机基率（~0.19）。
  「从 pre-position patch token 解码 intra-patch spatial phase」的信息源不足。
- **H6-B（浅 patch evidence 与深 semantic evidence 互补）：证伪。**
  Fuse（0.4753）反而低于 B4 单独（0.6127）——0.5/0.5 rank 融合被弱 P0 稀释。
  P0 不是互补源，是噪声源。
- **H6-C/D 未测试**（按规则不进入训练阶段）。

## 4. 附带的正资产（研究记录）

1. **B4（ViT4 终 token）语义变化判别力很强**：PR-AUC 0.6127，接近 ResNet 1/8
   detail（0.6535），且 **B4 > B12**（0.6127 vs 0.5143）——1−cos 视角下，
   full12 的最终 token 反而更「物体级」、变化判别更弱；这再次支持 R4-0→R4-1
   的 depth redundancy 叙事，并提示「shallow ViT 终 token 本身就是合格的
   semantic change source」。
2. **B1→B4 单调增强**（0.535→0.613）：全局 mixing 在 token 级逐步建立变化判别；
   变化信息不需要独立的 raw-image CNN 来提供 semantic 级证据。
3. **R6-D0 审计工具**（`analyse/run6_semantic_token_audit.py`）可直接复用于
   「候选 3：semantic source replacement」的任何新 semantic backbone——
   换源后照跑同一 gate（可加新门槛重新预注册）。

## 5. 停止执行内容（§9，严格遵守）

- ❌ 不实现 `token_reconstructor.py` / `detail_mode=token_recon`；
- ❌ 不启动 R6-1/2/3/4 任何 80K；
- ❌ 不做 PTPR bottleneck sweep、PixelShuffle stage 增减、SGDP 调参、
  depth4→3 / width192→128 rescue、回退 detail branch、改 loss/LR/steps。

## 6. 下一轮唯一出口（需重新预注册）

> **semantic source replacement（候选 3）**。

证据现状：B4=0.61 / P0=0.37 / Fuse=0.48 / Mobile-1/8=0.53 / ResNet-1/8=0.65——
semantic（B4）已强、浅层 patch 弱、外部 micro-detail 中等但都不过严格 gate。
下一轮候选语义源（供 GPT 决策参考，未预注册不作数）：更浅 ViT（depth 2-3）终
token、MobileNetV3 更深层 1/16 语义 token、ViT4 中段 block token（B2 已 0.60）、
或 token 级上采样 + SGDP 直连（去掉 P0 融合）。每个候选照 R6-D0 模式先零训练
gate（含 CTRL 自检），过 gate 才花 80K。

## 7. 立即执行顺序（已完成）

- [x] 归档旧决策文档至 `docs/temporary/过去的想法/`
- [x] 只写 `analyse/run6_semantic_token_audit.py`（未改模型代码）
- [x] GPU1 完整 SYSU test 跑 R6-D0（checksum + CTRL 自检 + P0/B1-B4/B12/Fuse）
- [x] [R6-D0-GATE] FAIL → 按规则不实现/不训练 R6-1
- [x] 本记录 + README + 快照 + 推送 GitHub
