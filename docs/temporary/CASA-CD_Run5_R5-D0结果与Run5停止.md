# CASA-CD Run5：R5-D0 审计结果与 Run5 停止记录（Mobile raw gate FAIL）

> **日期**：2026-10-01
> **方案**：`docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md`（GPT 版，已按 §58 执行）
> **执行事实**：MobileNetV3-Small 官方权重已备（`pretrained_weight/mobilenet_v3_small-047dcff4.pth`，
> 实测 features0-3 = 10,488 参数）；`mobile_detail.py` / `sgdp_head.py` 已实现并通过
> smoke（T-R5 全绿：exact 继承、参数 2,102,587、SGDP 时间交换对称、FLOPs 1.6778G ≤2.0G）；
> R5-D0 在 GPU1 用完整 SYSU test（4000 对）无训练跑完。
> **结果纪律**：本审计无正式 TEST RESULTS；raw gate 数字来自完整 test 集的确定性前向。

---

## 0. 结论先行

```text
[MOBILE-GATE] FAIL
  D4 PR-AUC  = 0.4831  (门槛 >= 0.50)   ✗  差 0.017
  D8 PR-AUC  = 0.5317  (门槛 >= 0.52)   ✓
  D4 Top32 prec = 0.5006 (门槛 >= 0.46) ✓
  D8 Top32 prec = 0.5105 (门槛 >= 0.48) ✓
```

按预注册 §12：

> **不训练 MobileDetail。Run5 候选 ② 立即永久停止。**
> 后续另开 Run6，只讨论 ViT 预算重分配；不再找第二个、第三个
> MobileNet / EfficientNet / ShuffleNet prefix。

即：**R5-1 / R5-2 / R5-3 全部不启动**（这正是 raw gate 的设计目的——用一个
零训练 audit 淘汰边缘候选，省下 1-3 个 80K）。

---

## 1. R5-D0-D：MobileDetail-P3 raw gate（预注册判据）

与 R4-D0 完全相同的协议（pool 16×16、s=1−cos(f1,f2)、GT occupancy>0）：

| 尺度 | PR-AUC | Spearman | Top32 prec | Top32 coverage | 门槛（PR/prec） |
|---|---:|---:|---:|---:|---|
| 1/4 (f1) | **0.4831** | +0.2825 | 0.5006 | 0.3287 | 0.50 / 0.46 |
| 1/8 (f2→f3) | 0.5317 | +0.3209 | 0.5105 | 0.3318 | 0.52 / 0.48 |

对照（同一协议）：
ResNet 1/4=0.6263 / 1/8=0.6535；Light32 0.3251 / 0.3173；Light48 0.3965 / 0.3282。

**解读**：MobileDetail 是迄今最强的非 ResNet detail（1/4 处比 Light48 高 0.087），
但 1/4 只达到 ResNet 的 77%（预注册要求 80%）。按规则 FAIL，不因"接近"放行。

---

## 2. R5-D0-A（H2）：PSD pretrained stem 漂移 —— 强支持被重写

| 指标 | R4-1 ResNet stem | R4-2d PSD stem |
|---|---:|---:|
| conv rel_L2（vs ImageNet 初始） | 0.2108 | **0.6388** |
| conv cosine | 0.9931 | 0.8476 |
| bn weight/bias rel_L2 | 0.129 / 0.216 | 0.319 / 0.307 |
| bn running_mean/var rel_L2 | 11.847 / 0.637 | 10.863 / 0.863 |
| 激活 cos(init, trained)（前 256 对） | 0.9161 | **0.8376** |
| 激活 RMS ratio | 0.9690 | 0.7848 |

判据（§6.3）：PSD conv rel_L2 ≥0.15 且 ≥1.5×R4-1（0.639 ≥ 0.316 ✓），
激活 cos 0.838<0.90 且 R4-1≥0.95（✓）→ **强支持「PSD stem 被重写」**。
GPT 本轮把它排第 4，实测应上调为 PSD 失败的候选主因之一；但注意 R4-1 的 ResNet
stem 同样在统一 lr 下训练，rel_L2 只有 PSD 的 1/3——所以问题更可能是「PSD 的
7×7 stem 承担了不成比例的重写压力」，而不是「统一 lr 必然毁掉 pretrained stem」。

## 3. R5-D0-B（H1）：TileAdapter（inference-only）—— adapter 功能较弱

用零参数 TileAdapter（channel repeat×2/√2）替换 R4-2 三个 learned 1×1 adapters，
完整 SYSU test 一次：

```text
F1 = 0.8099 (ΔF1 = -0.0131)   IoU = 0.6805 (ΔIoU = -0.0187)   ref 82.30/69.92
```

判据（§7.1）：|ΔF1|<0.15 且 |ΔIoU|<0.25 → **adapter 功能较弱，显著削弱假设①**。
（GPT 把①排第一，实测被否：learned adapter 的 channel remapping 不是关键。）

## 4. R5-D0-C（H3）：FI 接口 geometry —— inconclusive + 重大附加发现

比值判据失效原因：R4-1（base）的 FI 在 1/2、1/4 尺度激活近零，比值爆炸。
改判为 inconclusive（base≈0 的尺度不计入强支持条件）。但 hook 与权重核对
揭示了一个比 interface mismatch 更重要的现象——**旧 FI 训练的零权重吸收**：

| 模型 | c2_c5 (1/2) | c3_c5 (1/4) | c4_c5 (1/8) |
|---|---|---|---|
| R4-1 | q 权重范数 **0.00000**，kv 0.035 | norm2/q/kv **全部 0.00000** | norm2 8.36 / q 2.73 / kv 6.35（活跃） |
| R4-2 | 全部活跃 | **q = 0.00000**（kv 活跃） | **q = 0.00000**（kv 活跃） |
| R4-2d | 全部活跃 | 全部活跃 | 全部活跃 |

即：F1 82.77 的 R4-1 参考模型里，FI 实际只用了 1/8 一路 detail 注入
（1/2、1/4 的 cross-attention 权重被训练成精确零——与 ViT 崩溃同类的 Adam
零权重吸收态）；R4-2 只用了 1/2 一路；R4-2d 三路全活但 F1 最低（82.02）。
**这与「旧 FI 三路注入都不可或缺」的直觉相反，且说明 legacy head 下的 detail
比较（R4-1 vs R4-2/2d）并非严格同接口对照**——该发现进入研究记录，供论文
消融讨论与 Run6 设计参考，本轮不为它开任何新训练。

## 5. 停止执行内容（§12 + §48，严格遵守）

- ❌ 不训练 MobileDetail（R5-1 / R5-2 / R5-3 全不启动）；
- ❌ 不再试 MobileNet features0-4/0-5、MobileNetV2/EfficientNet/ShuffleNet prefix；
- ❌ 不做 Mobile 冻结/小 LR、SGDP width/gate 变体、concat/add sweep、aux loss。

## 6. Run5 净收益（留存资产）

1. **H2/H1 两个归因假设被明确裁决**：PSD 失败更可能是 stem 被重写（H2 强支持），
   不是 adapter 接口问题（H1 被否）——Run4 的 4 条假设现在有 2 条有直接证据。
2. **旧 FI 零权重吸收发现**（§4 表）：任何未来使用 legacy FI/decoder 的对照实验
   都必须先查各尺度 q/kv 范数，否则"三路注入"是假前提。
3. **代码资产**：`models/model/mobile_detail.py`、`models/model/sgdp_head.py`
   （smoke 全绿，SGDP 参数 115,267、FLOPs 1.68G 达标）——Run6 若采用
   change-evidence head 设计可直接复用；`analyse/run5_postmortem_and_mobile_audit.py`
   （H1/H2/H3 + 任意新 detail 候选的 raw gate 可复跑）。
4. **MobileDetail 数据点**：10.5K 参数的完整 ImageNet 预训练前缀，raw ranking
   达 ResNet 的 77-81%——为 Run6 的「用成熟预训练子层」类设计提供了标定。

## 7. Run6 决策叉（需重新预注册，建议再带 GPT）

Run5 按规则收束后，证据链为：

```text
R4-2/2b/2d 三个自定义轻量 detail：FAIL（detail gate 均未过）
R5-D0 Mobile 成熟预训练 prefix：raw gate FAIL（1/4 差 0.017）
→ 保留 ViT4 + 独立 detail 的 0.22M 外围分配：两次证伪
→ Run6 唯一方向（§12/§48 预注册）：ViT semantic 参数预算重分配
```

候选（不预注册不执行）：① ViT4→ViT3/更窄 embed（腾出预算给 detail/head）；
② 去掉独立 detail、让 shallow ViT patch feature + 轻量 head 独挑（可用 R5-D0
工具先做 token 级 raw gate）；③ 重新选择 semantic 源（如 MobileNetV3 更深层
语义 + 极浅 ViT）。Run6 前同样先做零训练 raw gate 再决定 80K。

## 8. 立即执行顺序（已完成）

- [x] 准备官方 MobileNetV3-Small 权重（10,488 参数，features0-3）
- [x] 实现 mobile_detail.py / sgdp_head.py + T-R5 smoke（全绿）
- [x] R5-D0 完整审计（H1/H2/H3/D，GPU1，完整 test）
- [x] [MOBILE-GATE] FAIL → 按规则不启动任何 80K
- [x] 本记录 + README + 快照 + 推送 GitHub
