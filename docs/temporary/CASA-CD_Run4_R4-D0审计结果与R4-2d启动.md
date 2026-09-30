# CASA-CD Run4：R4-D0 Detail Interface Audit 结果与 R4-2d 启动记录

> **日期**：2026-09-30（R4-2/R4-2b gate 失败后的下一步，按
> `CASA-CD_Run4_R4-2失败后_下一步决策与PSD-Detail方案.md` §2-4 执行）
> **审查基准**：本地 repo = GitHub `YuqiWang-code/CASA-CD` main（父提交 `ec2f447`）
> **固定协议**：BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、
> seed 16、test-as-val、GPU1、ViT frozen。

---

## 0. 结论先行

R4-D0（零训练成本、完整 SYSU test 4000 对、只用 GPU1 前向）已跑完。
按预注册决策规则（§4）：

```text
A（adapter mismatch）支持尺度数 = 0/3
B（representation/pretraining/topology）支持尺度数 = 1/3
→ 判定 C（audit 模糊）
→ 预注册决策：优先 R4-2d PSD_DETAIL
```

**同时 C 的实际证据方向与 B 一致**：LightDetail 的 1/4、1/8 深层 raw feature
的变化判别能力已经崩坏（PR-AUC 0.32-0.33，仅略高于随机基率 ~0.19，ResNet 同尺度
0.63-0.65），adapter 前后 PR-AUC 变化 ≤0.03 —— adapter 修补不可能挽回，容量
bump（48 vs 32）也远不足以缩小与 ResNet 的差距。**adapter 路线（R4-2c）永久停止，
直接实现并训练 PSD-Detail（R4-2d）。**

---

## 1. Primary 指标（per-scale，pooled 16×16，s = 1−cos(f1,f2)）

| source | scale | PR-AUC | Spearman | Top32 prec | Top32 coverage |
|---|---:|---:|---:|---:|---:|
| R4-1 resnet_direct | 1/2 | 0.5467 | +0.4393 | 0.5498 | 0.3980 |
| R4-1 resnet_direct | 1/4 | 0.6263 | +0.4502 | 0.5710 | 0.4055 |
| R4-1 resnet_direct | 1/8 | **0.6535** | +0.4055 | **0.5948** | **0.4259** |
| R4-2 light32_raw | 1/2 | **0.5546** | +0.3934 | 0.5030 | 0.3470 |
| R4-2 light32_raw | 1/4 | 0.3251 | +0.2358 | 0.3655 | 0.2002 |
| R4-2 light32_raw | 1/8 | 0.3173 | +0.1021 | 0.3365 | 0.1660 |
| R4-2 light32_adapted | 1/2 | 0.5604 | +0.4199 | 0.5142 | 0.3590 |
| R4-2 light32_adapted | 1/4 | 0.3080 | +0.1972 | 0.3280 | 0.1722 |
| R4-2 light32_adapted | 1/8 | 0.2980 | +0.0433 | 0.3207 | 0.1526 |
| R4-2b light48_raw | 1/2 | 0.5229 | +0.3540 | 0.4919 | 0.3338 |
| R4-2b light48_raw | 1/4 | 0.3965 | +0.2738 | 0.4023 | 0.2390 |
| R4-2b light48_raw | 1/8 | 0.3282 | +0.0601 | 0.3546 | 0.1961 |
| R4-2b light48_adapted | 1/2 | 0.5526 | +0.3591 | 0.4984 | 0.3417 |
| R4-2b light48_adapted | 1/4 | 0.3530 | +0.2329 | 0.3660 | 0.2073 |
| R4-2b light48_adapted | 1/8 | 0.3347 | +0.0497 | 0.3584 | 0.2012 |

## 2. Feature-stat（per-scale，pooled 16×16）

| source | scale | mean | std | L2norm | zero% | neg% | cos(T1,T2) | \|dF\|/\|F\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4-1 resnet_direct | 1/2 | +0.341 | 0.219 | 3.228 | 0.00 | 0.00 | 0.9264 | 0.1683 |
| R4-1 resnet_direct | 1/4 | +0.131 | 0.139 | 2.130 | 0.00 | 0.00 | 0.7373 | 0.3629 |
| R4-1 resnet_direct | 1/8 | +0.106 | 0.169 | 3.108 | 0.00 | 0.00 | 0.6066 | 0.4952 |
| R4-2 light32_raw | 1/2 | +0.332 | 0.326 | 2.537 | 0.00 | 0.00 | 0.7170 | 0.3775 |
| R4-2 light32_raw | 1/4 | +0.213 | 0.211 | 2.292 | 0.00 | 0.00 | 0.6594 | 0.4163 |
| R4-2 light32_raw | 1/8 | +0.114 | 0.200 | 2.272 | 0.00 | 0.00 | 0.5211 | 0.5843 |
| R4-2 light32_adapted | 1/2 | −0.002 | 0.201 | 1.557 | 0.00 | 50.87 | 0.6317 | 0.4157 |
| R4-2 light32_adapted | 1/4 | +0.001 | 0.103 | 1.111 | 0.00 | 52.98 | 0.6739 | 0.4320 |
| R4-2 light32_adapted | 1/8 | +0.003 | 0.089 | 1.243 | 0.00 | 43.99 | 0.6342 | 0.4739 |
| R4-2b light48_raw | 1/2 | +0.338 | 0.330 | 3.181 | 0.00 | 0.00 | 0.7216 | 0.3743 |
| R4-2b light48_raw | 1/4 | +0.189 | 0.209 | 2.591 | 0.00 | 0.00 | 0.6557 | 0.4318 |
| R4-2b light48_raw | 1/8 | +0.133 | 0.227 | 2.911 | 0.00 | 0.00 | 0.5366 | 0.5766 |
| R4-2b light48_adapted | 1/2 | +0.022 | 0.226 | 1.742 | 0.00 | 44.56 | 0.6084 | 0.4360 |
| R4-2b light48_adapted | 1/4 | −0.002 | 0.132 | 1.398 | 0.00 | 49.63 | 0.7414 | 0.3870 |
| R4-2b light48_adapted | 1/8 | +0.001 | 0.137 | 1.887 | 0.00 | 59.52 | 0.5931 | 0.4903 |

## 3. 决策规则核对（§4）

- **情况 A（adapter mismatch）**：要求 ≥2/3 尺度 PR(Light raw) ≥ PR(ResNet)−0.02
  且 PR(Light adapted) ≤ PR(Light raw)−0.03 且 Spearman 同方向下降。
  实际：只有 1/2 尺度 raw 达标（0.5546 ≥ 0.5267），但该尺度 adapter 反而 +0.006
  → **A = 0/3，排除**。
- **情况 B（representation/topology）**：要求 ≥2/3 尺度 PR(Light raw) ≤ PR(ResNet)−0.04
  且 Light48 对 Light32 改善 <0.02。
  实际：1/4、1/8 尺度 PR gap 高达 −0.30 / −0.34（远超 −0.04 门槛）；
  但 1/4 的 width 改善 +0.0714 ≥ 0.02（Light48 在 1/4 仍只到 0.3965，远低于
  ResNet 0.6263）→ **B = 1/3**（仅 1/8 同时满足两个子条件）。
- **判定 = C（模糊）** → 预注册决策：**优先 R4-2d PSD_DETAIL**。

## 4. 证据解读（人工 gate 读判）

1. **浅层（1/2）不是瓶颈**：Light raw 0.5546 ≥ ResNet 0.5467 —— 随机初始化的
   浅层 DWConv 已经够用。
2. **深层（1/4、1/8）raw feature 崩坏**：PR-AUC 0.32 量级（随机基率 ~0.19），
   与 ResNet 同尺度差 ~0.30 —— 这才是 R4-2 掉 −0.47 F1 的 feature 级证据。
3. **adapter 不是主因**：adapter 前后 PR 变化 ≤0.03，Spearman 微降但不构成
   「raw 好、adapter 毁」的 A 型证据；feature-stat 显示 adapter 把非负特征变成
   零均值、~50% 负值（接口统计变化确实存在，但 ranking 能力没被显著破坏）。
4. **width 不是主因**：48 通道在 1/4 有 +0.07 改善，但离 ResNet 仍差 0.23；
   1/8 几乎无改善（+0.01）。
5. **结论**：LightDetail 失败 = 深层尺度的「随机初始化 + 无 residual +
   先 DW 降采样后 PW 混合」表达力不足，pretrained-stem + residual + MixDown
   顺序反转的 PSD-Detail 是对症设计。

## 5. 后续执行（按修订方案）

```text
R4-D0 → 判定 C → R4-2d PSD_DETAIL（唯一变量 = detail branch）
  - PSD-Detail 0.078M：ImageNet ResNet18 conv1/bn1 原位复制 stem
    + ResidualDS(64) + MixDown(64→128) + ResidualDS(128) + MixDown(128→256)
  - 输出 64/128/256，与 FI 直接兼容、无 adapters；ViT4 frozen、原 FI/decoder、80K、seed16
  - Gate 不降低：F1 ≥ 82.47 且 IoU ≥ 70.11（相对 R4-1 82.77 / 70.61）
  - 不过 → 停止当前轻量 detail 路线，不启动 SABI/DFPD
```

- 实现：`models/model/psd_detail.py`（78,464 参数，smoke T-PSD0..T-PSD5 全过）
- audit 代码：`analyse/run4_detail_interface_audit.py`；脚本：
  `train_scripts/UltraLight/Run4/audit_R4_D0_detail_interface_SYSU.sh`、
  `train_R4_2d_PSD_DETAIL_SYSU.sh`、`train_R4_2c_ADAPTER_ALIGN_SYSU.sh`（已永久停用）
- 完整报告：服务器 `/home/yqwang/outputs/CASA-CD/UltraLight/Run4/R4_D0_DETAIL_AUDIT/audit_report.txt`

---

## 6. 后续结果（2026-09-30 补记）

R4-2d PSD_DETAIL 已按上文执行完毕：**F1 82.02 / IoU 69.52，gate FAIL
（ΔF1 −0.75 vs R4-1）**——PSD 反而比 LightDetail（82.30）低 0.28。按预注册
Stop-3 **轻量 detail 路线终止**（不启动 SABI/DFPD）。完整结果、训练证据与
Run5 方向候选见
[`CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`](CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md)。
