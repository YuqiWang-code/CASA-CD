## GitHub 更新（手动操作）

代码或文档更新后，手动同步到 GitHub（提交信息统一用 `update code`）：

```bash
cd f:/Code_Repositories_2/CursorCode/CASA-CD
git add models/ train_scripts/ analyse/ docs/ others/ README.md .gitignore
git commit -m "update code"
git push origin main
```

- 预训练权重 `pretrained_weight/*.pth` 不进 git，更新时上传到 GitHub Releases
  （仓库页 → Releases → Draft a new release → 上传附件 `.pth`）。
- `docs/参考文献/` 的调研 PDF 随 `docs/` 一并提交（`.gitignore` 未忽略 PDF）。

---

# CASA-CD

Change-Aware Selective Aggregation Network for ultra-lightweight fully-supervised
binary change detection in remote sensing images.

硕士课题：极轻量遥感二值变化检测中的变化感知非对称 Token 建模。路线记录见
[`docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md`](docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md)，
服务器与数据规范见
[`docs/RSML-3_服务器环境与变化检测数据统一说明.md`](docs/RSML-3_服务器环境与变化检测数据统一说明.md)。

## 研究定位与约定

- **方法创新导向**：这是研究生论文课题，核心是方法创新（变化感知非对称 token 建模 + 极轻量结构），
  不做工程化堆叠，也不把 loss 调参 / 训练技巧包装成创新贡献。
- **单 seed**：当前阶段只用单 seed 验证有效性与创新性，不做多 seed 统计显著；如需论文级结果，再按需补充。
- **训练协议**：沿用 ChangeViT 官方协议（BCE+Dice、poly LR、max_steps=80000、seed 16），
  **test 集当验证集、每 epoch 在 test 上挑 best**，与本实验室其它 CD 项目一致。

## 方法

- **Baseline：ChangeViT-Tiny**（Pattern Recognition 2025）：Plain ViT（DeiT-Tiny 预训练）+
  ResNet18 detail-capture branch + Feature Injector + Decoder。
- **参数量口径**：官方代码的 ResNet18 含未参与前向的 `layer4+fc`（死参数 ~8.9M），
  我们复现保持官方代码原样，日志报总参数 **20.66M** / FLOPs **26.32G**；
  论文表格的 11.68M 是按有效参数统计（ViT 5.54M + ResNet≤layer3 2.78M + Decoder 3.44M ≈ 11.76M），
  两者前向计算完全一致，FLOPs 吻合（26.32G ≈ 论文 27.15G）。CASA-CD 轻量化时再删死参数。
- **CASA-CD 计划（baseline 复现后推进）**：
  - **CASAA**（Change-Aware Asymmetric Token Modeling，灵感来自 SAT, CVPR 2026）：
    完整保留 Query（逐位置判别能力不变），只压缩提供上下文的 K/V；疑似变化 token 保留、
    稳定背景 token 强聚合，注意力交互量 `O(N²) → O(NK), K≪N`。已实现为 Paired
    Late-Stage CASAA（ViT blocks 8-11，K=64=Kc32 直保留+Kb32 背景聚类，**零新增参数**）。
    **终局（Run1-3，2026-09-29）**：change-aware router 按预注册停止规则终止——
    Oracle（GT 路由）证明机制上限 SYSU +1.48，但 cosine / detail / rank-fused 三个可部署
    信号都无法转化为 F1 收益（detail-only 相对 A1 −0.12）；留存结论：冻结 ViT 下
    Full-Q + K=64 content 压缩基本无损（A1），论文中降为 analysis/ablation。
  - **Ultra-Light Multi-Scale Change Representation（主线二，Run5 已按规则收束）**：
    目标 `<3M` 有效推理参数（工程目标 ≤2.20M）。Run4（Light32/Light48/PSD 三个自定义
    轻量 detail）按 Stop-3 终止；Run5（MobileDetail-P3 + SGDP，≈2.103M 设计）的
    R5-D0 无训练 raw gate **FAIL**（D4 PR-AUC 0.4831 < 0.50）→ 按预注册规则
    候选② 永久停止，未消耗任何 80K。下一步 **Run6 = ViT semantic 预算重分配**
    （需重新预注册）。方案与结果见
    [`docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md`](docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md)、
    [`docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md`](docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md)。

- 模型入口：`models/train.py`（训练）、`models/eval.py`（独立测试）、`models/smoke_test.py`（冒烟）
- 网络定义：`models/model/`（encoder / decoder / layers / resnet，上游 ChangeViT 微调）
- 损失 `BCE + Dice`；`max_steps=80000`，batch 16，256×256，seed 16，lr 2e-4（poly）
- 日志：开头输出全部配置 + 总参数量 + 可训练参数量 + FLOPs(G)；每 epoch 一行
  （Loss + Recall/Precision/OA/F1/IoU/Kappa）；结尾 `=== TEST RESULTS ===` 正式测试区块

## 实验结果（Baseline Run1）

> ChangeViT-T（官方协议：BCE+Dice、poly LR、max_steps=80000、batch 16、seed 16，加载
> `deit_tiny_patch16_224-a1311bcf.pth`）在 4 数据集的复现，**全部完成**。

| 数据集 | ChangeViT-T 官方 (F1) | 复现 F1 | IoU | OA | Kappa | 说明 |
|---|---|---|---|---|---|---|
| CDD | — | **97.75** | 95.60 | 99.44 | 97.43 | 官方未评测 CDD |
| LEVIR | 91.81 | **91.95** | 85.10 | 99.18 | 91.52 | 建筑小目标、极不平衡 |
| SYSU | — | **82.48** | 70.19 | 91.91 | 77.23 | 官方未评测 SYSU |
| WHU | 94.53 | **94.84** | 90.18 | 99.60 | 94.63 | 建筑小目标、极不平衡 |

- 官方评测过的两个数据集复现均略高于论文：LEVIR 91.95 vs 91.81（+0.14，IoU 85.10 vs 84.86）、
  WHU 94.84 vs 94.53（+0.31，IoU 90.18 vs 89.63）——复现成立（差异主要来自 label 阈值
  gray≥128 与数据集版本）。
- 复杂度：TOTAL 20.661M（含死参数）/ **EFFECTIVE 11.754M**（论文口径 11.68M）/ FLOPs 26.32G（论文 27.15G）。
- SYSU 运行期间与 STR-RepNet 的新任务挤 GPU0，触发 196 次 OOM 自动断点续训（协议未变，
  每个 epoch 均为完整训练，崩溃只丢半截 epoch），最后在 GPU1 上跑完。该结果为官方协议下
  的有效结果；如需更干净对照，可在空卡上重跑 SYSU。

## 实验结果（CASAA Run1）

> Paired Late-Stage CASAA：ChangeViT-T 最后 4 个 ViT block（0-based 8-11）换成
> 变化感知非对称注意力——**Full Q（N=256，逐位置判别不变）+ 压缩 K/V（K=64）**：
> A2 主方法 `router=change`（Kc=32 change-score TopK 直保留 + Kb=32 共享背景聚类），
> A1 对照 `router=content`（K=64 纯内容聚类，SAA-style）。routing 参数自由、
> 确定性、T1/T2 对称；qkv/proj 原位继承 DeiT-Tiny 预训练，**零新增参数**；
> 训练协议与 baseline 完全一致（BCE+Dice、poly、80000 steps、batch 16、seed 16）。
> 实现：`models/model/layers/casaa.py`；脚本：`train_scripts/CASAA/Run1/`。
> 实验设计与判据：`docs/temporary/CASA-CD_CASAA_Run1_修改与实验设计建议.md`。

| 变体 | LEVIR F1 | LEVIR IoU | SYSU F1 | SYSU IoU | 说明 |
|---|---:|---:|---:|---:|---|
| baseline（已有） | **91.95** | 85.10 | 82.48 | 70.19 | ChangeViT-T 复现 |
| A1 SAA-style（对照） | 91.94 | 85.08 | **82.50** | 70.21 | K=64 纯内容聚类，无变化感知 |
| A2 CASAA（主方法） | 91.86 | 84.94 | 82.35 | 70.00 | Kc=32 直保留 + Kb=32 背景聚类 |

- 复杂度：参数 11.754M 不变（零新增）；FLOPs 26.2593G（baseline 26.3246G）；
  4 个 run 均 0 retry、无 OOM。完整日志：`outputs/CASAA/Run1/`。
- **机制结论**：
  1. **A1 ≈ baseline**（LEVIR −0.01 / SYSU +0.02）→ 晚阶段把 K/V 压到 25% 基本无损，
     「压缩冗余上下文」成立；
  2. **A2 ≈ A1 且略低**（−0.08 / −0.15）→ 参数自由的 cosine 变化路由未带来可测的
     额外收益，change-aware 的机制价值在 Run1 设定下未被证明（单 seed，±0.15 属噪声量级）。
- **筛选判据（设计文档 §16）未通过**：A2 未优于 baseline，也未优于 A1，故按预注册
  规则**未启动** CDD/WHU 补全。下一步候选（§17 预设路径）：keep_ratio 0.5 /
  只改最后 2 个 block / adaptive change quota / 更换更敏感的 change 信号。

- **⚠️ Run1 重要勘误（2026-09-28 发现）**：官方协议（统一 lr=2e-4）会把 ViT 在
  ~1600 steps 内训练成**精确零权重**（Adam 小梯度全步长 + 归零后梯度消失的吸收态）。
  核验全部 Run1 checkpoint：LEVIR/CDD/WHU 的 ViT 150/150 键全零（optimizer 矩也全零），
  SYSU 仅 best 检查点健康——**Run1 的 LEVIR 数字全部来自「死 ViT」模型，不检验
  CASAA**；SYSU 的比较仅在 best（epoch 9）附近有效。Run2 起全部改为
  `--freeze_vit 1`（冻结 ViT）重做机制实验。详见
  [`docs/temporary/CASA-CD_ViT崩溃发现与Run2修订.md`](docs/temporary/CASA-CD_ViT崩溃发现与Run2修订.md)。

## 实验结果（CASAA Run2）

> 冻结 ViT 的 Oracle 机制诊断（A3，DIAGNOSTIC-ONLY，GT patch occupancy 路由）。
> 背景：Run1 后证实官方协议会把 ViT 训练成零权重（见上节勘误与
> `docs/temporary/CASA-CD_ViT崩溃发现与Run2修订.md`），Run2 起全部 `--freeze_vit 1`，
> 唯一变量 = router。实现：`models/model/layers/casaa.py`（qkv 切片投影 + oracle
> 路由 + 诊断统计）、`analyse/casaa_router_diagnostic.py`（Router Audit）；
> 脚本：`train_scripts/CASAA/Run2/`。

| 变体 | LEVIR F1 / IoU | SYSU F1 / IoU | 说明 |
|---|---:|---:|---|
| baseline Run1（训练 ViT） | 91.95 / 85.10 | 82.48 / 70.19 | LEVIR 死 ViT，SYSU best 有效 |
| A1 content-only（冻结 ViT） | 91.84 / 84.91 | 82.04 / 69.55 | Run2 有效对照 |
| **A3 Oracle（冻结 ViT，诊断）** | 91.88 / 84.98 | **83.52 / 71.70** | GT occupancy 路由，不可部署 |

- **判据（决策文档 §4.5）：通过。** Oracle 相对 A1：SYSU **F1 +1.48**（门槛 +0.30）、
  IoU +2.15 同向；LEVIR +0.04（未降超 0.15）。冻结 ViT + Oracle（83.52）甚至超过
  完全训练的 baseline（82.48）——「知道变化在哪」比训练 ViT 本身更值钱。
- **收益分布**：只在 SYSU（密集变化，中位变化 patch 50）出现；LEVIR（54% 图像零变化）
  上 Oracle 几乎无增益——机制在「变化真正存在且多」时有效。
- **Router Audit（冻结健康 ViT）**：cosine 分数质量明显不足——Spearman 0.10-0.29、
  Top32 precision 0.13-0.39；SYSU 真实变化 patch 数 P10=11/P50=50/P90=182，
  固定 Kc=32 严重错配。→ 结论：**机制有价值，瓶颈在 deployable change signal**。
- **下一步（预注册分支 A）**：CASAA-v2 = Detail-guided parameter-free score
  （已有 1/8 detail 特征 32×32→AvgPool→16×16 的 `1−cos(d̄1,d̄2)` 与 ViT cosine 分数
  rank 归一化 1:1 融合，零新增参数/零新 loss），仍 K=64/Kc=32、冻结 ViT，
  `train_scripts/CASAA/Run3/`，LEVIR+SYSU。

## 实验结果（CASAA Run3）

> 可部署信号终局（冻结 ViT，Router Audit gate → detail-only 救援 run）。
> 实现：`router=detail_fused`（0.5R(ViT cosine)+0.5R(detail 1/8 cosine)）与
> `router=detail`（纯 detail）；detail 1/8 = resnet.layer3 32×32 → AvgPool → 16×16。
> 脚本：`train_scripts/CASAA/Run3/`；决策：`docs/temporary/CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md`。

- **Router Audit（SYSU）**：fused 未过 gate（PR-AUC +0.014 < 0.03、Spearman −0.052）；
  detail-only 明显优于 ViT cosine（PR-AUC +0.059、Top32 precision 0.39→**0.63**、
  coverage 0.24→**0.46**）→ 按决策树只做 1 个 SYSU detail-only 80K。
- **A4-D 最终（SYSU）**：F1 **81.92** vs A1 82.04（−0.12，判据 ≤82.19 → **失败**）。
- **完整证据链（SYSU，相对 A1）**：ranking Top32 precision 0.39 → −0.15；
  0.63 → −0.12；1.0（oracle）→ +1.48——可部署 ranking 提高 60% 未转化为 F1 收益，
  机制只在 ranking 接近完美时有效。→ **按预注册规则停止 change-aware router 迭代，
  转主线二**（见下节）。

## 实验结果（Run4 · 主线二极轻量结构）

> UL-V4：TinyViT4-192（DeiT prefix-4 原位继承 + pos_embed 14×14→16×16 插值，
> corrected loader）+ 逐组件轻量化。全部冻结 ViT、BCE+Dice、80000 steps、batch 16、
> seed 16、test-as-val；只用 GPU1；脚本：`train_scripts/UltraLight/Run4/`。

| Run（SYSU） | 结构变化 | F1 | IoU | Params | FLOPs | 判据 |
|---|---|---:|---:|---:|---:|---|
| R4-0 A0_FULL12_FROZEN | 健康 full12 冻结参考 | **83.14** | 71.14 | 11.754M | 26.32G | 参考 |
| R4-1 VIT4_OLDHEAD | depth 12→4 | 82.77 | 70.61 | 8.195M | 24.10G | **通过**（ΔF1 −0.37 ≥ −0.50） |
| R4-2 VIT4_LIGHTDETAIL | ResNet→LightDetail 32/64/128 | 82.30 | 69.92 | 5.492M | **10.73G** | 未过（ΔF1 −0.47 > −0.30） |
| R4-2b LIGHTDETAIL48 | 预注册容量 fallback 48/96/160 | 82.32 | 69.95 | 5.533M | 10.98G | 未过（ΔF1 −0.45） |
| R4-D0 DETAIL_AUDIT | 无训练 feature interface audit | — | — | — | — | 判定 **C** → 跳过 adapter，走 PSD |
| R4-2d PSD_DETAIL | PSD-Detail 0.078M（pretrained stem + residual DS） | 82.02 | 69.52 | 5.491M | 11.52G | **未过**（ΔF1 −0.75 > −0.30）→ Stop-3 |

- 关键事实：4-block 冻结 prefix（82.77）已超过 trained baseline（82.48）与全部
  CASAA 冻结模型；corrected loader 的健康 full12 参考 = **83.14**（新基线）。
- **R4-D0 审计结论（2026-09-30）**：用 R4-1/R4-2/R4-2b best checkpoint 在完整 SYSU
  test 上对比三尺度 detail feature 的变化判别能力（PR-AUC/Spearman/Top32）——
  浅层 1/2 尺度 Light raw ≥ ResNet（0.555 vs 0.547），但 1/4、1/8 深层 raw 崩坏
  （PR-AUC 0.32 量级 vs ResNet 0.63/0.65，仅略高于随机基率）；adapter 前后变化
  ≤0.03、width bump 远不足以弥补 → **不是 adapter 问题，是随机初始化 + 无 residual +
  先 DW 降采样后混合的表达力问题**。预注册规则 A=0/3、B=1/3 → 判定 C →
  按规则直接进入 R4-2d PSD_DETAIL（adapter 修补 R4-2c 永久停止）。详见
  [`docs/temporary/CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md`](docs/temporary/CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md)。
- **R4-2d PSD 最终结果（2026-09-30）**：F1 **82.02** / IoU 69.52（Recall 79.78 /
  Precision 84.39 / OA 91.75 / Kappa 76.68；5.491M / 11.52G），gate FAIL
  （ΔF1 −0.75 vs R4-1）——PSD 反而比随机初始化 LightDetail 低 0.28。训练证据排除
  欠训练（best@epoch64，末段无上升趋势）。**按预注册 Stop-3 终止轻量 detail 路线
  （不启动 SABI/DFPD），Run4 正式收束**；Run5 候选方向（预算重分配 / 成熟超轻
  pretrained 子层 / 去掉独立 detail 分支）需重新预注册。详见
  [`docs/temporary/CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`](docs/temporary/CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md)。
- Run4 留存正面资产：健康 full12 frozen 参考 83.14、depth-4 证据（12→4 block 仅
  −0.37 F1、−3.56M 参数）、R4-D0 无训练 feature 诊断方法（可复用于 Run5 筛选）。

## 实验结果（Run5 · 成熟预训练 MicroDetail + SGDP）

> Run5 方案：`docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md`；
> 脚本：`train_scripts/UltraLight/Run5/`。R5-D0 无训练 audit 已跑完 → **Mobile raw
> gate FAIL → 按预注册规则 Run5 候选② 永久停止（未启动任何 80K）**。结果与 Run6
> 决策叉见
> [`docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md`](docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md)。

| Run（SYSU） | 结构变化 | F1 | IoU | Params | FLOPs | 判据 |
|---|---:|---:|---:|---:|---:|---|
| R5-D0 MOBILE_AUDIT | MobileDetail-P3 raw gate（无训练） | — | — | — | — | **FAIL**（D4 PR 0.4831<0.50，其余 3/4 项过） |

- R5-D0 关键数字（完整 test 4000 对）：Mobile 1/4 PR-AUC 0.4831（门槛 0.50）、
  1/8 0.5317（0.52）、Top32 prec 0.5006/0.5105（0.46/0.48）——3/4 子门槛过、
  1/4 PR 差 0.017，按预注册规则**不放行**（这是 raw gate 省 80K 的设计目的）。
  MobileDetail 仍是迄今最强非 ResNet detail（1/4 比 Light48 高 0.087，达 ResNet 77%）。
- Postmortem 附加结论：① H2 强支持「PSD 失败主因是 pretrained stem 被重写」
  （PSD conv rel_L2 0.639 vs R4-1 0.211）；② H1 否定「adapter 关键」假设
  （TileAdapter ΔF1 仅 −0.013）；③ 发现旧 FI 的零权重吸收：R4-1 参考模型的 FI
  只有 1/8 一路注入活跃（1/2、1/4 权重精确归零）——legacy head 下的 detail 对照
  并非严格同接口比较，已记入研究记录。
- 代码：`models/model/mobile_detail.py`（10,488 参数，ImageNet 原位继承）、
  `models/model/sgdp_head.py`（115,267 参数，FLOPs 1.68G 达标，Run6 可复用）；
  审计：`analyse/run5_postmortem_and_mobile_audit.py`。
- **下一步（Run6，需重新预注册）**：ViT semantic 参数预算重分配（更浅/更窄 ViT
  腾预算、或去掉独立 detail、或换 semantic 源），先做零训练 raw gate 再决定 80K。

## 参考文献

- Baseline：`docs/参考文献/baseline/ChangeViT(PR2026).pdf`
  （Zhu et al., ChangeViT: Unleashing Plain Vision Transformers for Change Detection in
  Remote Sensing Images, Pattern Recognition 2025；代码 https://github.com/zhuduowang/ChangeViT）
- 创新点来源：`docs/参考文献/baseline/SAT(CVPR2026).pdf`
  （SAT: Selective Aggregation Transformer for Image Super-Resolution；
  arXiv:2604.07994；https://github.com/PhuTran1005/SAT）
- SAT 核心机制提取（SAA + 聚类压缩 K/V，去框架化，供 CASAA 直接复用）：
  [`others/SAT/saa.py`](others/SAT/saa.py)，说明见 [`others/SAT/README.md`](others/SAT/README.md)

## 目录结构

```
models/                        # 全部代码（ChangeViT 上游 + 本仓库改造）
  train.py                     #   训练入口（baseline / --mode casaa|saa / --vit_depth / --detail_mode）
  eval.py                      #   独立测试入口（同款架构参数）
  smoke_test.py                #   冒烟测试（baseline 等价 + CASAA 机制 + Run4 深度/轻量测试）
  main.py                      #   上游原版（仅参考）
  model/                       #   encoder / decoder / trainer / layers / resnet
  model/layers/casaa.py        #   CASAA 核心（change score + 确定性聚类 + 非对称注意力）
  model/light_detail.py        #   Run4 LightDetail（32/64/128 与 48/96/160 DSConv）
  model/psd_detail.py          #   Run4 PSD-Detail（pretrained stem + residual DS pyramid，0.078M）
  dataset/                     #   DataLoader（A/B/label + list 格式）
train_scripts/
  baseline/Run1/               # ChangeViT-T baseline 启动脚本（4 数据集 + 串行队列）
  CASAA/Run1/                  # CASAA Run1（A1/A2 × 4 数据集 + 双卡/单卡串行队列）
  CASAA/Run2/                  # CASAA Run2（冻结 ViT：A1 对照 + A3 Oracle 诊断）
  CASAA/Run3/                  # CASAA Run3（A4 detail 可部署信号终局）
  UltraLight/Run4/             # 主线二：R4-0/1/2/2b 逐组件单变量（阶段 gate）
analyse/                       # 分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标 → docs/temporary/*.txt
  casaa_router_diagnostic.py   #   Router Audit（V/D/F 三路 score 对齐 GT）
  run4_detail_interface_audit.py  #  R4-D0：ResNet vs Light raw/adapted 三尺度对齐 GT
  param_breakdown.py           #   Run4 U1：组件级参数预算
  vit_pretrain_audit.py        #   Run4 U2：corrected DeiT loader 原位继承审计
others/                        # 参考实现（非本仓库模型代码）
  SAT/                         #   SAT(CVPR2026) 核心机制提取：saa.py（SAA + 聚类压缩）
outputs/                       # 训练日志（训练结束后下载到这里）
docs/                          # 项目文档（temporary / 参考文献 / 服务器说明）
.claude/                       # 服务器部署 skill 与 SSH 辅助脚本（不进 git）
```

## 服务器环境（RSML-3）

- 环境 `casacd`（clone 自 `cd_base`）：**torch 2.14.0+cu132 / Python 3.10 / CUDA 13.2**，
  补装 `einops opencv-python fvcore`。
- GPU：2 × RTX 5090（Blackwell `sm_120`，每卡 32 GB），PCIe 无 NVLink。
- 关键路径：
  - 代码 `/home/yqwang/projects/CASA-CD/`
  - 数据集 `/share_datasets/CD/{CDD,LEVIR,SYSU,WHU}-CD-256/`
  - 预训练权重 `/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth`
  - checkpoint `/share_datasets/yqwang/checkpoints/CASA-CD/baseline/Run1/<dataset>/`
  - 训练日志 `/home/yqwang/outputs/CASA-CD/baseline/Run1/<dataset>/train_log.txt`

## 数据集（A/B/label + list 格式）

| 数据集 | Train / Val / Test | 目标 F1 | 说明 |
|---|---:|---:|---|
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | ≈97 | 抗伪变化，label 为 JPG（阈值 ≥128） |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | ≈92 | 建筑小目标、极不平衡 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | ≈84 | 通用地表变化 |
| WHU-CD-256 | 5,947 / 743 / 744 | ≈95 | 建筑小目标、极不平衡 |

## 训练 / 测试

1. 本地改代码 → `python .claude/_deploy.py` 同步到服务器。
2. 服务器启动（`nohup`，每脚本带断点续训重试循环），脚本在 `train_scripts/baseline/Run1/`：
   ```bash
   cd /home/yqwang/projects/CASA-CD/train_scripts/baseline/Run1
   nohup bash run_queue.sh > /dev/null 2>&1 &   # CDD → LEVIR → SYSU → WHU 串行
   ```
   CASAA 实验脚本在 `train_scripts/CASAA/Run1-3/`（`run_screen.sh` 双卡并行 /
   `run_screen_gpu1_serial.sh` 单卡串行，`--mode casaa|saa`，详见各目录 README）；
   主线二脚本在 `train_scripts/UltraLight/Run4/`（逐组件 gate，只用 GPU1）。
   **注意必须串行**：ChangeViT-T batch 16 单任务 ~15.7GB（FeatureInjector 对 c2 全 token
   交叉注意力 ~8.6GB 注意力矩阵），两个任务并跑会超过单张 5090 的 32GB 导致 OOM。
3. 训练日志格式：
   - 开头：全部配置 + 总参数量 + 有效参数量（论文口径）+ 可训练参数量 + FLOPs(G)。
   - 每个 epoch 一行：Loss + Recall / Precision / OA / F1 / IoU / Kappa 六项指标（test 集）。
   - 结尾：`=== TEST RESULTS ===` 参数量 + FLOPs + 六项指标（best checkpoint 正式测试）。
4. checkpoint：每 run 一个文件夹，只放 `last.pth`（断点续训，含优化器）与 `best_F1=xxx.pth`（测试用）。
5. 训练结束后把 `train_log.txt` 下载回本地 `outputs/`（`baseline/Run1/`），权重不下载。

## 运行监控 / 分析

```bash
python .claude/_monitor.py              # 查看 baseline 训练进度 + GPU 占用
python .claude/_monitor_casaa.py        # 查看 CASAA/Run1 训练进度 + GPU 占用
python .claude/_ssh.py '<cmd>'          # 通用 SSH 执行
python analyse/extract_metrics_to_excel.py   # outputs → docs/experiment_metrics.xlsx
python analyse/models_to_txt.py --tag baseline --run Run1  # models 快照 + 指标 → docs/temporary/
python analyse/models_to_txt.py --tag CASAA --run Run1
python analyse/models_to_txt.py --tag CASAA --run Run2
python analyse/models_to_txt.py --tag UltraLight --run Run4
```

## 注意事项

- ChangeViT 官方协议在 epoch 0 后跳过一次评估（原代码行为），复现沿用。
- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（torch 2.6+ 默认
  `weights_only=True` 会拒收非张量对象），train.py 已处理。
- `.sh` 脚本需 LF 行尾（Windows 编辑后 `_deploy.py` 上传，本地已确认 LF）。
- ChangeViT 依赖 xformers 是可选的：缺失时代码自动回退到原生 attention。
