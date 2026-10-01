# CASA-CD 下一步决策 Prompt（Run6：ViT 预算重分配，带去网页 GPT，附 GitHub 链接）

> 仓库：https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新，HEAD=`dc47540`）
> 用法：把下面整段直接粘贴给网页 GPT。让它先读仓库里的指定文档再回答。

---

我是硕士课题「极轻量遥感二值变化检测」的负责人。仓库在
https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新）。

请先读仓库里这几份文档（按顺序），它们是我项目的事实权威：
1. `README.md`（全部结果表 + 约定）
2. `docs/temporary/CASA-CD_交接文档_2026-09-30.md`（上下文交接）
3. `docs/temporary/CASA-CD_Run5_成熟预训练MicroDetail_SGDP可执行方案.md`（上一轮你的方案，已执行完）
4. `docs/temporary/CASA-CD_Run5_R5-D0结果与Run5停止.md`（**本轮最重要：R5-D0 审计结果 + Run5 停止记录 + Run6 决策叉**）
5. `docs/temporary/CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`（Run4 终局）
6. `docs/temporary/CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md`（R4-D0 audit）
7. `docs/temporary/过去的想法/CASA-CD_Run4_极轻量结构主线_可执行方案.md`（Run4 原始设计）
8. `train_scripts/UltraLight/Run5/README.md`（Run5 gate 定义）
代码重点：`models/model/encoder.py`、`mobile_detail.py`、`sgdp_head.py`、`trainer.py`、
`analyse/run5_postmortem_and_mobile_audit.py`、`analyse/run4_detail_interface_audit.py`。

## 一句话现状

目标：二值变化检测，4 数据集（CDD/LEVIR/SYSU/WHU，256×256，label 阈值 gray≥128），
最终模型有效推理参数 **<3M（工程目标 ≤2.20M）**，在 2M 量级超过轻量 SOTA
（RFANet 2.86M / SeCoR 2.50M / Lighter 1.10M / CGLNet 0.99M）。
已走完两条路线并按预注册规则终止：CASAA change-aware router（Run1-3）；
轻量 detail 路线（Run4 三个自定义 detail + Run5 MobileNet 预训练 prefix，全部
raw/端到端 gate 失败）。**Run5 已按 §12 收束，预注册规定下一轮 Run6 只讨论
「ViT semantic 参数预算重分配」，不再试任何新的 detail branch。**
现在需要你给出 Run6 的可执行、可预注册方案。

## 必须遵守的硬约束（不要违反）

- 方法创新导向；**不改 loss、不把训练技巧当创新**；单 seed=16（不做多 seed）。
- 每个实验**唯一变量 + 预注册 gate**，gate 不过不进入下一步；不做 sweep。
- 训练协议固定：BCE+Dice、Adam(lr=2e-4, betas=(0.9,0.99), wd=1e-4)、poly(power 0.9)
  +200 iter warmup、max_steps=80000、batch 16、256×256、seed 16、test-as-val
  （每 epoch 在 test 上挑 best）；ViT 一律冻结（解冻需先过 2K health gate，
  且只允许一次 vit_lr=2e-5 的 80K）。
- 服务器只有 GPU1 可用；SYSU 一个 80K 约 3-3.5h、LEVIR ~4.4h——**正式 80K 总数
  ≤3-4 个就要到定稿**；任何新候选必须先过**零训练 raw gate**（我有现成工具，
  见下）才允许花 80K。
- 参数量口径：forward 图内全部参数（冻结照计、死参数不计）；不把量化/剪枝/蒸馏
  当核心贡献。

## 完整证据链（全部数字最新，可直接引用）

### 全部正式结果（SYSU，F1 / IoU，test-as-val）

| Run | 结构 | F1 | IoU | Params | FLOPs | 判据 |
|---|---:|---:|---:|---:|---|
| baseline Run1 | ChangeViT-T 官方协议（ViT 可训练，后证实部分死 ViT） | 82.48 | 70.19 | 11.754M | 26.32G | 历史参考 |
| R4-0 A0_FULL12_FROZEN | 健康 full12 冻结 + corrected DeiT loader | **83.14** | 71.14 | 11.754M | 26.32G | 新健康参考 |
| R4-1 VIT4_OLDHEAD | ViT depth 12→4，其余原样（ResNet detail + 旧 FI/decoder） | 82.77 | 70.61 | 8.195M | 24.10G | **通过**（ΔF1 −0.37） |
| R4-2 / R4-2b | ResNet→LightDetail 32/64/128（48/96/160）+ adapters | 82.30 / 82.32 | 69.92 / 69.95 | 5.49M | 10.7-11.0G | 未过（−0.47/−0.45） |
| R4-2d | ResNet→PSD-Detail 0.078M（pretrained stem + residual DS） | 82.02 | 69.52 | 5.491M | 11.52G | 未过（−0.75） |
| R5-D0 | MobileDetail raw gate（无训练） | — | — | — | — | **FAIL**（见下） |

（LEVIR：baseline 91.95；冻结 A1 91.84；最终目标 ≥91.50。）

### R5-D0 审计结果（完整 SYSU test 4000 对，零训练）

- **Mobile raw gate FAIL**（预注册门槛是 ResNet raw ranking 的 ~80%）：
  MobileNetV3-Small features0-3（10,488 参数，ImageNet 完整预训练）的
  1/4 PR-AUC **0.4831**（门槛 0.50，差 0.017）、1/8 0.5317（0.52 ✓）、
  Top32 prec 0.5006/0.5105（0.46/0.48 ✓）→ 3/4 子门槛过但按规则 FAIL 不放行。
  MobileDetail 是迄今最强非 ResNet detail（1/4 比 Light48 高 0.087，达 ResNet 77%）。
- **H2 强支持「PSD 失败主因是 pretrained stem 被重写」**：PSD stem conv rel_L2
  0.639 vs R4-1 ResNet stem 0.211；激活 cos(init,trained) 0.838 vs 0.916。
- **H1 否定「adapter 关键」**：TileAdapter（零参数、channel repeat/√2）替换 R4-2
  learned adapters 后 ΔF1 仅 −0.013。
- **旧 FI 零权重吸收（重大发现）**：F1 82.77 的 R4-1 参考模型里，FeatureInjector
  实际只有 1/8 一路 detail 注入活跃（1/2、1/4 的 norm2/q/kv 权重被训练成精确零）；
  R4-2 只有 1/2 活跃；R4-2d 三路全活但 F1 最低（82.02）。→ legacy FI 不是「三路
  注入」的稳定对照，且它自身也在吸收零权重。

### 已就绪的 Run6 可复用资产（重要）

- **SGDP head 已实现并通过全部 smoke**：`models/model/sgdp_head.py`（115,267 参数，
  difference-first + 语义门控 + coarse-to-fine），T1/T2 交换严格对称，
  **FLOPs 1.6778G（≤2.0G 硬门槛达标）**，参数 2,102,587 的 R5-2 组合设计完全成立
  （ViT4 1,976,832 + Mobile 10,488 + SGDP 115,267）——head 侧不需要重新设计。
- **零训练 raw gate 工具**：`analyse/run4_detail_interface_audit.py`（feature 级
  PR-AUC/Spearman/Top32）与 `analyse/run5_postmortem_and_mobile_audit.py` 的 D 部分，
  可对任何新 semantic/detail 候选直接复用。
- **预算事实**：ViT4-192（depth 4, width 192, DeiT 原位继承）独占总预算的 1.977M；
  SGDP 0.115M；剩余 ~0.11M。depth-4 证据只来自 width=192；**width 缩减（192→128/96）
  或 depth 3 尚无任何实验**。

## Run6 决策叉（预注册只允许这一域：ViT semantic 预算重分配）

候选（你选一个主推，或给出等价/更好的变体，但**不得**回到 detail branch 搜索）：
1. **缩减 ViT width**（192→128/96，甚至 depth 3）：腾出预算给 SGDP/无 detail 的
   token 重建；需要先回答「缩宽后 16×16 semantic 还有多少 change 判别力」——
   可用现成 raw gate 工具在 token 级零训练测量（我的 R4-D0 工具池化后即可复用）。
2. **去掉独立 detail branch**：ViT4-192 frozen + 只靠 16×16 token + SGDP（把省下的
   ~0.1M 加回 SGDP 或做 patch-token 上采样重建）。R5-D0 已显示 Mobile 1/8 raw
   PR-AUC 0.53 本身也不高，需要先测「ViT4 16×16 token 自身的 raw change ranking」
   作为对照（这个测量零训练、几分钟，建议列入 Run6-D0）。
3. **换 semantic 源**（如 MobileNetV3 更深层 1/16 语义 + 极浅/极窄 ViT，或
   ViT4 改小 patch）：预算与预训练继承都要重新审计。

## 请你输出的内容（一次性给全，要可执行、可预注册）

1. **证据判读**：给定 R5-D0 的四个结果（Mobile 77% ranking、H2 stem 重写、
   H1 adapter 弱、旧 FI 零权重吸收），Run6 在「语义-细节」之间应该怎么重新分配？
   明确回答：还要不要独立 detail path？如果不要，小目标/边界精度靠什么保住？
2. **Run6-D0 零训练审计**：完整定义（用哪些已训练 checkpoint/预训练权重、
   测哪些尺度的什么指标、预注册判定阈值），用于在花 80K 前裁决语义候选。
   阈值要像我之前一样给具体数字。
3. **Run6 唯一主推方案**：完整逐层定义（模块/通道/kernel/stride/BN/激活/残差/
   共享权重/预训练来源与继承方式）；参数预算表（合计 **≤2.20M**，保留 ~0.1M
   余量）与 FLOPs 粗估；明确与候选 1/2/3 的关系。
4. **预注册 gate**：每个 80K 的唯一变量、对照 reference（R4-0 83.14 / R4-1 82.77，
   注明用哪个）、F1/IoU/Δ/复杂度阈值具体数字；SYSU 终 gate 沿用
   F1≥82.30 & IoU≥69.92 & ≤2.20M，LEVIR ≥91.50。
5. **停止规则**：每个 gate 失败后的唯一出路（像之前一样明确「永久停止 X，只能
   进入 Y」），失败路径不再回 detail branch。
6. **执行顺序与实验预算**：从 raw gate → smoke 清单（冻结 ViT checksum、
   预训练继承断言、时间交换对称）→ dry run → 80K，总共 ≤3-4 个正式 run 到定稿；
   四数据集铺开条件（LEVIR 过才补 CDD/WHU）。
7. **论文机制故事**：如果成立，最终方法贡献怎么写（现有资产：R4-0→R4-1 的
   depth redundancy 证据 12→4 block 仅 −0.37 F1、省 3.56M；SGDP 的
   change-evidence-first + 语义门控机制已实现待验证）。

请直接给出完整方案文档，我会按之前 PSD / Run5 的格式执行（存进 docs/temporary/
并按 gate 一步步跑，每一 gate 都先零训练审计再花 80K）。
