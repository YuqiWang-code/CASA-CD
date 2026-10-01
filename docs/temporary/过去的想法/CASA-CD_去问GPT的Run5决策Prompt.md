# CASA-CD 下一步决策 Prompt（带去网页 GPT，附 GitHub 链接）

> 仓库：https://github.com/YuqiWang-code/CASA-CD （main 分支，代码与下述数字全部同步）
> 用法：把下面整段直接粘贴给网页 GPT。让它先读仓库里的指定文档再回答。

---

我是硕士课题「极轻量遥感二值变化检测」的负责人。仓库在
https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新）。

请先读仓库里这几份文档（按顺序），它们是我项目的事实权威：
1. `README.md`（结果表 + 全部约定）
2. `docs/temporary/CASA-CD_交接文档_2026-09-30.md`（上下文交接）
3. `docs/temporary/CASA-CD_Run4_R4-2失败后_下一步决策与PSD-Detail方案.md`（上一轮你的方案，已执行完）
4. `docs/temporary/CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md`（audit 结果）
5. `docs/temporary/CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`（最终失败记录 + Run5 候选）
6. `docs/temporary/过去的想法/CASA-CD_Run4_极轻量结构主线_可执行方案.md`（Run4 原始设计）
7. `train_scripts/UltraLight/Run4/README.md`（阶段 gate 定义）
代码重点：`models/model/encoder.py`、`psd_detail.py`、`light_detail.py`、`decoder.py`、`trainer.py`。

## 一句话现状

目标：二值变化检测，4 数据集（CDD/LEVIR/SYSU/WHU，256×256，label 阈值 gray≥128），
最终模型有效推理参数 **<3M（工程目标 ≤2.20M）**，在 2M 量级超过轻量 SOTA
（RFANet 2.86M / SeCoR 2.50M / Lighter 1.10M / CGLNet 0.99M）。
已完成 ChangeViT-T baseline 复现；change-aware router（CASAA）按预注册规则终止；
主线二 Run4（<3M 极轻量结构，逐组件单变量）刚走完，**轻量 detail 分支路线
三次失败后按预注册 Stop-3 终止**。现在需要你给出 Run5 的下一步方案。

## 必须遵守的硬约束（不要违反）

- 方法创新导向；**不改 loss、不把训练技巧当创新**；单 seed=16（不做多 seed）。
- 每个实验**唯一变量 + 预注册 gate**，gate 不过不进入下一步；不做 sweep。
- 训练协议固定：BCE+Dice、Adam(lr=2e-4, betas=(0.9,0.99), wd=1e-4)、poly(power 0.9)
  +200 iter warmup、max_steps=80000、batch 16、256×256、seed 16、test-as-val
  （每 epoch 在 test 上挑 best）、ViT 一律冻结（解冻需先过 2K health gate）。
- 服务器只有 GPU1 可用（2×RTX 5090，另一张是别人的）；SYSU 一个 80K 约 3-3.5h、
  LEVIR ~4.4h，所以**请控制正式 80K run 的总数（最好 ≤3-4 个就能到定稿）**。
- 最终方法里参数量口径：forward 图内全部参数（冻结照计、死参数不计）。
- 不把量化/剪枝/蒸馏当核心贡献。

## 完整证据链（全部数字最新，可直接引用）

### 全部正式结果（SYSU，F1 / IoU，test-as-val）

| Run | 结构 | F1 | IoU | Params | FLOPs | 判据 |
|---|---:|---:|---:|---:|---|
| baseline Run1 | ChangeViT-T 官方协议（ViT 可训练，后证实部分死 ViT） | 82.48 | 70.19 | 11.754M | 26.32G | 历史参考 |
| R4-0 A0_FULL12_FROZEN | 健康 full12 冻结 + corrected DeiT loader | **83.14** | 71.14 | 11.754M | 26.32G | 新健康参考 |
| R4-1 VIT4_OLDHEAD | ViT depth 12→4，其余原样 | 82.77 | 70.61 | 8.195M | 24.10G | **通过**（ΔF1 −0.37 ≥ −0.50） |
| R4-2 VIT4_LIGHTDETAIL | ResNet C2-C4 → LightDetail 32/64/128（随机初始化 DSConv）+ 1×1 adapters | 82.30 | 69.92 | 5.492M | 10.73G | 未过（−0.47） |
| R4-2b LIGHTDETAIL48 | 容量 fallback 48/96/160 | 82.32 | 69.95 | 5.533M | 10.98G | 未过（−0.45） |
| R4-2d PSD_DETAIL | PSD-Detail 0.078M：ImageNet ResNet18 conv1/bn1 原位复制 stem + 1/2、1/4 残差 DS block + stride-2 先 PW 混合后 DW 降采样，无 adapters | 82.02 | 69.52 | 5.491M | 11.52G | **未过（−0.75）** |

（LEVIR：baseline 91.95；冻结 A1 91.84；最终目标 LEVIR ≥91.50。）

### R4-D0 无训练 audit（完整 SYSU test 4000 对，feature 级诊断）

三尺度 detail feature 池化到 16×16，s=1−cos(f1,f2) 对齐 GT occupancy 的 PR-AUC：

| 尺度 | ResNet direct | Light32 raw | Light32 adapted | Light48 raw |
|---|---:|---:|---:|---:|
| 1/2 | 0.547 | **0.555** | 0.560 | 0.523 |
| 1/4 | 0.626 | **0.325** | 0.308 | 0.397 |
| 1/8 | 0.654 | **0.317** | 0.298 | 0.328 |

结论：浅层不是瓶颈（Light≥ResNet），深层（1/4、1/8）raw 崩坏到接近随机；
adapter 前后变化 ≤0.03（adapter 不是主因）；width bump 只在 1/4 有 +0.07、
离 ResNet 还差 0.23。按预注册规则判 C → 直接做了 PSD。

### R4-2d PSD 失败的关键事实

- PSD（pretrained stem + residual + MixDown）反而比随机初始化 LightDetail **低 0.28 F1**；
- 训练曲线：best @epoch 64/106，末 20% 无上升趋势 → 排除欠训练；
- 我列了 4 条未验证假设（在终止文档 §4）：① LightDetail 的 43K 可学习 adapters
  提供了 detail→FI 接口自由度，PSD 裸 64/128/256 输出没有；② pretrained stem 在
  统一 lr=2e-4+wd=1e-4 下被重写；③ 深层 256ch 无 adapter 与原 FI 统计不匹配；
  ④ 单 seed 噪声 ±0.1-0.2。**请你先判断这 4 条里哪条最可能，以及有没有一个
  零训练成本的诊断可以一次性分辨（有的话给出具体脚本设计）。**

### 参数预算的结构性事实（最重要）

| 组件 | 参数 |
|---|---:|
| TinyViT4-192（DeiT 前 4 block，原位继承） | 1,976,832 |
| 原 FeatureInjector | 1,395,200 |
| 原 decoder（difference MLP + deconv + 上采样） | 2,040,192 |
| LightDetail 32/64/128 | 37,024 |
| PSD-Detail | 78,464 |

也就是说：**ViT4 一个组件就占掉 2.20M 预算的 1.977M**，剩下 detail+injector+
decoder 三件套总共只有 **~0.22M**。Run4 三次失败都是在这 0.22M 里换 detail
（0.037→0.065→0.078M）而注入器/解码器仍用 3.44M 原版——所以失败实验其实都在
「5.5M 总参数」量级，从未真正逼近 2.2M。这个预算结构请你正面回应：继续 ViT4-192
宽度 + 0.22M 外围，还是改变 ViT 部分本身（更窄/更浅/换 pretrained 源）的分配。

## 请你输出的内容（一次性给全，要可执行、可预注册）

1. **失败归因核查**：对 PSD 低于 LightDetail 的 4 条假设排序；若建议先做无训练
   诊断，给出该诊断的完整指标定义与判定阈值（像上次 R4-D0 那样，零训练成本优先）。
2. **Run5 唯一主推方案**：
   - 完整逐层定义（每个模块：输入输出通道、kernel、stride、BN/激活、残差、
     是否共享权重、是否用 ImageNet 预训练及从哪复制）；
   - 参数预算表（每个组件列出并合计，最终 **≤2.20M**，并给 FLOPs 粗估）；
   - 与三个候选方向（预算重分配 / 成熟超轻 pretrained 子层做 detail /
     去掉独立 detail 分支）的关系：选哪个、为什么。
3. **预注册 gate**：每一阶段（对应一个 80K run）的唯一变量、对照 reference
   （R4-0 83.14 或 R4-1 82.77，注明用哪个）、通过判据的具体数字
   （F1/IoU/Δ 阈值 + 参数量/FLOPs 硬门槛）。
4. **停止规则**：每个 gate 失败后的下一步（不要再给"调调再试"的模糊选项，
   要像上次一样明确"永久停止 X 路线"）。
5. **执行顺序**：从改代码 → smoke 清单（含冻结 ViT checksum、预训练继承断言）→
   dry run → 正式 80K，总共最多几个 80K run，四数据集铺开条件（LEVIR ≥91.50 后
   才补 CDD/WHU）。
6. **论文机制故事**：如果方案成立，最终方法的贡献陈述怎么写（第一贡献是
   R4-0→R4-1 的 depth redundancy 证据：12→4 block 仅 −0.37 F1、省 3.56M 参数；
   第二贡献由你的 Run5 方案决定）。

请直接给出完整方案文档，我会按你上次 PSD-Detail 文档的格式执行（我会把通过的
方案存进 docs/temporary/ 并按 gate 一步步跑）。
