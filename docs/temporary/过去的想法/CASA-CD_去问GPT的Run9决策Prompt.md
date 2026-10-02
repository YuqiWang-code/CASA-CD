# CASA-CD 下一步修改方案 Prompt（Run9：冲击四数据集硬目标，带去网页 GPT）

> 仓库：https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新，HEAD=`919760c`）
> 用法：把下面整段直接粘贴给网页 GPT。**最后一段已写明：请以一份完整 md 文档作为文件回复。**

---

我是硕士课题「CASA-CD：极轻量遥感二值变化检测」的负责人。仓库在
https://github.com/YuqiWang-code/CASA-CD （main 分支已是最新）。

请先按顺序读仓库里这些文档（事实权威）：
1. `README.md`（**新目标指标已写入**：研究定位与约定 + 数据集表；全部结果表）
2. `docs/ChatGPT_Project_Settings.md`（**项目长期指令：任务范围、证据纪律、
   硬约束、文献调研要求、交付格式——你的回答必须遵守其中「文献调研要求」与
   「工作方式与交付」**）
3. `docs/temporary/CASA-CD_Run8_R8-D0结果与B4-only路线终止.md`（最新终局：
   六轮负结果链 + 论文素材表 + 明确不成立的结论）
4. `docs/temporary/CASA-CD_Run8_B4-SPE最终路线与预注册.md`（上一轮你的方案）
5. `docs/temporary/过去的想法/CASA-CD_Run7_R7-D0结果与CSDP停止.md`、
   `CASA-CD_Run6_R6-D0结果与Run6停止.md`、`CASA-CD_Run5_R5-D0结果与Run5停止.md`、
   `CASA-CD_Run4_R4-D0审计结果与R4-2d启动.md`、`CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`
6. `docs/参考文献/文献索引.md`（2024–2026 已调研文献清单 + 官方链接 + 参考代码）
7. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`（服务器/数据规范）
代码重点：`models/model/{encoder,trainer,decoder,sgdp_head,depth_pyramid_head,b4_spe_head}.py`、
`analyse/{run4_detail_interface_audit,run5_postmortem_and_mobile_audit,run6_semantic_token_audit,run7_depth_source_audit,run8_b4_dense_recoverability_audit}.py`。

## 一、新的最终硬目标（本课题必须全部达到，不可打折）

四数据集 F1（test-as-val，同一协议、单 seed=16）：
```text
SYSU  >= 85.0
LEVIR >= 92.5
WHU  >= 95.0
CDD  >= 98.0
```
IoU 与 F1 同方向。**同时必须满足**：
- 有效推理参数 `<3M`（工程目标 ≤2.20M）；
- 已写明的全部约束：BCE+Dice、Adam(2e-4,0.9/0.99,wd=1e-4)、poly+200 warmup、
  80000 steps、batch 16、256×256、seed 16、test-as-val、ViT 冻结优先、GPU1 单卡、
  预注册 gate、不改 loss、不把训练技巧/量化/剪枝/蒸馏当核心贡献；
- **创新性、故事性、可解释性、轻量化**四项都要站得住。

注意：**新目标高于当前一切已跑模型，包括健康 full12 参考**——现有最好：
SYSU 83.14（R4-0 健康 full12）/ 82.77（R4-1）；LEVIR 91.95（baseline，ViT 部分
死权重）/ 91.84（冻结 A1）；WHU 94.84；CDD 97.75。即每个数据集都要**超越**
ChangeViT 健康参考 0.2~1.9 个 F1，同时把参数从 11.75M 压到 <3M。请在你的方案里
正面计算每个数据集的 gap 与可行性论证，不要假设「轻量保分」就够。

## 二、现状：六轮证伪后的完整证据链（都在仓库文档里，这里只给摘要）

```text
Run3  CASAA deployable router：Top32 precision 0.39→0.63 未转化为 F1（SYSU −0.12）→ 终止
Run4  自定义轻量 detail（Light32/48/PSD）：82.30/82.32/82.02，gate 三连败 → 终止
Run5  MobileNetV3 预训练 prefix：raw gate FAIL（1/4 PR-AUC 0.4831<0.50）→ 终止
Run6  PatchEmbed token（P0）重建：raw gate FAIL（0.3711<0.44）→ 终止
Run7  B2 深度截断（CSDP）：四数据集 gate FAIL（B2≈B4 只在 SYSU/LEVIR 成立）→ 终止
Run8  B4-only 无 detail 重建：dense recoverability FAIL（边界带 lift 3/4 数据集 <0.02）→ 终止
```
成立的正资产：
- depth 冗余：12→4 block 仅 −0.37 F1；四数据集 B4 稳健最优、B12 全面劣化、
  最优 depth dataset-dependent；
- token 冗余：A1 冻结 ViT + K=64 content 压缩基本无损（零新增参数）；
- 旧 FI 零权重吸收（R4-1 的 FI 实际只有 1/8 一路注入活跃）；
- 六轮全部靠**零训练 raw gate 在花 80K 前证伪**，整个课题正式 80K 只消耗过 1 个。
- 方法论纪律：raw ranking 只做筛选、不做端到端替代（Run3 与 Run8 两次独立证明）。

上述预注册停止规则仍在生效：**回到 detail branch 搜索 / P0 重建 / B2 截断 /
B4-only 无 detail**，必须明确论证为何推翻，否则默认否决。

## 三、本轮请你做的事

在**新硬目标（SYSU 85/LEVIR 92.5/WHU 95/CDD 98，且 <3M）**下，给出下一步修改方案。
目标显著高于现状，因此你可能需要新的机制假设、新的调研或新的结构路线——
**如果需要文献调研，严格遵守 `docs/ChatGPT_Project_Settings.md` 的调研准则**：
只查 2024–2026，CCF-A（CVPR/ICCV/ECCV/AAAI/NeurIPS 等）与权威 SCI（TGRS/JPRS/
TIP 等）分层标注；只引用可核验的出版社/CVF/ECVA/AAAI/arXiv/官方 GitHub；核对
题名/年份/venue/代码地址；区分已发表/录用/预印本；**不得虚构引用**。若引用了
新文献，给出官方链接并说明与本课题证据的关系（现有 `docs/参考文献/` 里 11 篇
与 `others/` 参考代码可直接复用，先查它们是否已覆盖）。

方案必须严格按项目交付格式（Settings 里「工作方式与交付」）组织，至少包含：
1. **证据表**（现有结果 vs 新目标的逐数据集 gap；六轮负结果中哪些教训直接约束本轮）；
2. **P0 正确性 / P1 方法瓶颈 / P2 实验工程**三级问题清单；
3. **候选机制 2–4 个的比较**（含与 2024–2026 文献的实质区别与创新边界），
   只选**一个主方案** + 必要对照；
4. **首选机制的数学定义与数据流**（逐层：通道/kernel/stride/归一化/激活/残差/
   共享权重/预训练来源与继承方式/训练图与推理图）；
5. **逐文件修改清单**（新增/修改哪些文件）；
6. **实验设计**：每个实验的唯一变量、数据集顺序、训练预算（GPU1 单卡，SYSU 一个
   80K 约 3-3.5h——总 80K 数量必须给预算并说明理由）、seed、成功阈值（给具体
   F1/IoU 数字的预注册 gate）、失败解释与**停止规则**；任何新结构候选必须先过
   零训练 raw gate（给具体指标与阈值）才允许花 80K；
7. **smoke / dry run 测试清单**（冻结 ViT checksum、预训练继承断言、时间交换对称、
   参数/FLOPs 硬门槛、arch sidecar 校验）；
8. **启动顺序与 checkpoint/log 路径**；
9. **论文机制故事与可解释性论证**（如何讲清「为什么达到 85/92.5/95/98 且 <3M」，
   可解释性靠什么：gate 分析、raw audit、消融链）；
10. 最后给「**立即执行顺序**」和「**仍需补充证据**」。

## 四、最后（重要）

请把上述完整方案**直接写成一份 md 文档，以文件形式回复我**（文件名建议
`CASA-CD_Run9_可执行预注册方案.md`；文档内请像你之前的 Run5/6/7/8 方案一样
带「结论先行、证据表、预注册 gate、停止规则、立即执行顺序」）。我会把该文件
存进 `docs/temporary/` 并按它一步步执行（先零训练 gate，再 dry run，再 GPU1 的
80K，逐 gate 判定）。
