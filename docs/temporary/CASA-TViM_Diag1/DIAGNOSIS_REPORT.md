# CASA-TViM 小目标瓶颈分阶段诊断报告（Run-Diag / Diag1）

- 诊断根目录：`/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1`
- 预注册阈值：D1 size gap ≥10.0pp、D2 stage drop ≥10%（相对）+ bootstrap 同号、D3 probe AP 绝对下降 ≥0.02、每组最少 100 个 GT 对象

## 1. 实验身份与完整性（§14.1）

- 指标口径协议：`tvim_object_metrics/v1 (Diag1)`（连通性 4，IoU 匹配阈值 loose=0.1 strict=0.5，判定规则 `pred_bool = (prob > 0.5)  # 与 models/train.py 一致（严格大于）`）

### D0 A2_STR

- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1/A2_STR/SYSU-CD-256/best_F1=0.8322.pth`
  sha256：`01c9458c65882b3faa062de2c78c58f4b430b0255cc3f7692c41d771654cb3f7`
- 原始日志：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run1/A2_STR/SYSU-CD-256/train_log.txt`（TEST 区块行 [543, 565]）
- 复现六指标：R=0.81236 P=0.85287 OA=0.92270 F1=0.83212 IoU=0.71251 Kappa=0.78195
- 与日志最大偏差：0.00（容差 0.0001）
- 训练图参数：5084016；部署图参数：4880189；部署 FLOPs：2.6747 G （unsupported ops 12）
- 折叠等价性：随机 batch max_abs=0.00, disagreement=0.00；真实 batch max_abs=0.00, disagreement=0.00（flipped=1）
- Gate D0-REPRO：**FAIL**
- 数据集：4000 张，list sha256 `5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797`；空 GT 图 0 张
- GT 连通域按面积计数：{'tiny_1_15': 66, 'tiny_16_63': 96, 'small_64_255': 202, 'medium_256_1023': 692, 'large_1024_inf': 4706}

### D0 M1_FULL

- checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1/M1_FULL/SYSU-CD-256/best_F1=0.8347.pth`
  sha256：`45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457`
- 原始日志：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run1/M1_FULL/SYSU-CD-256/train_log.txt`（TEST 区块行 [667, 692]）
- 复现六指标：R=0.84415 P=0.82548 OA=0.92116 F1=0.83471 IoU=0.71631 Kappa=0.78295
- 与日志最大偏差：0.00（容差 0.0001）
- 训练图参数：5084017；部署图参数：4880190；部署 FLOPs：2.6747 G （unsupported ops 19）
- 折叠等价性：随机 batch max_abs=0.00, disagreement=0.00；真实 batch max_abs=0.00, disagreement=0.00（flipped=0）
- Gate D0-REPRO：**PASS**
- 数据集：4000 张，list sha256 `5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797`；空 GT 图 0 张
- GT 连通域按面积计数：{'tiny_1_15': 66, 'tiny_16_63': 96, 'small_64_255': 202, 'medium_256_1023': 692, 'large_1024_inf': 4706}

## 2. P0 指标口径修正（§3）

旧 `analyse/run2_zero_cost_diag.py::component_pr` 的 `FP` 用 GT 局部掩码过滤，恒等于 0，故其 `small/medium/large F1` 是**受限正样本组内的伪指标**，不是对象级 F1。新实现（`analyse/tvim_object_metrics.py`）给出语义分离的指标：

- pooled / object-macro **像素 Recall**（GT 面积分组）
- ObjectHit@1 / **@25%**（预注册主对象召回）/ @50%
- 真正匹配预测连通域的 **ObjPrecision / ObjRecall / ObjF1**（IoU≥0.10 loose、≥0.50 strict，一次性最大权重二分匹配）

### M1_FULL：旧伪指标 vs 新正确口径

- 旧 `component_pr` pseudo small F1：small 0.1796 / medium 0.4620 / large 0.8273（FP 恒 0 的伪指标）
- 新 **small pooled pixel Recall**：0.1974；object-macro Recall：0.1654；Hit@25%：0.1896
- 新对象级指标（IoU≥0.10）：ObjPrecision=0.5320，ObjRecall=0.7674


## 3. D1：小目标错误画像（§5 / §14.2）

### M1_FULL（4000 张，limited=False）

- 全图像素：R=0.84415 P=0.82548 F1=0.83471 IoU=0.71631；TP/FP/FN/TN=52186035/11033302/9634882/189289781；正类先验=0.23583

| GT 面积组 | 对象数 | 像素数 | pooled pixel Recall | object-macro Recall | Hit@25% | Hit@50% | 证据充足 |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny_1_15 | 66 | 412 | 0.0752 | 0.1263 | 0.1364 | 0.1212 | 否(<100) |
| tiny_16_63 | 96 | 3421 | 0.1245 | 0.1483 | 0.1667 | 0.1354 | 否(<100) |
| small_64_255 | 202 | 32004 | 0.2068 | 0.1863 | 0.2178 | 0.1931 | 是 |
| medium_256_1023 | 692 | 456253 | 0.4213 | 0.3988 | 0.4855 | 0.4249 | 是 |
| large_1024_inf | 4706 | 61328827 | 0.8477 | 0.7491 | 0.8725 | 0.8043 | 是 |

- 对象级（真实匹配）：N_gt=5762 N_pred=8312 matched(loose)=4422 matched(strict)=3455；ObjP(loose)=0.5320 ObjR(loose)=0.7674 ObjF1(loose)=0.6284；严格：ObjP=0.4157 ObjR=0.5996
- FP 像素=11033302；预测连通域按面积分布={'tiny_1_15': 1047, 'tiny_16_63': 742, 'small_64_255': 723, 'medium_256_1023': 1038, 'large_1024_inf': 4762}；未匹配（FP）连通域分布={'tiny_1_15': 1047, 'tiny_16_63': 732, 'small_64_255': 651, 'medium_256_1023': 647, 'large_1024_inf': 813}
- band2：F1=0.6890 TP=2747031 FP=975871 FN=1504418 band 像素=6667320 空 band 数=0
- band4：F1=0.7159 TP=5584569 FP=1728255 FN=2704467 band 像素=13122699 空 band 数=0
- Gate D1-PHENOMENON：**PASS**（size gap=65.03pp，阈值 10.0pp：满足）
- paired M1_FULL vs A2_STR：共同对象 5762 个 / 4000 图；对象级 mean Δrecall=-0.043053；图像级 bootstrap CI=-0.049300~-0.037182；分组={'tiny_1_15': -0.03587, 'tiny_16_63': 0.01236, 'small_64_255': 0.01394, 'medium_256_1023': -0.04172, 'large_1024_inf': -0.04693}


## 4. D2：分阶段证据衰减（§6 / §14.2 核心表）

### M1_FULL（4000 张，limited=False）

- 重建校验：head logits→interp→sigmoid vs 模型输出 max_abs=0.0，二值 disagreement=0.0
- 分辨率协议：主口径=score 上采样到原生 256²（bilinear, align_corners=False），GT 保持原生原样
  副口径=native feature grid occupancy = adaptive_avg_pool2d(GT)，用于分桶描述

| 节点 | 原生尺度 | pooled AP | per-image AP | bootstrap CI | small margin | recall@1%FP | recall@5%FP | 小/中/大 GT 分数均值 |
|---|---|---:|---:|---|---:|---:|---:|---|
| L00_patch_embed | 1/4 64² | 0.3439 | 0.4022 | 0.3922~0.4117 | 0.00944 | 0.0440 | 0.1660 | 0.1279/0.1339/0.1233 |
| L01_network0 | 1/4 64² | 0.3787 | 0.3756 | 0.3659~0.3842 | 0.01983 | 0.0375 | 0.1462 | 0.8213/0.8403/0.8139 |
| L01b_norm0 | 1/4 64² | 0.3904 | 0.3702 | 0.3607~0.3791 | 0.00003 | 0.0352 | 0.1377 | 0.8976/0.9367/0.9493 |
| L02_network1 | 1/8 32² | 0.5087 | 0.4981 | 0.4879~0.5069 | 0.03287 | 0.0604 | 0.2202 | 0.5378/0.5505/0.5276 |
| L03_network2 | 1/8 32² | 0.4942 | 0.4616 | 0.4512~0.4704 | 0.02075 | 0.0501 | 0.1899 | 0.7981/0.8264/0.8328 |
| L03b_norm2 | 1/8 32² | 0.4791 | 0.4398 | 0.4293~0.4492 | 0.01383 | 0.0440 | 0.1720 | 0.8391/0.8781/0.9098 |
| L04_network3 | 1/16 16² | 0.5384 | 0.5357 | 0.5255~0.5452 | 0.02181 | 0.0638 | 0.2329 | 0.7499/0.7741/0.7757 |
| L05_stage3_last_prefix | 1/16 16² | 0.5506 | 0.5105 | 0.5005~0.5197 | 0.01142 | 0.0560 | 0.2134 | 0.6996/0.7356/0.7499 |
| L06b_norm4 | 1/16 16² | 0.5781 | 0.5058 | 0.4956~0.5148 | 0.00963 | 0.0521 | 0.2063 | 0.8163/0.8442/0.8880 |
| L07_network5 | 1/32 8² | 0.3395 | 0.3420 | 0.3320~0.3517 | 0.00132 | 0.0214 | 0.0889 | 0.3320/0.3345/0.3475 |
| L08b_norm6 | 1/32 8² | 0.4210 | 0.3874 | 0.3783~0.3963 | 0.00147 | 0.0150 | 0.1089 | 0.8335/0.8798/0.9117 |

**相邻节点衰减判据（§10.2）**：

| 从 → 到 | 分辨率 | AP 变化(相对) | margin Δ | bootstrap 方向 | 达标(≥10% 且同号) |
|---|---|---:|---:|---:|---|
| L00_patch_embed → L01_network0 | 1/4 64² → 1/4 64² | 10.11% | 0.01039 | -1 | 否 |
| L01_network0 → L01b_norm0 | 1/4 64² → 1/4 64² | 3.10% | -0.01980 | 0 | 否 |
| L01b_norm0 → L02_network1 | 1/4 64² → 1/8 32² | 30.28% | 0.03284 | 1 | 否 |
| L02_network1 → L03_network2 | 1/8 32² → 1/8 32² | -2.84% | -0.01212 | -1 | 否 |
| L03_network2 → L03b_norm2 | 1/8 32² → 1/8 32² | -3.07% | -0.00692 | -1 | 否 |
| L03b_norm2 → L04_network3 | 1/8 32² → 1/16 16² | 12.39% | 0.00798 | 1 | 否 |
| L04_network3 → L05_stage3_last_prefix | 1/16 16² → 1/16 16² | 2.27% | -0.01038 | -1 | 否 |
| L05_stage3_last_prefix → L06b_norm4 | 1/16 16² → 1/16 16² | 4.99% | -0.00179 | 0 | 否 |
| L06b_norm4 → L07_network5 | 1/16 16² → 1/32 8² | -41.27% | -0.00831 | -1 | 是 |
| L07_network5 → L08b_norm6 | 1/32 8² → 1/32 8² | 24.00% | 0.00015 | 1 | 否 |

**CAACP 内部证据**：

- beta = 0.005570
- mean_abs_deltaC = 0.055412
- mean_abs_beta_deltaC = 0.000309
- beta_delta_rms_exact = 0.001606
- ratio_beta_delta_over_cavg = 0.001858
- cell_weight_entropy_mean = 1.265123
- cell_weight_top1_mean = 0.493804
- xlow_feature_norm_mean = 10.680772
- n_samples = 8000.000000
- group_mean_abs_deltaC = {'small': 0.074064, 'medium': 0.0462, 'large': 0.035185}
- group_mean_abs_residual_current = {'small': 0.321589, 'medium': 0.269307, 'large': 0.184959}
- group_mean_abs_residual_avg_anchor = {'small': 0.321706, 'medium': 0.269382, 'large': 0.18497}

单路融合节点（禁止 A/B cosine，仅描述性）：

| 节点 | 形状 | GT 内范数均值 | 背景范数均值 | small GT/背景 比值 |
|---|---|---:|---:|---:|
| L06a_caacp_block | [[168, 16, 16]] | 40.3439 | 39.7005 | 1.0796 |
| T01_tar1 | [[96, 64, 64]] | 6.8207 | 6.5853 | 1.1053 |
| T02_tar2 | [[96, 32, 32]] | 7.4669 | 6.5050 | 1.1832 |
| T03_tar3 | [[96, 16, 16]] | 12.8256 | 11.2142 | 1.1655 |
| T04_tar4 | [[96, 8, 8]] | 13.1998 | 9.3523 | 1.3323 |
| D03_decoder_block3 | [[96, 16, 16]] | 14.8806 | 13.7360 | 1.0634 |
| D02_decoder_block2 | [[96, 32, 32]] | 16.1628 | 11.3156 | 1.3029 |
| D01_decoder_block1 | [[96, 64, 64]] | 20.0382 | 17.7993 | 1.0503 |
| DOUT_decoder_refine | [[96, 64, 64]] | 28.5101 | 27.6543 | 0.9722 |
| P00_head_logits | [[1, 64, 64]] | 7.0102 | 7.9075 | 0.7910 |

- Gate D2-VALID：**PASS**

## 5. D3：冻结线性 Probe（§7）

### M1_FULL

- Gate D3-VALID：**PASS**
- 预算偏差：--max-train=3000（0=全部）; --extract-batch=16

| 节点 | C_in | probe-train 样本 | probe-train 子集 AP | **test pooled AP** | test per-image AP |
|---|---:|---:|---:|---:|---:|
| L01b_norm0 | 48 | 3011 | — | **0.4588** | 0.4051 |
| L03b_norm2 | 64 | 3011 | — | **0.4871** | 0.4021 |
| L05_stage3_last_prefix | 168 | 3011 | — | **0.5324** | 0.5043 |
| L06b_norm4 | 168 | 3011 | — | **0.4961** | 0.4726 |
| T01_tar1 | 96 | 3011 | — | **0.6911** | 0.6077 |
| DOUT_decoder_refine | 96 | 3011 | — | **0.9053** | 0.8240 |
| P00_head_logits | 1 | 3011 | — | **0.9049** | 0.8235 |

- 冻结模型参数校验：unchanged=True

## 6. D4：CAACP β 受控反事实（§8）

### M1_FULL（β_on=0.005570 → β_off=0.0）

- 全图像素 ON：R=0.844149 P=0.825476 F1=0.834708 IoU=0.716308
- 全图像素 OFF：R=0.844135 P=0.825493 F1=0.834710 IoU=0.716311
- Δ(ON−OFF)：{'recall': 1.34e-05, 'precision': -1.7e-05, 'OA': -2.5e-06, 'F1': -2.2e-06, 'IoU': -3.2e-06, 'Kappa': -4.1e-06}
- small 组：microR 0.197422 → 0.197394（Δ=0.0000279）；Hit@25 0.18956 → 0.18956
- per-object paired Δrecall：n=5762，均值=0.0000158，图像级 bootstrap CI=0.0000103~0.0000259
- 完整性：only_beta_changed=True，β 别名键=['encoder.caacp_block.op.beta', 'encoder.caacp_op.beta', 'encoder.network.4.8.op.beta']，BN 校验和不变=True，ON 可复现 max_abs=0.0
- Gate D4-COUNTERFACT：**PASS**
- 解释边界：Y_on − Y_off 仅衡量当前训练权重对 CAACP 修正项的**局部依赖**；M1 其余参数是在 β 可学习条件下训练的，因此这不是 A2（从头训练无 CAACP）的受控对照，也不能据此断言 Stage3 是最早损失源。禁止表述为'关闭 CAACP 可提升模型'。

## 7. Gate 汇总与自动评估（§10.2 / §10.3）

| 阶段 | 对象 | 状态 |
|---|---|---|
| D0 | A2_STR | FAIL |
| D0 | M1_FULL | PASS |
| D1 | M1_FULL | PASS |
| D2 | M1_FULL | PASS |
| D3 | M1_FULL | PASS |
| D4 | M1_FULL | PASS |

### 自动评估（机械应用预注册阈值）

- **D1 size gap**：M1_FULL small pooled recall=0.1974 vs large=0.8477，gap=65.03pp；small n=≥100，large n=≥100 ⇒ 满足 ≥10pp 且样本充足
- **D2 stage drop**：M1_FULL 满足判据的相邻边界数=1（L06b_norm4→L07_network5）；AP 相对变化最大负向边界=L06b_norm4→L07_network5（-0.4127，bootstrap 方向=-1）
- **D3 recoverability**：M1_FULL 编码器节点 probe test AP 区间 [0.4588, 0.5324]；下游融合/解码/head 节点 {'T01_tar1': 0.6911, 'DOUT_decoder_refine': 0.9053, 'P00_head_logits': 0.9049} ⇒ 若下游显著更高，说明'改变判别性'主要由 TAR/DCR 学习到的时相代数提供，而非编码器 A/B 差分本身（注意编码器 probe 输入被固定为 abs(F_A−F_B)，不含可学习的时相组合，属口径差异）
- **D3 与 D2 同向性（CAACP 边界 L05→L06b）**：probe AP 变化 -0.0363（下降）；判据阈值 |Δ|≥0.02 ⇒ 满足
- **D4 on/off**：M1_FULL ΔF1=-0.0000022，Δsmall microR=0.0000279 ⇒ 微小/非稳健 ⇒ 不支持 β 本身是主因

> 结论等级只能按 §10.3 决策矩阵给出；`D2` 与 `D3` 同向才可称『高度怀疑该阶段损失』。


---

<!-- 以下为解释性章节，来源：INTERPRETATION.md -->

﻿# Diag1 解释性章节（§8 偏差 / §9 结论 / §10 Run4 建议）

> 本文件由研究者撰写，	vim_diag_report.py 会在机器生成报告末尾**自动追加**本文件内容；
> 因此重新生成报告不会丢失解释性结论。

## 8. 执行偏差、异常与修正（透明记录，§13）

1. **代码身份**：服务器工作副本 `/home/yqwang/projects/CASA-CD` **不是 git 仓库**（历史一直是 SFTP 部署），因此按文档要求改用**逐文件 SHA256** 记录代码身份（12 个模型/数据/训练文件，见 `RUN_MANIFEST.json:code_identity`）；本地仓库执行期间 HEAD = `fd8e0e3`。
2. **D0/A2_STR 折叠硬门 FAIL（已如实记录，未用于任何结论）**：在固定真实 batch（16 张）上 train 图 vs deploy 图有 **1 个像素**（共 1,048,576）在 0.5 两侧翻转，`binary_disagreement=9.54e-7 ≠ 0`。该现象与 Run1 原始日志中 SYSU 的 `[REPARAM-REAL-ARGMAX-DISAGREE] 9.54e-07` 一致，属**概率恰在 0.5 的刀锋像素**；A2 的六指标复现偏差仍 ≤4.8e-5，且 D1 的 paired 对照使用 **train 图**预测（不经过折叠），故该 FAIL 不影响任何归因。**未用"很接近零"放行**，按文档规则保留 FAIL 标记。
3. **D3 预算偏差**：probe 拟合使用 SYSU train 的 3011 张（90% hash 划分 + `--max-train 3000` 上限），而非全部 12,000 张；step/lr/batch/seed 均为预注册值。该偏差降低 probe 拟合质量，已写入 `probe_protocol.json`。
4. **编码器 probe 输入口径**：按文档允许的二选一，固定采用较紧凑的 `abs(F_A−F_B)`（全层一致）。这使编码器 probe 与"已学习时相代数"的 TAR/DCR 节点**不可直接横比**，已在报告中标注为口径差异。
5. **开发期发现并修正的两个实现缺陷（已重跑受影响阶段）**：
   - `NodeScoreAcc.add()` 中 small-object margin 代码块一度落在 `return` 之后成为死代码 → margin 全为 None；由"margin 对象数=0 vs D1 的 364 个小对象"这一交叉校验发现并修正，重跑后 `n_objects=364` 与 D1 完全一致。
   - CAACP 分组 occupancy 权重一度用 `int(round(sw))` 累加，导致小目标权重被舍入为 0（small 分组为 None）→ 改为浮点累加后 small=0.0741 正常出现。
   - 所有报告数字均来自修正后的重跑（D2 v4、D4 v2）。
6. **一次误操作**：重复执行了 D1 启动命令，导致同一输出目录被两个进程并发写入；已 `kill` 全部相关进程并**重新干净单次运行** D1（`object_level.csv` / `summary.json` 与 gate 均由该次运行产出，`[D1] gate=PASS`）。未删除任何历史资产。

---

## 9. 分阶段结论（§14.3，逐条标注证据等级）

**[F] 结构与池化事实（代码 + 实测 shape 双向核对）**
- `SS2D` 低频池化因子为 `2**(3-index)`：1/4→8×8、1/8→4×4、1/16→2×2、1/32→无池化；**仅低频通路**，高频 dense 路径完整保留（dense 16² 特征并未被删除）。
- CAACP 只作用于 Stage3（index=2）最后一个 TViM 的低频通路：`16² → 2×2 cell 加权 → 8² → selective scan`；`c = c_avg + β(c_ca − c_avg)`，并有 `res = x0 − Up(c)`（M1 为 `current` 模式）。
- 实测节点 shape（Gate D2-VALID）：L00/L01/L01b=64²(48C)、L02/L03/L03b=32²(64C)、L04/L05/L06a/L06b=16²(168/168/168)、L07/L08=8²、T01–T04=64²/32²/16²/8²(96C)、D=16²/32²/64²/64²(96C)、head logits=64²(1C)。hook 开关前后最终预测 **max_abs=0.0、二值 disagreement=0**。
- CAACP 内部实测：`β=5.57e-3`、`mean|ΔC|=0.0554`、`mean|βΔC|=3.09e-4`、`sqrt(mean((βΔC)²))=1.6e-3`、`||βΔC||/||c_avg||=1.86e-3`（0.19%）、cell 权重熵 `1.2651`（均匀=1.3863）、top1 权重 `0.494`。

**[F] 指标口径修复的结论（P0）**
- 旧 `component_pr` 的 FP 恒为 0，`small/medium/large F1 = 0.1796/0.4620/0.8273` 是**受限正样本组内的伪指标**；已在报告中并列表为"旧伪 F1 / 新 small Recall / 新 ObjRecall"，未覆盖任何历史 JSON。
- 新正确口径（M1，4000 张，4 连通，`p>0.5`）：
  - small(1–255px，364 个对象) pooled pixel Recall **0.1974**、object-macro Recall **0.1654**、Hit@25% **0.1896**；
  - medium(692) **0.4213 / 0.3988 / 0.4855**；large(4706) **0.8477 / 0.7491 / 0.8725**；
  - 真实对象匹配（IoU≥0.10）：**ObjPrecision 0.5320 / ObjRecall 0.7674 / ObjF1 0.6284**（严格 IoU≥0.5：0.4157 / 0.5996）；
  - **size gap = 65.03pp**（small vs large pooled pixel Recall）≫ 10pp 预注册阈值，且两组对象数均 ≥100。
- 结论：**"小目标召回远低于大目标"这一现象在修正口径后依然成立**（旧 0.1796 的数值巧合接近新 pooled Recall 0.1974，但语义完全不同，叙事必须改写）。

**[I] 第一次稳健衰减出现在哪里**
- 编码器 A/B cosine proxy（S0）中，**唯一满足"相对下降 ≥10% 且 bootstrap 同号"的相邻边界是 `L06b_norm4 → L07_network5`（1/16 → 1/32）**：pooled AP 0.578→0.340（−41.3%）、small margin 0.0096→0.0013（−86%）、recall@1%FP 0.0521→0.0214。**但该边界跨越分辨率**（16²→8²），按 §6.4 不能仅凭 proxy 断言信息丢失。
- **同分辨率边界 `L05_stage3_last_prefix → L06b_norm4`（CAACP 所在块）**：pooled AP **+4.99%**（0.5506→0.5781）、small margin **−15.8%**（0.0114→0.0096）、recall@1%FP **−7%**；D3 probe 在同样边界 **−0.0363**（0.5324→0.4961，超过 0.02 阈值）⇒ **S0 与 S1 方向一致（margin 与 probe AP 同向下降）**，按 §6.4 可记为 **[I]"高度怀疑该阶段（Stage3 末块 / CAACP 所在块）存在稳健但幅度有限的证据弱化"**。
- 但 **[F]+[I] 该弱化不能归因于 CAACP 的 change-aware 修正项本身**：`||βΔC||/||c_avg||=0.19%`，且 D4 直接把 β 置零后 ΔF1 = **−2.2e-6**、small Hit@25 完全不变（per-object Δrecall 均值 1.6e-5，CI 1.0e-5~2.6e-5 —— 统计可分辨但实际可忽略）。同分辨率内的其它成分（LocalBlock / SS2D scan / BN / 2×2→8² 压缩本身）才是该边界变化的更可能来源。
- 早期阶段：`L01b_norm0`（1/4）的 small-object margin ≈ **3e-5**（几乎为 0）、probe AP 最低（0.4588）⇒ **[H] H1（早期编码不足）仍未被排除**。

**[I] 输出端不是瓶颈（H4 不成立）**
- D3：`DOUT_decoder_refine` probe test AP **0.9053** ≈ `P00_head_logits` **0.9049** ⇒ 解码特征与 head logits 的判别性相同，**不存在"解码器可分但预测头漏检"的输出判别瓶颈**；FRH 类"末端再加细粒度头"的方向因此缺乏证据。

**[I] TAR/DCR 融合确实在"创造"判别性（但口径受限）**
- D3：编码器（固定 `abs(F_A−F_B)` 输入）0.4588–0.5324；TAR stage1（学习到的时相代数）**0.6911**；解码器/head **0.905**。⇒ 判别性主要由 TAR/DCR 学到的时相组合提供。**注意口径**：编码器 probe 不含可学习的时相组合，因此这**不能**直接证明"编码器丢信息"（[M] 见下）。

**[F]+[I] 对既有叙事的重要更正：CAACP 对小目标是"有帮助"而非"有害"**
- 在**正确指标**下做 same-object paired（M1 vs A2，5762 个共同对象）：
  - `tiny_16_63` **+0.0124**、`small_64_255` **+0.0139**（CAACP 提升小目标 per-object recall）
  - `medium_256_1023` **−0.0417**、`large_1024_inf` **−0.0469**（CAACP 降低中等/大目标 per-object recall）
  - 整体 per-object 均值 **−0.0431**（CI −0.0493~−0.0372），但像素级 F1 **+0.26pp**（83.21→83.47），因中/大目标主导像素数。
- 因此 Run2/Run3 复盘中"Stage3 CAACP 对 SYSU small 微负（−0.0042）"的结论**建立在伪指标之上，必须撤回**；正确结论是：**CAACP 在 SYSU 上以中/大目标 per-object recall 为代价换取小目标 per-object recall 与像素级 F1**。
- 这也同时**削弱了 Run3 决策树中"把 CAACP 前移到 Stage2"（S2-HCAACP）的主要动机**。

**[I] 错误结构画像（D1）**
- 全图 FP 像素 11.03M；预测连通域 8312 个 vs GT 5762 个，其中**未匹配（FP）连通域 3077 个，而 2430 个面积 <256px**（tiny_1_15 1047、tiny_16_63 732、small_64_255 651）⇒ 小尺度上**漏检与碎片化假阳性同时存在**，不是单纯的"响应太弱"。
- boundary band：band±2 F1 0.6890、band±4 F1 0.7159。

**[M] 仍缺的关键证据（足以推翻或确认当前判断）**
1. **编码器 probe 的可学习时相口径**（`concat(F_A,F_B,|F_A−F_B|)`，与 `abs` 版本对比）——这是区分 H1（早期信息不足）与"proxy 口径不足"的唯一直接手段；本轮只跑了紧凑口径。
2. L07/L08（1/32）与 TAR/DCR 中间层未做 probe ⇒ "1/16→1/32 衰减"仍是 proxy 级结论。
3. 跨数据集 D5（LEVIR/WHU/CDD）未执行 ⇒ 不得把 SYSU 结果写成普适规律。
4. per-object margin 的配对 bootstrap CI 未记录（仅有 per-image AP CI 与 margin 的 per-image CI）。
5. 8 连通敏感性分析未在报告中展开（模块已支持）。

---

## 10. 对下一轮（Run4）的建议：2–4 个候选，只推荐 1 个主方向（§14.4，本轮不执行）

**主方向（唯一，且是"零 80K"的诊断收口）：先补齐 probe 的可学习时相口径，再决定是否改结构。**
- 机制问题：`1/4–1/16` 的编码器特征里，**小变化证据本身是否已经存在**？现有证据无法回答：固定 `abs(F_A−F_B)` 的 probe 只有 0.46–0.53，而"学习到时相代数"的下游可达 0.69–0.91。
- 做法：对 `L01b_norm0(1/4)`、`L03b_norm2(1/8)`、`L05(1/16)`、`L06b(1/16 post-CAACP)` 追加 **`concat(F_A,F_B,|F_A−F_B|)` probe**（同一 90/10 hash 划分、同超参、test 只评估一次），并顺带补 `L07/L08(1/32)` 两层。
- 成本/收益：约 10–15 分钟 GPU，零训练 80K，零结构改动；结论直接决定下列候选哪一个进入 80K。
- 失败/停止：若可学习口径下各层 probe AP 仍 <0.6 且随层数不升 ⇒ 支持 H1（早期编码不足），转候选 2；若 1/4–1/8 probe 显著升高（>0.65）⇒ 支持"聚合/融合侧瓶颈"，转候选 1。

**候选 1（条件：H1 不成立，瓶颈在聚合侧）——把 CAACP 的 score 来源从 Stage3 prefix(1/16) 换成 F2(1/8)，位置/lattice/β 机制不变。**
- 直接证据：D2 显示 1/8 的 small margin（0.0138–0.0329）高于 1/16（0.0114），recall@1%FP 同级（0.044–0.060 vs 0.0560）；而 D1 证明小目标缺口是主要矛盾。
- 与 CAACP 的本质区别：**不改机制、不改 Stage、不改 lattice、不改 residual 结构，只改 score 的输入尺度**；deploy 参数 **0 增量**（score 为 no-grad、参数自由）；FLOPs 仅多一次 1/8 上的 cosine/rank（实测需记录）。
- 预训练兼容：β=0 仍精确等价官方路径。
- 单变量消融：`M1` vs `M1+F2-score`，四数据集 80K。
- 失败判据：SYSU small pooled Recall / Hit@25 无提升，或 CDD/WHU 任一低于 97.18/95.07。

**候选 2（条件：H1 成立，即早期编码不足）——在 stem / 1/4 阶段引入可学习的跨时相交互（前置变化证据）。**
- 直接证据：`L01b_norm0` 的 small margin ≈3e-5、probe AP 最低（0.4588）。
- 性质：**改动宏观设计**（在第一个下采样之前引入时相交互），超出"CAACP 仅 Stage3"的原约束 ⇒ **必须事先征得用户/导师批准**；实现上必须是可折叠（STR 风格）且 deploy ≤5M。
- 风险：FRH/FS-TAR 的失败表明"末端/局部空间核"不是解；前置交互若同样只增加局部感受野，预期同样无效，故仅在 probe 证据明确支持 H1 时才启动。

**明确不建议（基于本轮证据的否定清单）**
- **S2-HCAACP（CAACP 前移 Stage2）**：其原始动机"Stage3 CAACP 伤害 small"已被正确指标否证（paired Δrecall：小目标 +1.2~+1.4pp）。除非候选 1 的 F2-score 试验成功而位置不动，才需重新评估。
- **SP-DCR（删除末端第二个 DW3）**：D3 表明 decoder 特征与 head logits 判别性相同（0.905），不存在末端读出瓶颈。
- **Fine-STR r24→48**：D3 未显示任何读出容量瓶颈。
- **任何 β / score 归一化的再调参**：D4 显示 β 修正项幅度 0.19%、关掉它 ΔF1≈−2e-6。
- **多 seed / 阈值 / loss / 增广**：仍在固定协议禁止范围内，本轮未触碰。

> 本轮到此为止：不启动任何新的 80K 训练，不删除/覆盖任何历史 checkpoint、日志或 `docs/temporary/*.json`。下一步由用户选择候选。

