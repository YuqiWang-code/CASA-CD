## GitHub 更新（手动操作）

代码或文档更新后，手动同步到 GitHub：

```bash
cd f:/Code_Repositories_2/CursorCode/CASA-CD
git add -A
git commit -m "<描述性提交信息>"
git push origin main
```

- 预训练权重 `pretrained_weight/*.pth` 不进 git，更新时上传到 GitHub Releases
  （仓库页 → Releases → Draft a new release → 上传附件 `.pth`）。
- `docs/参考文献/` 的调研 PDF 随 `docs/` 一并提交（`.gitignore` 未忽略 PDF）。

---

# CASA-CD

CASA-TViM-STRNet：超轻量遥感二值变化检测（变化感知上下文聚合 + 结构重参数化）。

硕士课题：极轻量遥感二值变化检测。当前主线 **CASA-TViM-STRNet** =
TinyViM-S-Slim（ICCV 2025 主干瘦身版）+ CAACP-SS2D（创新一）+ TAR/DCR（创新二），
deploy 4.880M ≤5M。早期路线（ChangeViT baseline / CASAA / UltraLight / STR-Fusion /
CASA-STR/SHViT）已全部归档——历史实验记录见 git 历史、
`docs/temporary/过去的想法/` 与 `docs/experiment_metrics.xlsx`（历史 sheet 行），
本 README 只保留当前主线。
服务器与数据规范见
[`docs/RSML-3_服务器环境与变化检测数据统一说明.md`](docs/RSML-3_服务器环境与变化检测数据统一说明.md)。

## 研究定位与约定

- **方法创新导向**：核心是方法创新（变化感知上下文聚合 + 结构重参数化），
  不做工程化堆叠，也不把 loss 调参 / 训练技巧包装成创新贡献。
- **最终硬目标（必须同时全部达到）**：四数据集 F1——**SYSU ≥85、LEVIR ≥92.5、
  WHU ≥95、CDD ≥98**（IoU 与 F1 同方向）；同时满足：**有效推理（deploy）参数 ≤5M**、
  训练协议（BCE+Dice、Adam 2e-4、poly+200 warmup、80K、batch 32、seed 16、
  test-as-val、threshold 0.5）、单 seed、预注册 gate、不改 loss/增广/阈值，
  以及**创新性、故事性、可解释性、轻量化**四项要求。
- **从头训练纪律**：每个实验（主实验与全部消融对照）一律从头训练——同一 ImageNet
  预训练权重 + 固定 seed 构建后完整 80K，禁止用已有 checkpoint 微调/续训作为实验组；
  实验目的是证明模块本身的有效性（唯一变量），不是工程化堆 SOTA。
- **单 seed**：当前阶段只用单 seed 验证有效性与创新性，不做多 seed 统计显著。
- **结构改动纪律**：新模块 β/γ/aux 分支一律零初始化（epoch-0 与官方预训练逐位一致）；
  重参数化分支必须可折叠为单卷积并通过 train↔deploy 等价性 smoke
  （0.5 二值 disagreement = 0）；CAACP 只允许改 Stage3 末个 TViM 的低频池化。

## 方法

**CASA-TViM-STRNet**（deploy 4,880,190 参数 / FLOPs 2.93G / 256² 输入）：

```
[A;B] (2B) → TinyViM-S-Slim shared
    stem → F1(1/4,48) → F2(1/8,64) → stage2 prefix(1/16,168)
    → [CAACP 共享 score] → stage2 final TViM → F3(1/16,168)
    → slim stage4 → F4(1/32,224)
per-time [F1,F2,F3,F4] → MultiScaleTAR(encoder_dims=(48,64,168,224), D=96)
    → DCRDecoder(D=96) → head Conv1x1 → bilinear → sigmoid
```

- **主干 TinyViM-S-Slim**（ICCV 2025，1000e ImageNet EMA 权重，744 键逐位继承）：
  Stage4 深度裁剪（Local×3 + final TViM）+ key 重映射；1/4–1/32 四级特征；
  Laplace 频率解耦（低频下采样进 SS2D + 高频 RepDW dense 路径完整保留）。
- **创新一 CAACP-SS2D**：Stage3 末个 TViM 的低频 2×2 AvgPool 替换为
  **双时相变化分数加权的 2×2 cell 聚合**（A/B 共享权重、规则网格、dense 路径不变）：
  `c = c_avg + β·(c_ca − c_avg)`，β=0 初始化 → epoch-0 精确恢复官方预训练。
  变体：`score_mode=rank`（Run1 公式 `w∝ε+rank`）/ `cp`（Run2 公式 `w∝1+s·r`）；
  `residual_mode=current` / `avg_anchor`（Run3 E4）。
- **创新二 STR（TAR/DCR）**：TAR = 每尺度双时相代数重参数化
  （concat + sum + signed-diff 三分支，aux 零初始化，deploy 折叠为单 1×1；
  Run3 E5 提供 stage1 专用 `TemporalRepFine3x3` 折叠为单 3×3）；
  DCR = 可折叠解码器（RepLocalBlock/RepPW1x1，deploy 无 BN 残留）。
  可选 `frh=1` 时 head 换 STRFineHead（128² 重参数化头，deploy 折叠单 3×3，+768）。
- **SS2D kernel 后端**：`selective_scan_cuda_oflex`（自建 .so，需 LD_LIBRARY_PATH 指向
  torch/lib；已对拍 torch 参考 <2e-5）；官方 kernel 在 torch 2.14/CUDA 13.2/sm_120
  无可用 wheel。兜底 mamba-ssm / vendored Triton。
- 模型入口：`models/train.py`（训练，`--arch casa_tvim_str`）、`models/eval.py`（独立测试）、
  `models/smoke_test.py`（冒烟，`--mode casa_tvim_str`，T-CA-1..10）
- 损失 BCE + Dice；`max_steps=80000`、batch 32、256×256、seed 16、lr 2e-4（poly）
- 日志：开头全部配置 + 参数/FLOPs；每 epoch 一行六项指标（test 集）；
  结尾 `=== TEST RESULTS ===` 正式测试区块（含 deploy 折叠等价性、
  [CAACP-BETA]/[CAACP-WEIGHT-ENTROPY]/[PRETRAIN-DRIFT] 等诊断行）

## 实验结果（CASA-TViM Run1 · CAACP-SS2D 主线，已完成）

> **当前主线（2026-10-03 起）**：按
> [`docs/temporary/过去的想法/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md`](docs/temporary/过去的想法/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md)
> 执行——**CASA-TViM-STRNet = TinyViM-S-Slim + CAACP-SS2D + TAR/DCR**。
>
> 机器验证（服务器 GPU smoke T-CA-1..6 全绿）：slim trunk **4,645,180** 参数；
> 预训练 744 键逐位继承；epoch-0 恒等 bitwise 0（β=0 与 rep aux 双验证）；
> β/x_proj/A_logs/stem 梯度链非零；change score T1/T2 交换对称；
> **deploy 4,880,190 ≤ 5M**、FLOPs 2.93G。
>
> 训练：16 run（4 变体 × 4 数据集）全部完整 80K，run_all.sh 单脚本串 4 波、
> 每波 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发）、batch 32；
> 崩溃自动续训、整波失败即中止。
>
> | 变体 | caacp | rep | CDD | LEVIR | SYSU | WHU | 状态 |
> |---|---|---:|---:|---:|---:|---|
> | A0_TVIM_PLAIN | 0 | plain | 0.9676 / 0.9373 | 0.9108 / 0.8362 | 0.8238 / 0.7006 | 0.9478 / 0.9008 | 完成（vs SHViT 版 A0 +2.09 / +1.18 / −0.08 / +1.08） |
> | M1_FULL | 1 | full | **0.9718 / 0.9451** | 0.9107 / 0.8360 | **0.8347 / 0.7163** | **0.9507 / 0.9060** | 完成（Δ vs A0 +0.42 / −0.01 / **+1.09** / +0.29；**WHU 达标**） |
> | A1_CAACP | 1 | plain | 0.9675 / 0.9370 | 0.9103 / 0.8353 | 0.8211 / 0.6965 | 0.9484 / 0.9019 | 完成（CAACP 单独≈A0） |
> | A2_STR | 0 | full | 0.9711 / 0.9439 | **0.9119 / 0.8381** | 0.8321 / 0.7125 | 0.9453 / 0.8963 | 完成（STR 单独：CDD +0.35 / LEVIR +0.11 / SYSU +0.83 / WHU −0.25） |
>
> **Run1 结论（16/16 全部完成，全部 disagree 带内、deploy 4.880M ≤5M）**：
> 1. **骨干 floor 大幅抬升**：TinyViM-S-Slim A0 比 SHViT 版 A0 高 CDD +2.09 /
>    LEVIR +1.18 / WHU +1.08（SYSU 持平）。
> 2. **CAACP-SS2D（创新一）机制成立且 β 被训练采纳**（β 5.6e-3~3.9e-2）：单独 ≈ A0，
>    但 M1−A2 边际 SYSU +0.26 / WHU +0.54——变化感知上下文聚合与 STR-rep 组合
>    在 SYSU/WHU 上有条件互补；M1 完整方法 SYSU **+1.09pp**（A0 0.8238→0.8347）。
> 3. **STR-rep（创新二）跨骨干保持数据集依赖模式**：CDD/LEVIR/SYSU 正向（SYSU +0.83）、
>    WHU 负向——transferability 再次验证。
> 4. **硬目标**：**WHU 95.07 ≥ 95 ✓（达标）**；CDD 97.18（−0.82）、LEVIR 91.19
>    （−1.31）、SYSU 83.47（−1.53）逼近但未达。对照 29.57M 的 VMamba-Tiny
>    full_last2（98.42/91.44/83.45/95.14）：本模型以 **1/6 参数（4.88M）** 达到
>    LEVIR −0.25 / SYSU +0.02 / WHU −0.07 / CDD −1.24 的接近水平。
> - 汇总：`docs/experiment_metrics.xlsx`（CASA-TViM/Run1 16 行）；快照与指标表：
>   `docs/temporary/CASA-TViM_Run1/models_and_metrics_CASA-TViM_Run1.txt`、
>   `docs/temporary/CASA-TViM_Run1/metrics_tables.md`（4 变体 × 4 数据集 × 六指标 + 训练/推理参数 + FLOPs）；
>   架构图：`docs/temporary/CASA-TViM_Run1/*.png`。

## 实验结果（CASA-TViM Run2 · LEVIR/SYSU 定向提升，已完成）

> **目标**：保持 CDD ≥ 97.18、WHU ≥ 95.07，同时把 LEVIR 91.07 → ≥ 92、SYSU 83.47 → ≥ 84。
> 按
> [`docs/temporary/过去的想法/CASA-CD_Run2_LEVIR_SYSU定向提升_证据调研与结构改进方案_2026-10-04.md`](docs/temporary/过去的想法/CASA-CD_Run2_LEVIR_SYSU定向提升_证据调研与结构改进方案_2026-10-04.md)
> 执行，两项纯结构性改进（无 loss/aug/threshold 调参）：
>
> - **CP-CAACP（首选一，0 新参数）**：CAACP 的 cell 权重从 `w∝ε+rank` 改为
>   `w∝1+s·r`（s=1−cos 余弦变化 × rank 归一化，置信度保留）。零变化像素 q=0 →
>   全零变化 cell 严格退化为均匀池化。`--caacp_score_mode cp`。
> - **FRH（首选二，deploy +768 参数）**：预测头 64² 1×1 → **128² 重参数化细粒度头**
>   `up2 → base 1×1 + γ·[RepDW3→1×1]`（γ=0 初始化），deploy 折叠为单个 3×3 Conv(96→1)。
>
> 机器验证（smoke T-CA-7/8 全绿，Run1 回归 T-CA-1..6 不变）；12 run（3 变体 × 4 数据集）
> 完整 80K，batch 32、seed 16，协议与 Run1 逐项一致。
>
> | 变体 | caacp | score | frh | CDD | LEVIR | SYSU | WHU | 状态 |
> |---|---|---|---:|---:|---:|---:|---|
> | E1_CP_CAACP | 1 | cp | 0 | 97.14 / 94.45（−0.04） | 91.14 / 83.72（**+0.07**） | 83.13 / 71.12（−0.34） | 95.18 / 90.81（**+0.11**） | 完成（唯一 CDD/WHU 双保变体；WHU 达标） |
> | E2_FRH | 1 | rank | 1 | 97.17 / 94.50（−0.01） | 91.21 / 83.83（**+0.14**） | 82.97 / 70.89（−0.50） | 94.71 / 89.95（−0.36） | 完成（LEVIR 正、SYSU/WHU 反） |
> | E3_CP_FRH | 1 | cp | 1 | **97.18 / 94.51**（±0.00） | **91.29 / 83.98**（**+0.22**） | 82.85 / 70.72（−0.62） | 94.98 / 90.44（−0.09） | 完成（LEVIR 最好但 SYSU 最差） |
>
> **Run2 结论（12/12 全部完成，全部 disagree 带内、deploy 4.880~4.881M ≤5M）**：
> 1. **LEVIR 定向正向但幅度远小于目标**：+0.07 / +0.14 / +0.22（91.07 → 91.29 峰值），
>    未达 ≥92；FRH 的细粒度收益与机制叙事一致，但不足以跨过门槛。
> 2. **SYSU 反向退步（−0.34 / −0.50 / −0.62）**：CP 与 FRH 对 SYSU 均为负向，
>    组合更差；文档 §15.5/15.6 预注册的"失败即停"分支激活。
> 3. **WHU/CDD 保持**：E1 是唯一 CDD/WHU 双保变体（97.14 / 95.18，WHU 达标）。
> 4. **D1/D2 零成本诊断（`docs/temporary/run2_zero_cost_diag.json`）机制证据自洽**：
>    - D1：LEVIR 零变化图（占 54%）rank cell 熵 **1.2504 四数据集最低**且 abs 分数低
>      ——rank-only 在零变化图上最强制不均匀，CP 的修复方向证据最强；但 +0.07/+0.22
>      的幅度说明 rank-confusion 不是 LEVIR 的主瓶颈。
>    - D2：SYSU 运行期 D2 诊断曾报告"small components F1 仅 0.1796"——**该数值来自后续
>      Diag1 证实的伪指标（FP 恒为 0），已作废**；正确口径见下方「Diag1」章节
>      （small pooled pixel Recall **0.1974**、ObjRecall 0.7674、ObjPrecision 0.5320）。
>      FRH 对 SYSU 负向的结论仍成立（F1 −0.50），但"因为小目标召不回"的机制解释已被 Diag1 修正。
> - 自动对比报告：`docs/temporary/run2_report.md`；快照：
>   `docs/temporary/models_and_metrics_CASA-TViM_Run2.txt`；
>   汇总：`docs/experiment_metrics.xlsx`（CASA-TViM/Run2 12 行）。

## 实验结果（CASA-TViM Run3 · SYSU 小目标瓶颈定向，已完成）

> 按
> [`docs/temporary/过去的想法/CASA-CD_Run2复盘_SYSU小目标瓶颈与下一轮结构实验设计_2026-10-05.md`](docs/temporary/过去的想法/CASA-CD_Run2复盘_SYSU小目标瓶颈与下一轮结构实验设计_2026-10-05.md)
> 执行。复盘结论：Run2 的 E1/E2/E3 在 SYSU 都是 Recall 被压掉 3.5~4.3pp（Precision 反升），
> 交互项全非负 → 不是优化冲突，而是**机制/尺度错配**；SYSU 最大缺口 = small components
> （<256px）F1 0.1796，这类目标在 1/16 已是 sub-token。
>
> - **E4 RA-CAACP（创新一升级，0 参数）**：residual 从 `x−Up(c)` 改为 `x−Up(c_avg)`。
> - **E5 FS-TAR（创新二升级，+73,728 deploy ≈ 4.954M）**：stage1（1/4）TemporalRep1x1 →
>   TemporalRepFine3x3（signed-diff 分支变 3×3），deploy 折叠为单 3×3 Conv。
>
> 开训前证据链：Z1/Z2 零成本复盘确认 H-E1/H-E2（E1/E2/E3 的 SYSU small Recall
> 全部 0.153~0.159，`docs/temporary/run3_z12_summary.md`）；smoke T-CA-9/10 全绿。
>
> | 变体 | CDD | LEVIR | SYSU | WHU | 状态 |
> |---|---|---:|---:|---:|---|
> | E4_RA_CAACP | 97.13 / 94.42（−0.05） | 91.07 / 83.60（±0.00） | 83.16 / 71.17（−0.31） | 95.02 / 90.50（−0.05） | **FAIL**（CDD/WHU/SYSU 三硬条件未过） |
> | E5_FS_TAR | 97.13 / 94.43（−0.05） | 91.18 / 83.78（**+0.11**） | 83.24 / 71.29（−0.23） | 94.91 / 90.31（−0.16） | **FAIL**（硬条件 + small 均未过） |
>
> **Run3 结论（8/8 全部完成，全部 disagree 带内、deploy 4.880/4.954M ≤5M）**：
> 1. **E4/E5 均 FAIL 预注册条件**（`docs/temporary/run3_report.md` 全条件裁决）。
>    按决策树：**停 RA 线**（不再调 Stage3 residual/β/叠 CP）、**不再扩大 spatial kernel**。
> 2. **两个关键负结果（机制证据价值高）**：
>    - RA 曾被认为"把 SYSU small 从 0.1796 微抬到 0.1842"——**该组数值已被 Diag1 证伪
>      （伪指标）**；Diag1 用正确口径的 same-object 配对显示 CAACP **提升**小目标 per-object
>      recall（tiny_16_63 **+1.24pp**、small_64_255 **+1.39pp**），代价是中/大目标
>      （medium −4.17pp、large −4.69pp），像素级 F1 净 +0.26pp。
>    - FS-TAR 未恢复小目标（correct 口径下小目标 margin 仍下降）——**空间核扩大不是解**，
>      与 Diag1「瓶颈在尺度选择性可读性」一致。
> 3. **五轮结构尝试（E1~E5）稳定模式**：LEVIR 0~+0.22（始终 ≤+0.25）、SYSU 全部
>   −0.23~−0.62——**M1 仍是 SYSU 最优模型**；score/头/残差/时相核四个方向的"后端修正"
>   已系统性耗尽（Diag1 进一步证明：β 项 0.19%、head 与 decoder 可读性相同）。
> 4. **下一步**：Run3 决策树曾指向 **S2-HCAACP** 与 **SP-DCR**，但 **Diag1 已撤回其动机**
>   （CAACP 块配对 margin −1.1%、无读出瓶颈）。当前唯一推荐候选见下方「Diag1」章节。
> - 报告：`docs/temporary/run3_report.md`；Z 复盘：`docs/temporary/run3_z12_summary.md`；
>   脚本：`train_scripts/CASA-TViM/Run3/`。

## 诊断（CASA-TViM Diag1 · SYSU 小目标瓶颈分阶段诊断，已完成）

> 任务文档：`docs/temporary/CASA-CD 小目标瓶颈分阶段诊断｜DSH 完整执行文档.md`；
> 流水线：`train_scripts/CASA-TViM/Diag1/run_diag.sh`（P0/D0/D1/D2/D3/D3c/D2b/D5/D1conn8/D4/report，
> Gate 不 PASS 即停）；报告：服务器 `$DIAG/DIAGNOSIS_REPORT.md`（本地副本
> `docs/temporary/CASA-TViM_Diag1/`）。**只读诊断，未训练任何新 80K，未覆盖历史产物。**

**Gate**：P0/D0-M1/D1/D2/D3/D3_concat/D3_stratified/D4/D1_conn8 及 D5（CDD/LEVIR）全 PASS；
D5-WHU D1 = WARN（small 仅 79 个对象，仅描述性）；**D0-A2 = FAIL**（固定真实 batch 上 1 个像素
（1/1,048,576）概率恰在 0.5 的刀锋翻转，如实保留、未用于任何结论）。

**1. 指标口径修复（旧结论作废）**：`analyse/run2_zero_cost_diag.py::component_pr` 的 FP 被 GT 局部
掩码过滤后**恒为 0**，故旧 `small 0.1796 / medium 0.4620 / large 0.8273` 是**受限正样本伪指标**。
新口径（4 连通、`p>0.5`、原生 256²、真实对象匹配）：

| GT 面积组 | 对象数 | pooled pixel Recall | object-macro Recall | Hit@25% |
|---|---:|---:|---:|---:|
| tiny_1_15 | 66 | 0.0752 | 0.1263 | 0.1364 |
| tiny_16_63 | 96 | 0.1245 | 0.1483 | 0.1667 |
| small_64_255 | 202 | 0.2068 | 0.1863 | 0.2178 |
| medium_256_1023 | 692 | 0.4213 | 0.3988 | 0.4855 |
| large_1024_inf | 4706 | **0.8477** | 0.7491 | 0.8725 |

small(1–255px) pooled **0.1974**、Hit@25 **0.1896** ⇒ **size gap 65.0pp**；对象级（IoU≥0.10）
**ObjPrecision 0.5320 / ObjRecall 0.7674 / ObjF1 0.6284**，3077 个未匹配预测连通域中 **2430 个 <256px**
（小目标区域同时存在漏检与碎片化假阳性）。8 连通敏感性下结论不变。

**2. 分阶段证据（D2/D3，SYSU，4000 张）**
- **整体变化信息充足**：可学习时相口径 probe（`concat(F_A,F_B,|Δ|)`）AP — 1/4 **0.547**、1/8 **0.623**、
  1/16 **0.789**、1/32 **0.829**（L07 为编码器最可分层）⇒ **H1（早期编码不足）否证**；
  固定 `abs(F_A−F_B)` 口径只有 0.46–0.53（口径差异已记录，容量 caveat 已写入协议）。
- **输出端不是瓶颈**：decoder refine probe 0.9053 ≈ head logits 0.9049。
- **D2 的"1/16→1/32 信息衰减（−41%）"被证伪**：属 cosine proxy 失效（§6.4），不是信息丢失。
- **CAACP β 修正项可忽略**：`||βΔC||/||c_avg|| = 0.19%`；β 置零 ΔF1 = −2.2e-6、small Hit@25 不变；
  CAACP 块的配对 margin 变化仅 **−1.1%**（未过 10% 判据）。

**3. 本轮新增主结论 H5 —— 尺度选择性可读性坍塌（D3c + D2b + D5 三向一致）**
- **分层 probe**：整体可读性几乎全部来自大目标（small-vs-背景 AP **0.00045–0.0043**，large **0.55–0.91**）；
  **small-object margin 随深度单调坍塌** 0.0779(1/4) → 0.0400(1/8) → 0.0238(1/16) → 0.0152(CAACP 后)
  → **0.0022(1/32)**，而 large margin 反升至 0.21；**small/large margin 比 0.64 → 0.013**，
  决策头 P00 为 0.0281 vs 0.3732（**13× 失衡**）。
- **配对 cosine margin CI**：仅 3 个边界过 ≥10%+同号判据 —— Stage3 前缀内 `L03→L03b` **−11.6%**、
  `L04→L05` **−23.2%**、`1/16→1/32` **−75.6%**（与 probe margin −86% 同向）。
- **跨数据集 D5 一致性**：SYSU small pooled Recall **0.1974** vs LEVIR **0.7046** / CDD **0.7369** /
  WHU **0.6420**（低 3.3–3.7 倍），SYSU **ObjPrecision 最低（0.5320）**、变化先验最高（0.236）
  ⇒ **SYSU 的小目标问题是数据集特异的严重异常**。

**4. 下一轮唯一推荐候选（A，待 deploy 预算核算）**：把 **1/4–1/8 的细尺度变化证据**（small margin
最高处）以 **可折叠 + γ=0 零初始化门控**接入 DCR 的 64² 层级；与 FRH（末端加头）和 E5 FS-TAR
（只改 1/4 时相代数的核形状）的本质区别是"把浅层小目标证据显式送进决策层"。
**注意**：3×3 Conv(144→96) 折叠 ≈ +124.5K 参数会把 deploy 推到 ~5.005M **超 5M**，须改 1×1
折叠（+13.9K）或降通道。候选 B（减少 1/16→1/32 的 small-margin 坍塌）属骨干改动，须导师批准。

## Run4（CASA-TViM R4-FET1 · 1/4 细尺度证据 1×1 可折叠旁路，进行中）

> 设计文档：`docs/temporary/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md`；
> 脚本：`train_scripts/CASA-TViM/Run4/`（`README.md` 含完整执行顺序与预注册门槛）；
> 代码身份：`train_scripts/CASA-TViM/Run4/SOURCE_IDENTITY.json`（与 Diag1 manifest 的逐文件差异）。
> **状态：代码已实现并部署到 RSML-3；全部训练前门已 PASS；两波 8×80,000 步正在训练。
> 本节不预填任何 Run4 TEST 成绩，也不声称已达成论文目标。**

**唯一结构变量 `fine_tap`（其余全部相同）**

| 变体 | `fine_tap` | 说明 |
|---|---:|---|
| `M1_R4CTRL` | 0 | 同期唯一变量对照，先跑 |
| `E6_FET1` | 1 | 唯一正式主实验，后跑 |

`P=f1a`、`Q=f1b` 取自共享编码器 `norm0`（1/4、48C、64²）：`D=|Q−P|`，
`U=Conv1x1([P,Q])+b`，`V=Conv1x1_noBias(D)`，`T=γ(U+V)`，`R'=R+T`（加在 DCR refine 之后）。
`γ`、`W_diff` 零初始化且新模块最后构造（`fork_rng` 局部 RNG）⇒ epoch-0 与关掉 FET **逐位一致**。
训练图 **+13,921**、部署图折叠为单条 `Conv1x1(144→96)` **+13,920** ⇒ deploy
**4,894,110 ≤ 5,000,000**（CTRL 4,880,190）。

**exact-80K（P0 协议修正）**：`--exact_max_steps 1` 在第 80,000 次 `optimizer.step()` 处严格停止，
poly 归一化分母固定 `args.max_steps=80000`；历史整 epoch 口径仍为默认（`exact_max_steps=0`），
两个变体同版本同实现。

**训练前门（全部 PASS，2026-10-10，RSML-3）**

| 门 | 结果 |
|---|---|
| 代码身份 | 服务器 SHA256 与本地/`SOURCE_IDENTITY.json` **逐条相同**；Diag1 的 12 文件里仅 `casa_tvim_str_net.py`/`train.py`/`eval.py` 改动，其余逐字节相同 |
| 现有 `casa_tvim_str` smoke | **ALL OK**（旧行为未改，含 T-CA-2 `retained=744/805 worst=0.00e+00`） |
| T0–T9 验收 | **PASS=53 / FAIL=0 / SKIP=0** |
| §10.2 互补性 preflight | **PASS**（4000 张、12.8 s；`n_missed_small=295`、`rescue_rate=0.5085`、`gap=0.1303`，按图 bootstrap 95%CI **[0.0669, 0.1934]**；复现 Diag1 small recall 0.19742 / Hit@25 0.18956、n=364） |
| 真实数据 3-step dry-run | **PASS**（E6 与 CTRL；几何增广同步、`gray>=128` 逐位一致、batch32 峰值 ≈9.8 GB/进程） |
| 对象指标口径预校验 | **`OBJECT_PASS_MATCHES_DIAG1_D5`**：四库 M1 部署图上 small recall/Hit@25/ObjP/ObjR 与 Diag1 §14 D5 逐项一致（SYSU 0.19742/0.18956/0.53200/0.76744 等） |
| T11 断点恢复 | **协议精确，非逐位**：步数/per-group LR/optimizer/四项 RNG 状态逐项恢复；对照组（同 seed 的两次连续从头 run）差异数与"连续 vs 恢复"**完全相同**（1098/1302、1288）⇒ 管线跨进程即不可逐位复现，故**不声称"精确恢复"**；已启用 `ckpt_backup_watchdog.py` 做 `last.pth.bak` 原子备份缓解写盘崩溃 |

关键实测：epoch-0 前向 `torch.equal`=True（`max_abs=0`）；整模型 train↔deploy
`max_abs=0.000e+00` 且二值 disagreement=0（随机与真实 batch=16）；FET 折叠 FP64 `1.2e-15`；
deploy fvcore **2.7313 G**（`unsupported_ops=20`，与设计预测一致）；第 1 步 `γ.grad` 非零、
γ 开门后 `W_diff` 梯度非零、encoder/TAR/DCR/head 均仍有梯度。

**预注册门槛（不得事后放宽）**：SYSU small(1–255px) pooled Recall ≥ **0.2474** 且 ≥CTRL+5.0pp，
Hit@25 ≥ **0.2396** 且 ≥CTRL+5.0pp，ObjRecall ≥CTRL+1.0pp，ObjPrecision 与 <256px 未匹配预测
不劣于 CTRL；跨库守门 CDD ≥97.18、LEVIR ≥91.07、SYSU ≥83.47、WHU ≥95.07 且不低于同期 CTRL；
四库同时 ≥98.00 / 92.50 / 85.00 / 95.00 才是 `PAPER-TARGET-PASS`。判定走
`analyse/run4_fet_report.py`，失败按设计文档 §6.4 决策树**停止该线**。

## 参考文献

- **文献总索引**：[`docs/参考文献/文献索引.md`](docs/参考文献/文献索引.md)——
  2024–2026 调研文献清单（分层级/官方链接/存放位置），新增文献 PDF 时先在此登记。
- 主干来源：`docs/参考文献/baseline/Ma_TinyViM_Frequency_Decoupling_for_Tiny_Hybrid_Vision_Mamba_ICCV_2025_paper.pdf`
  （TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba, ICCV 2025；
  S=5.6M/0.9G@224、ImageNet-1K 79.2(300e)/80.3(1000e)；代码 https://github.com/xwmaxwma/TinyViM；
  slim 重实现 `models/model/tinyvim_s_slim.py`）
- Novelty 边界文献（PDF 与差异说明见文献索引 §1.2）：SChanger / SCAM / CFNet /
  CAM-CD / SeCoR / Mamba-CD（JSTARS 2025-2026）——CAACP 与它们的分界必须落在
  "scan 前 structured context compression / 固定 cell lattice"，不是方向加权、
  显式对齐或普通 cosine focuser。
- 历史基线（已归档，仅论文背景引用）：ChangeViT（PR 2025，`docs/参考文献/baseline/`）；
  SAT（CVPR 2026 Findings，CASAA 历史机制来源）。

## 目录结构

```
models/                        # 全部代码（当前主线 casa_tvim_str）
  train.py                     #   训练入口（--arch casa_tvim_str）
  eval.py                      #   独立测试入口（同款架构参数）
  smoke_test.py                #   冒烟测试（--mode casa_tvim_str，T-CA-1..10）
  model/
    tinyvim_s_slim.py          #   TinyViM-S-Slim（Stage4 裁剪 + 预训练 key 重映射）
    casa_tvim_str_net.py       #   CASA-TViM-STRNet（当前主线主架构）
    str_tar.py                 #   TAR（TemporalRep1x1 / TemporalRepFine3x3）
    str_dcr.py                 #   DCRDecoder + RepLocalBlock/RepPW1x1
    str_reparam.py             #   折叠原语（fold_conv_bn / RepDW3 等）
    str_fine_head.py           #   STRFineHead（Run2 FRH，可选）
    layers/ss2d.py             #   SS2D 自包含移植（oflex/mamba-ssm/Triton 三后端）
    layers/caacp_ss2d.py       #   CAACP-SS2D（创新一：score_mode / residual_mode）
    layers/mamba_scan_triton.py#   vendored Triton 兜底 kernel
  dataset/                     #   DataLoader（A/B/label + list 格式）
train_scripts/
  CASA-TViM/Run1/              #   4 变体 × 4 数据集（A0/M1/A1/A2，已完成）
  CASA-TViM/Run2/              #   CP-CAACP / FRH / 组合 × 4（E1/E2/E3，已完成）
  CASA-TViM/Run3/              #   RA-CAACP / FS-TAR × 4（E4/E5，已完成）
  CASA-TViM/Diag1/             #   小目标瓶颈分阶段诊断流水线（run_diag.sh，只读；已完成）
analyse/                       # 分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标 → docs/temporary/*.txt
  tvim_object_metrics.py       #   ★正确口径：GT 面积分组 Recall / ObjectHit / 对象级 ObjP-ObjR-F1 / 边界带
  tvim_diag_common.py          #   诊断基础设施 + D0 对拍审计（audit / protocol 子命令）
  tvim_small_error_audit.py    #   D1：逐 GT 对象错误画像 + same-object 配对 Δ + 预注册样例图（--connectivity）
  tvim_stage_recoverability.py #   D2：分阶段 hook（编码器 A/B cosine proxy + CAACP 内部）
  tvim_stage_raw_stats.py      #   D2b：cosine margin 的配对 image-level bootstrap CI
  tvim_linear_probe.py         #   D3：冻结线性 probe（--encoder-input absdiff|concat）
  tvim_probe_stratified.py     #   D3c：按 GT 面积分层的 probe AP / per-object margin
  tvim_caacp_counterfactual.py #   D4：CAACP β ON/OFF 受控反事实
  tvim_diag_report.py          #   汇总报告 + 预注册判据自动评估 + RUN_MANIFEST.json
  tests/test_tvim_*.py         #   P0 单测（对象指标 10 项 + 日志解析器逐字对拍）
  run2_zero_cost_diag.py       #   ⚠️ 历史诊断（component_pr 的 FP 恒为 0，口径已被 tvim_object_metrics 取代）
  run2_report.py               #   Run2 自动对比报告
  run3_report.py               #   Run3 自动对比报告（含 §13 预注册裁决）
others/                        # 参考实现（非本仓库模型代码）
  TinyViM-main/                #   TinyViM(ICCV2025) 官方核心（tinyvim.py + tvimblock.py）
outputs/                       # 训练日志（训练结束后下载到这里）
docs/                          # 项目文档（temporary / 参考文献 / 服务器说明）
.claude/                       # 服务器部署与 SSH 辅助脚本（不进 git）
```

## 服务器环境（RSML-3）

- 环境 `casacd`：**torch 2.14.0+cu132 / Python 3.10 / CUDA 13.2**。
- GPU：2 × RTX 5090（Blackwell `sm_120`，每卡 32 GB）。
- 关键路径：
  - 代码 `/home/yqwang/projects/CASA-CD/`
  - 数据集 `/share_datasets/CD/{CDD,LEVIR,SYSU,WHU}-CD-256/`
  - 预训练权重 `/home/yqwang/projects/CASA-CD/pretrained_weight/tinyvim_s_1000e.pth`
  - checkpoint `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/<Run>/<variant>/<dataset>/`
  - 训练日志 `/home/yqwang/outputs/CASA-CD/CASA-TViM/<Run>/<variant>/<dataset>/train_log.txt`

## 数据集（A/B/label + list 格式）

| 数据集 | Train / Val / Test | 目标 F1（硬目标） | 说明 |
|---|---:|---:|---|
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | **≥98** | 抗伪变化，label 为 JPG（阈值 ≥128） |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | **≥92.5** | 建筑小目标、极不平衡 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | **≥85** | 通用地表变化 |
| WHU-CD-256 | 5,947 / 743 / 744 | **≥95** | 建筑小目标、极不平衡 |

## 训练 / 测试

1. 本地改代码 → `python .claude/_deploy.py` 同步到服务器（models/ + train_scripts/ + analyse/）。
2. 服务器启动（`nohup`，每脚本带断点续训重试循环），脚本在 `train_scripts/CASA-TViM/<Run>/`：
   ```bash
   cd /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Run3
   nohup bash run_all.sh > /home/yqwang/outputs/CASA-CD/CASA-TViM/Run3/run_all.log 2>&1 &
   ```
   run_all.sh 串行多波、每波 4 数据集并行（GPU0=CDD+LEVIR、GPU1=SYSU+WHU，每卡 2 并发
   batch 32，~19GB/卡）；任一 run 3 次 retry 失败 → WAVE-ABORT。
3. 训练日志格式：
   - 开头：全部配置 + 参数量 + FLOPs(G)。
   - 每个 epoch 一行：Loss + Recall / Precision / OA / F1 / IoU / Kappa（test 集）。
   - 结尾：`=== TEST RESULTS ===`（含 deploy 折叠等价性、β/熵/drift 诊断行）。
4. checkpoint：每 run 一个文件夹，只放 `last.pth`（断点续训，含优化器）与
   `best_F1=xxx.pth`（测试用）。
5. 训练结束后把 `train_log.txt` 下载回本地 `outputs/CASA-TViM/<Run>/`（权重不下载）。

## 运行监控 / 分析

```bash
python .claude/_ssh.py '<cmd>'          # 通用 SSH 执行
python .claude/_download_tvim_logs_run3.py   # 下载某 Run 的 train_log.txt
python analyse/extract_metrics_to_excel.py   # outputs → docs/experiment_metrics.xlsx
python analyse/models_to_txt.py --tag CASA-TViM --run Run3   # models 快照 + 指标
python analyse/run3_report.py           # 自动对比报告 + 预注册裁决
python analyse/run2_zero_cost_diag.py --ckpt_run Run3 --variants E5_FS_TAR --datasets SYSU-CD-256
                                         # D1/D2 机制诊断（服务器上跑）
```

### 小目标瓶颈分阶段诊断（Diag1，只读；已完成）

任务文档：`docs/temporary/CASA-CD 小目标瓶颈分阶段诊断｜DSH 完整执行文档.md`；
一键流水线：`train_scripts/CASA-TViM/Diag1/run_diag.sh`，阶段
`p0 d0 d1 d2 d3 d3c d2b d5 d1conn8 d4 report`（Gate 不 PASS 即停）：

```bash
cd /home/yqwang/projects/CASA-CD/train_scripts/CASA-TViM/Diag1
bash run_diag.sh                                    # p0+d0(M1)+d1+d2+d4+report
bash run_diag.sh d3c d2b d5 d1conn8                 # 分层 probe / margin CI / 跨数据集 / 8 连通
D5_DATASETS="LEVIR-CD-256 WHU-CD-256" bash run_diag.sh d5   # 指定子集（可双卡并行）
```
产物（服务器）：`/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1/`
（`DIAGNOSIS_REPORT.md`＝机器章节 + 自动追加 `INTERPRETATION.md`、`RUN_MANIFEST.json`、
`REPRODUCE.md`、各阶段 `gate.json`/`summary.json`/CSV/PNG）；本地副本：`docs/temporary/CASA-TViM_Diag1/`。

## 注意事项

- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（train.py 已处理）。
- `.sh` 脚本需 LF 行尾（Windows 编辑后由 `_deploy.py` 上传，本地已确认 LF）。
- 训练脚本必须导出 `LD_LIBRARY_PATH=.../torch/lib:.../env/lib`（oflex kernel 依赖）。
- `run_all.sh` 的 `wait` 必须传 `$pid`（数值），直接传变量名会被 bash 当作 job 名。
