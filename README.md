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
>    - D2：SYSU 损失集中在 **small components（F1 仅 0.1796，vs large 0.8273）**与
>      boundary 环带（band2 0.6710 最低）——FRH 的 3×3 邻域修正解决"边界锐化"而非
>      "极小目标语义召回"，与 SYSU 负向一致。
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
>    - RA 把 SYSU small 从 0.1796 微抬到 **0.1842**（E1~E5 五个改动中最高），但整体
>      Recall −3pp——**推翻"residual cancellation 是 small 主因"**：change-aware c
>      参与 residual 减法恰恰是 CAACP 高 Recall 的来源。
>    - FS-TAR 未恢复 tiny evidence（small R 0.1477 反而最低）——**空间核扩大不是解**，
>      问题在更早的语义编码层。
> 3. **五轮结构尝试（E1~E5）稳定模式**：LEVIR 0~+0.22（始终 ≤+0.25）、SYSU 全部
>   −0.23~−0.62——M1 仍是 SYSU 最优模型；score/头/残差/时相核四个方向的"后端修正"
>   已系统性耗尽。
> 4. **下一步（决策树指向）**：**S2-HCAACP**（CAACP 前移 Stage3→Stage2/1/8，small 在
>   1/8 仍有 <4 cells 可解析，先验 SYSU +0.25~+0.65）与 **SP-DCR**（删末端第二个 DW3，
>   减少 spatial mixing 压制弱响应，先验 +0.15~+0.45）——各自单变量、不叠模块。
> - 报告：`docs/temporary/run3_report.md`；Z 复盘：`docs/temporary/run3_z12_summary.md`；
>   脚本：`train_scripts/CASA-TViM/Run3/`。

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
analyse/                       # 分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标 → docs/temporary/*.txt
  run2_zero_cost_diag.py       #   D1/D2 零成本诊断（score 分组 + boundary/component F1）
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

### 小目标瓶颈分阶段诊断（Diag1，只读）

任务文档：`docs/temporary/CASA-CD 小目标瓶颈分阶段诊断｜DSH 完整执行文档.md`；
一键流水线：`train_scripts/CASA-TViM/Diag1/run_diag.sh`（P0 指标口径 → D0 复算/折叠 → D1 错误画像 →
D2 分阶段 hook → D3 冻结 probe → D4 β 反事实 → 报告，Gate 不 PASS 即停）。
产物（服务器）：`/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1/`
（`DIAGNOSIS_REPORT.md`、`RUN_MANIFEST.json`、`REPRODUCE.md`、各阶段 `gate.json`/`summary.json`）；
本地副本：`docs/temporary/CASA-TViM_Diag1/`。

## 注意事项

- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（train.py 已处理）。
- `.sh` 脚本需 LF 行尾（Windows 编辑后由 `_deploy.py` 上传，本地已确认 LF）。
- 训练脚本必须导出 `LD_LIBRARY_PATH=.../torch/lib:.../env/lib`（oflex kernel 依赖）。
- `run_all.sh` 的 `wait` 必须传 `$pid`（数值），直接传变量名会被 bash 当作 job 名。
