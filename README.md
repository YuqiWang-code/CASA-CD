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

## 研究目标与实验纪律（**2026-10-11 修订并固定**）

> 本节是当前**唯一有效**的实验纪律，优先于 `docs/temporary/过去的想法/` 下的任何历史方案、
> 以及 Run1–Run6 各设计文档中"预注册 gate 未过即不得训练"一类条款。
> 修订原因：把「硬目标 = 必达门」改为「硬目标 = 努力方向」，并把「前测不过不开训练」
> 改为「诊断从宽、以服务器 80K + 消融为准」。

### 1. 硬目标是「努力方向」，不是「必达门」

- 四数据集 F1 硬目标：**CDD ≥98.00 / LEVIR ≥92.50 / SYSU ≥85.00 / WHU ≥95.00**
  （IoU 与 F1 同方向）。经调研，**达到这组数值即可确保 SOTA**，因此它决定我们的努力方向与取舍。
- **但它不是成功判据**：未达到 **≠** 方案不行、**≠** 实验失败。只要在**受控消融**下相对
  同期 CTRL 有可复现的提升，就是有效性证据。
- 论文如实报告与硬目标的差距；**不得**把「未达硬目标」写成「方法无效」，也**不得**把
  「未达硬目标」当作放弃某条机制线的理由。

### 2. 有效性由「服务器 80K 训练 + 消融」判定（训练优先）

- **每一轮实验、每一次改进与迭代，都必须上服务器跑完整 80,000 step 拿到指标。**
  服务器 RSML-3（2×RTX 5090 32GB）资源充足、基本每天无人使用 ⇒ **能用就用**；
  不要为了省算力而少跑、短跑或只做本地推断。
- **训练前的只读诊断从宽**：只要「大概能判断有效性、不会白跑」——链路能跑通、
  前向与折叠 smoke 不炸、deploy 参数门与协议门不炸、机制方向不与被数据否决过的结论直接冲突——
  就**直接上服务器训练**，不再要求通过多层预注册硬门才放行。
- **有效性最终由消融实验判定**，且**消融同样在服务器跑完整 80K**（不是短跑、不是只读推断）。
  最小消融范式：`CTRL`（同期同代码、开关关闭）vs `EXP`（唯一变量打开），
  同 seed / 同协议 / 同预算，比较**四库 F1** + 小目标对象级指标 + 未匹配小预测数量。
- 每轮的**负面结论同样要记录**（为什么没提升），这既是论文「负结果 + 机制归因」一节的素材，
  也是避免重复踩坑的依据。

### 3. 规模、协议与结构纪律（**仍然不变**）

- deploy 有效参数 **≤5,000,000**（当前基线 4,880,190，余量 119,810）；单 seed；
  四库各从同一 ImageNet 预训练权重**独立从头训练**；每库严格 **80,000** `optimizer.step`；
  batch 32；256×256；`gray>=128`（由 `Transforms.Normalize` 应用）；`prob > 0.5`（严格大于）；
  Adam(2e-4, 0.9/0.99, wd=1e-4)；poly^0.9 + 200 iter warmup；test-as-val。
- **方法创新导向**：核心是方法创新（变化感知上下文聚合 + 结构重参数化），
  不做工程化堆叠，也不把 loss 调参 / 训练技巧包装成创新贡献。
- **从头训练纪律**：每个实验（主实验与全部消融对照）一律从头训练——同一 ImageNet
  预训练权重 + 固定 seed 构建后完整 80K，禁止用已有 checkpoint 微调/续训作为实验组；
  实验目的是证明模块本身的有效性（唯一变量）。
- **单 seed**：当前阶段只用单 seed 验证有效性与创新性，不做多 seed 统计显著。
- **结构改动纪律**：新模块 β/γ/aux 分支一律零初始化（epoch-0 与官方预训练逐位一致）；
  重参数化分支必须可折叠为单卷积并通过 train↔deploy 等价性 smoke
  （0.5 二值 disagreement = 0）；CAACP 只允许改 Stage3 末个 TViM 的低频池化。
- **训练目标**：主损失固定 BCE+Dice。**追加辅助损失已获批准（2026-10-11，仅限边缘监督）**：
  形式为在编码器前两层频率分解的**高频分支**（`local_conv` 输出）上接轻量边缘头，
  用 GT 边缘图做辅助监督；**权重 λ=0.2 固定，禁止搜索**；辅助分支只做监督、
  **不得影响推理**（train↔deploy 等价性 smoke 必须 disagreement = 0）。
  其余任何 loss 改动仍属待批准事项，见 §4。
- **代码身份链必须可从 git 复现**（2026-10-11 修补）：`.gitattributes` 除 `*.sh` 外
  另有 `* text=auto eol=lf`，把全部文本文件钉成 LF——因为 `core.autocrlf=true` 是
  Windows 系统级默认值，否则任何人 clone 都会把 `.py/.json/.md` 检成 CRLF，
  使 `SOURCE_SHA256.txt` / `SOURCE_IDENTITY_FILES` 的逐字节身份链**只在原机器上闭合**
  （实测：修复前全新 clone 的身份核验 FAIL，修复后 24/24 PASS）。
  唯一例外见 `.gitattributes` 内注：`models/model/utils.py` 被单独钉成 CRLF，
  以与 Run7 消融运行时被哈希的字节保持一致（该文件在 `SOURCE_IDENTITY_FILES` 中，
  训练中途规范化其字节会让同批 run 的身份不同源）。下一个 Run 在无历史产物依赖后应统一改 LF。

- **研究诚信（不变）**：文献只用可核验题名/年份/venue/DOI/官方 GitHub，绝不编造引用；
  不编造或篡改任何实验数字；产物只做 `_INVALID_*` / `_SUPERSEDED_*` **改名隔离，绝不删除**；
  每个实验保留源码 SHA 与命令，保证可复现。

### 4. 待决事项（需课题负责人明确）

| 事项 | 现状 | 影响 |
|---|---|---|
| 是否允许**追加辅助损失**（边缘监督 / 深监督 aux head） | **已批准（2026-10-11）**：仅限 GT 边缘图辅助监督，λ=0.2 固定不搜；两种落地形式都跑（`aux_only` 训练期深监督、deploy 不变；`aux_inject` 零初始化注入解码器） | Run7（HFAE）据此执行；辅助分支不得影响推理，由 train↔deploy 折叠 smoke 守门 |
| 论文叙事里「边缘 / 高频」能否作为贡献点 | R8（EdgeRefNet, TGRS 2026）已占据「边缘引导细化 + 边缘监督」 | 需按「STR 内部增强 / 消融」定位，不得抬成独立第三贡献；引用细节在正式写作前必须重新核验 |



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

> 设计文档：`docs/temporary/过去的想法/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md`（2026-10-11 归档）；
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
| 门禁预校验 | `check_run.py` 8 用例双向打通（2 正 + 5 负 + 历史 Run3 日志拒绝）；`--mode object` 从 checkpoint 独立实测 deploy 参数（CTRL **4,880,190** 四库一致、≤5M） |

> **事故与修复（2026-10-10）**：第 1 波跑到 ~6.8k/80k 时，用真实短跑的 TEST 区块演练门禁，
> 发现 `[ACTUAL-OPT-STEPS]` 被误加到 `test_best()` 而非 `test_best_strfusion()`，
> 正式区块缺少设计文档 §6.2/§7-T10 要求的步数标记。按 §6.4 第 1 条该波 **layer-A INVALID**，
> 且两变体必须同版本，故：修 `train.py` → 真实短跑复核（E6/CTRL 区块均出现
> `[ACTUAL-OPT-STEPS] 30`，`check_run.py` 除短跑步数外全 PASS）→ T0–T9 重跑
> `PASS=53/FAIL=0/SKIP=0` + 两个 dry-run PASS → 旧产物**不删除**，移到
> `Run4/_INVALID_pre_gatefix_20261010/` 并标注 `not_used_for_any_conclusion` → 两波同时重启。
> 代价约 35 min GPU 时间。详见 `train_scripts/CASA-TViM/Run4/README.md` §7。

关键实测：epoch-0 前向 `torch.equal`=True（`max_abs=0`）；整模型 train↔deploy
`max_abs=0.000e+00` 且二值 disagreement=0（随机与真实 batch=16）；FET 折叠 FP64 `1.2e-15`；
deploy fvcore **2.7313 G**（`unsupported_ops=20`，与设计预测一致）；第 1 步 `γ.grad` 非零、
γ 开门后 `W_diff` 梯度非零、encoder/TAR/DCR/head 均仍有梯度。

**预注册门槛（不得事后放宽）**：SYSU small(1–255px) pooled Recall ≥ **0.2474** 且 ≥CTRL+5.0pp，
Hit@25 ≥ **0.2396** 且 ≥CTRL+5.0pp，ObjRecall ≥CTRL+1.0pp，ObjPrecision 与 <256px 未匹配预测
不劣于 CTRL；跨库守门 CDD ≥97.18、LEVIR ≥91.07、SYSU ≥83.47、WHU ≥95.07 且不低于同期 CTRL；
四库同时 ≥98.00 / 92.50 / 85.00 / 95.00 才是 `PAPER-TARGET-PASS`。判定走
`analyse/run4_fet_report.py`，失败按设计文档 §6.4 决策树**停止该线**。

---

# Run4 结果（已完成，`STATUS = MECHANISM-FAIL`）

两波 8×80,000 步全部跑完（2026-10-10 22:52 自动收口）；协议审计 8/8 PASS、日志硬门 8/8 PASS、
层 A（P0 有效性）8/8 通过。**结论：FET1 未过预注册机制门，按 §6.4 第 2 条否决 1/4 tap 线。**

## 1. 每 run 正式结果（各 `train_log.txt` 最后一个完整 TEST 区块）

| run | F1 | IoU | Recall | Precision | OA | Kappa | deploy | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E6_FET1** CDD | 97.16 | 94.47 | 97.06 | 97.25 | 99.30 | 96.76 | 4.894 M | 2.7313 G |
| **E6_FET1** LEVIR | 91.11 | 83.66 | 89.70 | 92.55 | 99.11 | 90.64 | 4.894 M | 2.7313 G |
| **E6_FET1** SYSU | 83.34 | 71.43 | 80.69 | 86.16 | 92.39 | 78.41 | 4.894 M | 2.7313 G |
| **E6_FET1** WHU | 94.71 | 89.95 | 92.42 | 97.12 | 99.59 | 94.50 | 4.894 M | 2.7313 G |
| M1_R4CTRL CDD | 97.16 | 94.49 | 97.00 | 97.33 | 99.30 | 96.77 | 4.880 M | 2.6747 G |
| M1_R4CTRL LEVIR | 91.09 | 83.64 | 89.67 | 92.55 | 99.11 | 90.62 | 4.880 M | 2.6747 G |
| M1_R4CTRL SYSU | 83.26 | 71.32 | 82.89 | 83.63 | 92.14 | 78.12 | 4.880 M | 2.6747 G |
| M1_R4CTRL WHU | 94.79 | 90.09 | 93.34 | 96.28 | 99.59 | 94.58 | 4.880 M | 2.6747 G |

全部 8 个 run：`[ACTUAL-OPT-STEPS] 80000`、二值分歧 0（真实 batch）、deploy ≤5 M、TEST 区块完整。
**E6 相对 CTRL 在 F1 上基本中性**：CDD 持平、LEVIR +0.02pp、SYSU +0.08pp、WHU −0.08pp。

**FET 门控值 γ（训练后）**：CDD −1.5e-07、LEVIR −1.1e-05、SYSU **+1.8e-05**、WHU −1.2e-04
—— 零初始化的 γ 仅开到 **1e-5 量级**（对照：`W_diff` 与 γ 均从 0 起步），即该旁路对决策的贡献
极小；这是"效果为何微弱"的直接机制读数，而非猜测。

## 2. 层 B：SYSU 小目标机制（绝对门与同期增量门必须同时满足）

| 指标（SYSU，1–255px，n=364） | CTRL | **E6** | 增量 | 绝对门 | 增量门 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| small pooled Recall | 0.2085 | **0.2472** | **+3.87pp** | ≥0.2474 | +5.0pp | **FAIL**（绝对门差 **0.023pp**） |
| small Hit@25 | 0.1978 | **0.2170** | +1.92pp | ≥0.2396 | +5.0pp | **FAIL** |
| ObjRecall（IoU≥0.10） | 0.7553 | 0.7543 | −0.10pp | — | ≥CTRL+1.0pp | **FAIL** |
| ObjPrecision | 0.5330 | **0.5627** | **+2.97pp** | ≥CTRL | — | **PASS** |
| 未匹配 <256px 预测域 | 2392 | **2103** | **−289** | — | ≤CTRL | **PASS** |
| SYSU F1 | 83.26 | 83.34 | +0.08pp | ≥83.47 | ≥CTRL | **FAIL** |
| SYSU IoU | 71.32 | 71.43 | +0.11pp | 同向 | — | PASS |

**机制总判 FAIL。** 关键读法：方向**符合** H5 预测（小目标召回 +3.87pp、ObjP +2.97pp、
碎片化 <256px 假阳性 −289 ⇒ **不是**"伪变化放大"分支），但**幅度远低于预注册的 +5pp**，
且绝对门只差 **0.023pp** 未过——按 §6.3"绝对门槛与同期增量门槛必须同时满足，不得选择更容易的一个"，
两项均未达标，故判负；Hit@25 与 ObjRecall 亦未达标。

## 3. 层 C：跨库守门与论文硬目标

| 数据集 | E6 F1 | 同期 CTRL | 不退守门 | 守门 | 论文硬目标 | 硬目标 |
|---|---:|---:|---:|---|---:|---|
| CDD | 97.16 | 97.16 | 97.18 | **FAIL** | 98.00 | FAIL |
| LEVIR | 91.11 | 91.09 | 91.07 | PASS | 92.50 | FAIL |
| SYSU | 83.34 | 83.26 | 83.47 | **FAIL** | 85.00 | FAIL |
| WHU | 94.71 | 94.79 | 95.07 | **FAIL** | 95.00 | FAIL |

**必须同时说明的协议事实**：同期 CTRL 本身即低于 Run1 历史值 —— CDD 97.16（历史 97.18）、
SYSU 83.26（83.47）、WHU 94.79（95.06）—— 即 **exact-80K 协议修正本身使 WHU/SYSU 基线下降
0.2–0.3pp**。因此 CDD/SYSU/WHU 的守门未过**不能单独归因于 FET**；但 E6 也未能把这部分补回来
（WHU 还比 CTRL 低 0.08pp）。

## 4. §6.4 决策与后续纪律

> **决策：`MECHANISM-FAIL` → FET 机制否决；停止 1/4 tap 线；不试 1/8 或 3×3；不通过
> 阈值/loss/seed 搜索挤分；不追加模块。**

保留负结果与机制分析。若后续要推进小目标方向，按 §6.4/§9 应回到 **Diag1 D1 的 FP 空间画像**
（只诊断、不加头），且任何新结构都需要**新的预注册**与导师批准（§9 `[I]`）。

**产出物**：`verdict.json`（机器可读三层裁决）、`README_SNIPPET.md`（本节的自动渲染源）、
`objects/*.json`（8 个 run 的部署图对象指标）、`PROTOCOL_AUDIT.json`（两变体协议审计）、
`KNIFE_EDGE_*.json`（折叠刀锋像素定位证据）。

**产物与交接**（`train_scripts/CASA-TViM/Run4/`）

| 文件 | 作用 |
|---|---|
| `README.md` | 设计、执行顺序、预注册门槛、事故记录（§7）、断点恢复（§6）、禁止的捷径 |
| `EVIDENCE_INDEX.md` | ①–⑥ 全部已完成的验证证据与其结论、绝对路径（服务器副本 `$OUT/EVIDENCE_INDEX.md`） |
| `HANDOFF.md` | 会话中断时的接手说明：进程清单、产物落盘路径、手工收口命令、结果读法、不得改动的纪律 |
| `SOURCE_IDENTITY.json` | 代码身份（含与 Diag1 manifest 的逐文件差异分类） |
| `audit_protocol.py` / `check_run.py` | 协议审计（步数预算/epoch 算术/manifest/LR 计划/区块内步数标记）与逐 run 日志硬门 |
| `post_train.sh` | 收口五段：协议审计 → 日志硬门 ×8 → 部署图对象指标 ×8 → 三层裁决 → README markdown（前两段失败即 fail-closed） |
| `auto_post_train.sh` | 无人值守守候：8 个 run 全部完成且无 trainer 运行时自动执行 `post_train.sh` |
| `ckpt_backup_watchdog.py` | `last.pth` → `last.pth.bak` 原子备份（写盘崩溃缓解） |

服务器运行时快照：`$LOG/STATUS.md`（逐 run 步数/速率/ETA）；正式结果落盘为
`$LOG/verdict.json` 与 `$LOG/README_SNIPPET.md`（四张表由 JSON 渲染，无人工转抄）。

---

# Run5 结果（已完成，`STATUS = NO_80K`；正式训练次数 = 0）

> 设计文档：`docs/temporary/过去的想法/CASA-CD_Run5_原生高频证据复用与结构改进实验设计_2026-10-10.md`（2026-10-11 归档）
> （SHA256 `8d721d77…1239bc`）；执行索引与证据：`train_scripts/CASA-TViM/Run5/README.md`；
> 决策锁：`train_scripts/CASA-TViM/Run5/SELECTION_LOCK.json`。
> **只读前测在 val 初筛即判定不合格，`models/` 未被修改，未写训练脚本，未跑任何 80K。**

## 1. 前测做了什么

唯一问题：TinyViM 内部、`cat(x_low,x_high)` 与 `out_proj` **混合之前**的
`local_conv(x_high)`（原生局部/高频路径），对 M1 已漏检的 SYSU small 对象（1–255px）
是否提供**同尺度旧 tap（`L03b_norm2`/`L06b_norm4` 冻结 probe）之外**的增量判别证据。
两个候选插点互斥：`Pair-2` = 1/8 → `TAR.stage2.temporal`；`Pair-3` = 1/16(CAACP) → `TAR.stage3.temporal`。
选层只用 `train.txt` 的 D3 SHA1-hash 10% 固定 val（1212 张），test 不参与选层。

执行链路：S0 源码/资产身份 `--require-all` PASS → 单测 46 项 PASS → P0 素材锁
（M1 `best_F1=0.8347.pth` SHA256、SYSU test list `sha256_text_lines`、两个 probe 的
`concat`/`C_in=192/504`）逐条匹配 → DRY 排练（`--limit 600`）→ 正式前测（val + 统计 26.8 s，
预算 600 s）→ **独立最小复算审计 25/25 OK**。

## 2. 判定证据（val，`n_missed_small=101`、`n_fp_like=387`，有效样本数门通过）

初筛门（预注册）：`gap>=0.05` 且 `novel_rescue>=0.05`。

| 候选 | rescue(HF) | fp_like(HF) | **gap** | novel_rescue | tap_rescue（同尺度旧 probe） | 初筛 |
|---|---:|---:|---:|---:|---:|---|
| `Pair-2`（1/8） | 0.2970 | 0.2636 | **0.0335** | 0.1683 | 0.2772 | ❌ `gap<0.05` |
| `Pair-3`（1/16 CAACP） | 0.0891 | 0.1137 | **−0.0246** | 0.0693 | 0.1089 | ❌ `gap<0.05` |

`eligible = []` ⇒ 无合格候选 ⇒ 按 §5.3.2/§5.3.3 **退出 `NO_80K`**，且**禁止对 test 做任何挑选**
（`test_pass_executed=false`，G1–G8 未评价，阈值保持预注册值未被修改）。

## 3. 机制读数（val，同尺度 256² rank 口径）

| 量 | Pair-2 HF | Pair-2 TAP | Pair-3 HF | Pair-3 TAP |
|---|---:|---:|---:|---:|
| small margin | 0.00697 | 0.00685 | 0.00120 | 0.00086 |
| medium margin | 0.01343 | **0.07808** | 0.01737 | 0.02521 |
| large margin | **−0.01465** | **0.14478** | 0.02313 | 0.13069 |
| small-vs-bg AP | 0.01418 | 0.02056 | 0.00257 | **0.02132** |

- 在 small 尺度上原生 HF 与同尺度冻结 tap **几乎无差别**（Δ=+0.0001 / +0.0003）；
  在 medium/large 尺度上原生 HF 明显更差（1/8 的 large margin 甚至为负）。
- 原生 HF 对**未匹配小预测**的响应与其对**真实漏检对象**的响应几乎相同
  （Pair-2：0.2636 vs 0.2970；Pair-3：0.1137 vs 0.0891）⇒ 它**不选择性编码漏检 small 变化**。
- 与设计文档预注册失败模式 **R1（`x_high` 主要是纹理/配准伪差）与 R2（与 `norm2/norm4` 旧 tap 冗余）**
  一致：`novel_rescue` 非零（Pair-2 17/101）说明不是完全冗余，但判别性（gap）门不过。

## 4. 缺陷记录与纪律

前测脚本首版有一处 P1 缺陷（`missed_small` 漏掉 `bin_index(area) ∈ small` 组过滤，
medium/large 漏检被计入，1250 vs 正确 295），由**独立最小复算**发现——该复算精确重现
Diag1/Run4 冻结值（`n_small=364`、pooled `0.19742166`、`Hit@25=0.18956044`、missed `295`）。
修复后新增回归单测，并在 `gate.json` 增加 `small_reference_reconciliation` 区块与 Diag1 冻结值直接对账。
缺陷期与标签取代期产物以 `_INVALID_*` / `_SUPERSEDED_*` **改名保留，未删除**；
失败口径下的任何数字都未被采用为结论。

## 5. 下一步（按 Run5 设计文档 §8.4 D1 / §10，唯一立即动作）

只读 **SYSU FP 空间画像**：对预测 <256px 未匹配连通域统计 `(i)` 与 GT 边界距离 ≤4px 的比例、
`(ii)` 变化图与 A/B 单时相边缘的重叠、`(iii)` 组件面积/纹理能量、`(iv)` 与错位邻域关系，
按图像级 bootstrap 报 CI；**不增模型、不跑 80K**。若结论指向 1/16→1/32 表征尺度，
再组织导师会审骨干级改动并**重新立项**（需写明不再保证 ImageNet epoch0 对应）。

## 6. §10 止损只读诊断结果（已完成）

`analyse/tvim_run5_fp_profile.py`（`run5_fp_profile_v1`），SYSU test 4000 张、**39.5 s**、只读。
未匹配小预测 **2430 个**（与 Diag1 冻结的 “2430 unmatched <256px” 逐数一致）；
面积匹配背景对照 2397 个。

| §10 子项 | 结果（图像级 bootstrap 1000，seed 16） |
|---|---|
| (i) 与 GT 边界距离 | p25=0、**p50=30.2px**、p95=196.8px；`frac_px_within_4px_of_gt` **严格双峰**：65.5% 为 0（1591 个）、32.1% 为 1.0（781 个） |
| (ii) A/B 边缘重叠（1px 邻域） | FP **0.5545** CI95[0.5384, 0.5722]；对照 **0.4361**；配对差 **+0.1210 CI95[+0.0954, +0.1459]** |
| (iii) 面积/纹理/变化能量 | 面积中位 22；`change_abs` FP **0.9730** vs 对照 0.6063 vs **GT missed 0.8626** vs GT matched 1.2176 |
| (iv) 错位邻域 | **`best_shift_iou` max=0.1556、`n(≥0.5)=0`**：±3px 下无任何 FP 能与 GT 匹配 |

**机制结论**

1. **错位/配准假设被否决**：0/2430 可在 ±3px 内匹配到 GT。
2. FP 是**两个总体**：约 32%（808 个）紧贴 GT 边界的碎片；约 66%（1591 个）完全独立的假变化
   （4px 内无 GT 像素）；两者在边缘重叠、变化能量、面积上几乎相同。
3. **不是“证据不足”，而是“判别不足”**：FP 区域的 `change_abs`（0.973）**高于真实漏检 small GT
   （0.863）**、远高于背景（0.606）——原始变化/高频能量在假阳性处更强。
4. ⇒ **边缘/高频增强或检测模块会优先放大已在过 firing 的信号**，从数据上否决 “edge map 救小目标”；
   与 Run5 前测的 `NO_80K` **互相独立地**一致。
5. 结论指向 **变化-纹理判别 + 边界精度**，**不指向 1/16→1/32 表征尺度丢失**；骨干级改动在本画像下
   同样缺乏依据，需导师会审并重新立项。

产物：`…/diagnostics/TViM-Run5-FP-PROFILE-20261011/`（`fp_profile.json` sha256
`b65a0d1d37ef37cbff9ea680ea4b9f8c4f13c31ce4b712aaff593ccf96f560fd`）；`gate.json`
`status=COMPLETED` 仅表示执行完成且 `not_a_gate=true`，不得用于事后放宽 Run5 §5.4 的 G0–G8。

---

# Run6 结果（已完成，`STATUS = NO_80K`；正式训练次数 = 0）

**Run6 的唯一问题**：冻结的 Run1 `M1_FULL` **既有特征**里，能否把「FP-like 未匹配小预测」与
「真实 small 变化」分开？（即：SYSU 小目标的瓶颈是不是"信息根本不在表征里"）

**只读前测**（`analyse/tvim_run6_discriminability.py`，`run6_discriminability_v1`）：
拟合 = `train.txt` 的 D3 SHA1-hash 90%（10788 图 / 4362 对象）；评估 = `test.txt` 全量 4000 图
（**327 正 + 2430 负**，与 Diag1/Run5 冻结值**逐数一致**）；probe = 标准化线性 logistic
（L2 由按图 5 折 CV 在拟合集选）；test AUC/AP 用**图像级** bootstrap 1000、seed 16。
用时 **69.6 s**（预算 900 s）；独立最小复算审计 **44/44 checks OK**；`models/` **未修改**
（core12 + Run5 冻结模块 13 文件 SHA256 DIFF=0）。

**K 门读数（test 全量；阈值在见结果之前冻结）**

| 门 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| K1 | `n_pos≥300` 且 `n_neg≥2000` | **327 / 2430** | PASS |
| K2 | test `AUC≥0.80` 且 CI 下界 `>0.75` | AUC **0.97238**，CI95[0.95542, 0.98746] | PASS |
| K3 | test `AP≥0.35` 且 CI 下界 `>0.25` | AP **0.92156**，CI95[0.88684, 0.95124] | PASS |
| K4 | winner − **0.5992** ≥ 0.15 | **+0.37318**（手工证据通道上限远被超过） | PASS |
| **K4b** | winner − `baseline_full` ≥ 0.05 | **+0.00801** | **FAIL** |
| K5 | winner − `baseline_amp` ≥ 0.10 | **+0.21306** | PASS（`gate_stat=consistency`） |
| K7 | train-CV AUC 降序选一 | **C2**（`c2_interior_consistency`）：C1 0.96217 / C2 0.96819 | PASS |

**各 probe 组（同对象集合、同框架）**

| probe 组 | test AUC | test AP |
|---|---:|---:|
| `baseline_hand`（F14 手工证据通道） | 0.53998 | 0.12953 |
| `baseline_amp`（纯幅度基线） | 0.75932 | 0.26083 |
| `baseline_head`（**模型自身置信度**：logit/prob 统计） | **0.95646** | 0.85759 |
| `baseline_full`（amp+hand+head，K4b 分母） | **0.96437** | 0.89726 |
| `C1_features`（跨尺度一致性） | 0.94218 | 0.85420 |
| `C2_features`（对象内部/边界一致性） | 0.95986 | 0.88216 |
| `candidate_C1` = full+C1 | 0.97141 | 0.91821 |
| **`candidate_C2` = full+C2（胜出）** | **0.97238** | **0.92156** |

**机制结论（数据驱动，非推测）**

1. **可分辨信息确实存在**：C2 probe 达 AUC 0.9724 / AP 0.9216，远超手工证据通道 0.5400 与
   纯幅度基线 0.7593 ⇒ 小目标瓶颈**不是"信息不在表征里"**（H6a 成立）。
2. **但增量几乎全部来自模型自己的 head 置信度**：`baseline_head` 单独就有 AUC 0.9565，
   完整基线 0.9644；C2 在其上只加 **+0.0080**（预注册要求 ≥+0.05）⇒ **K4b FAIL ⇒ `NO_80K`**。
3. 这与设计文档 §1.3 / §9 预注册的 **R2**（"一致性信息其实已在 head 里，门控无新增"，
   依据 F05 `DOUT` AP 0.9053 ≈ head 0.9049）**方向一致**；叠加 **R3**（Run4 前测 PASS 而 80K 无变化）
   ⇒ 在 4,880,190 参数不变、不改损失/增强/阈值/种子的冻结协议下，**读出层结构增益已被穷尽**。
4. **Run6 训练次数 = 0**，`models/` 未被修改，未写训练脚本，未开 80K。
   按设计文档 **§7.6 收口**：写成"负结果 + 机制归因"一节（含可复现的门槛与前测方法论），
   保留两个已成立贡献（CAACP-SS2D、TAR/DCR 重参数化多尺度融合），
   下一步只能走**骨干级/预训练级重立项**（需导师签字，且不再保证 ImageNet epoch0 对应）。
5. **论文硬目标（98.00/92.50/85.00/95.00）不作为本轮的判据或结论**；Run1 主线仍为
   97.18 / 91.07 / 83.47 / 95.07。

**过程记录（诚实披露）**：首轮全量前测（68.9 s）的产物因实现缺陷被判无效——我在重构 C2 时
把 head 组三列改名，漏掉 `logit_mean` / `logit_std` / `logit_ring` 与 `refine_mag` 共
**4 列从未赋值**，整列 NaN 被中位数填补静默成常数 0（远程 `n_missing_imputed_*` 显示
amp 4362 / head 13086 / full 17448 = 列数×4362，逐项吻合）。修正为显式赋值 +
**fail-closed P0 门**（任何特征列整列缺失即 `[P0_INVALID]`）+ 4 项回归单测，
并给独立审计加了 H10 硬检查；首轮产物以 `_INVALID_20261011_0826` **改名保留（未删除）**，
上表与结论全部取自修正后第二轮。K4b 在修正前后分别为 +0.00943 / +0.00801，**结论方向不变**。

产物：`…/diagnostics/TViM-Run6-DISCRIMINABILITY-20261011/`（`gate.json` sha256
`7ef7ef7d98c780950ff1a0ac4637b534e246b0c0efcb14a67ffbf52cd089877c`，
`status=FAIL`、`decision=NO_80K`）。详见 `train_scripts/CASA-TViM/Run6/README.md`。

## Run7（CASA-TViM HFAE · 编码器前两层频率分解高频 → 边缘检测辅助监督，**执行中**）

> **状态：消融正在服务器执行（12 run × 严格 80,000 `optimizer.step`，2×RTX 5090 三波串行）。**
> 本节是**执行索引**；**结果节**将在 12 个 run 全部跑完后由
> `bash train_scripts/CASA-TViM/Run7/finish_run7.sh` 产出，并按
> `train_scripts/CASA-TViM/Run7/README.md` §8.5b 的骨架补到本文件。

**要回答的问题**：把编码器 stage0（1/4，HF 占 75%）与 stage1（1/8）频率分解的**高频分支**
（`local_conv(x_high)`）接出来做「变化边缘检测」**辅助监督**，让边缘梯度**回流进高频分支**，
能否在受控消融下带来四库一致的 F1 提升——而不是又一次"判别信息已经在 head 里"（Run6 的教训）。

**为什么这是新机制**：Run5 的**无监督** HF tap 已被否证、F14 手工边缘通道已被否证、
Run2/Run4 的**末端**旁路两次 `MECHANISM-FAIL`、Run6 的**读出层**门控 K4b FAIL。
HFAE 的承重部件是**监督**（去掉即退化为已否证的无监督注入），且注入点在
**DCR 的 1/4 输入**（`decoder(feats)` 之前），不是末端。

**受控消融（唯一变量）**：`M1_R7CTRL`（同期同代码对照）/ `E9_HFAE_A`（+边缘监督，train-only，
**deploy 与 CTRL 逐位同值** 4,880,190 + 2.6747 G）/ `E10_HFAE_B`（+边缘监督 + 零初始化注入，
deploy 4,890,366 + 2.7170 G）。`λ=0.2` 预注册固定、**禁止搜索**。

**已完成的验证**（详见 `train_scripts/CASA-TViM/Run7/README.md` §7）：
T0–T11 全 PASS；train↔deploy 折叠等价 disagreement = 0；三形态 deploy 参数门经 `eval.py` 实测；
注入点读码核验；收口链 6 步 **12-run 全量排练 `FINISH_RC=0`**；波次衔接与
`rc=0/2/4/5` 四条退出路径全部端到端实测；预注册文档 ↔ 裁决代码逐条对拍一致。

**关键文件**：预注册
`docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md`（含 §7 实施澄清）·
执行索引与 runbook `train_scripts/CASA-TViM/Run7/README.md` §8 ·
会话交接 `train_scripts/CASA-TViM/Run7/HANDOFF.md` ·
判据唯一规范形式 `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json`。

> **硬目标（98.00/92.50/85.00/95.00）是努力方向而非必达门**（见 §1）；
> 接受标准只看**相对同期 CTRL**（≥3/4 库 F1 ≥ CTRL 且 SYSU F1 > CTRL，且机制自检全过）。
> 未达硬目标**不等于**失败，但**绝不可**因此回调 λ/阈值/seed。

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
