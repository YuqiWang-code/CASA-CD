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

CASA-CD：遥感二值变化检测（变化感知上下文聚合 + 结构重参数化）。两条主线：

| 主线 | 骨干 | 状态 | 约束口径 |
|---|---|---|---|
| **CASA-TViM-STRNet**（Run1–Run7） | TinyViM-S-Slim（ICCV 2025） | Run7 收官，`MECHANISM-PASS`（winner `E9_HFAE_A`） | deploy **4,880,190 ≤ 5M** |
| **CASA-VMamba-STRNet**（Run8，当前） | **VMamba-Tiny**（NeurIPS 2024，`vssm_tiny_0230`） | **进行中**（4 wave × 4 库） | 主干 **29.94M** + 非主干 **2.04–2.34M**（无硬上限，见 §「参数上限变更」） |

硕士课题：遥感二值变化检测。Run1–Run7 已建立并验证「CAACP-SS2D（变化感知上下文聚合）+
STR/TAR-DCR（可折叠结构重参数化）」这条主线；Run8 在**换主干**的同时把它重做成
**双曲流形 + 双分支特化 QKV** 的形式，并引入一条几何监督损失（HGSL）。
早期路线（ChangeViT baseline / CASAA / UltraLight / STR-Fusion / CASA-STR/SHViT）已全部归档——
历史实验记录见 git 历史、`docs/temporary/过去的想法/` 与 `docs/experiment_metrics.xlsx`。
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
| **`deploy ≤5M` 参数上限** | **2026-10-15 变更（Run8 起生效）**：原「有效推理（deploy）参数 ≤5M」**解除**，改为「**除主干（VMamba-Tiny）本身以外，模型总体参数量尽量小**」，不设硬数值上限 | 论文的「超轻量」定位必须改写为**分项报告**（主干 / 非主干分别列）；Run7 及以前的 `≤5M` 结论仍按其当时口径成立、不得混用 |
| **新增损失的审批** | **2026-10-15 批准 1 条**（Run8 的 HGSL，λ=0.2 固定不搜）；**再新增任何损失都必须先经用户审核** | 控制损失条数，避免"堆 loss"式训练 |
| 换主干 | **2026-10-15 用户裁定**：换成 **VMamba-Tiny** | 与 Run1–Run7 的 TinyViM 主线**不可横比**；Run8 必须自带同期 CTRL |




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

## Run4 设计（CASA-TViM R4-FET1 · 1/4 细尺度证据 1×1 可折叠旁路，**已完成**）

> 设计文档：`docs/temporary/过去的想法/CASA-CD_Run4_小目标瓶颈结构改进与实验设计_2026-10-10.md`（2026-10-11 归档）；
> 脚本：`train_scripts/CASA-TViM/Run4/`（`README.md` 含完整执行顺序与预注册门槛）；
> 代码身份：`train_scripts/CASA-TViM/Run4/SOURCE_IDENTITY.json`（与 Diag1 manifest 的逐文件差异）。
> **状态（2026-10-11 更新）：已完成** —— 8×80,000 步全部跑完，结论 **`STATUS = MECHANISM-FAIL`**，
> 详见下方「# Run4 结果」一节。本节保留为当时的**预注册设计**（不预填成绩）。

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

## Run7（CASA-TViM HFAE · 编码器前两层频率分解高频 → 边缘检测辅助监督，**已完成**）

> **状态（2026-10-12 12:36 更新）：已完成** —— 12 run × **严格 80,000** `optimizer.step` 全部跑完，
> 收口链 `FINISH_RC=0`（`gate_rc=0 / check_run_fail=0 / object_rc=0 / verdict_rc=0 / audit_rc=0`），
> 判定 **`STATUS = MECHANISM-PASS`**（`winner = E9_HFAE_A`）。**结果节见下方「# Run7 结果」。**
> 执行索引与 runbook：`train_scripts/CASA-TViM/Run7/README.md` §8；
> 会话交接 `train_scripts/CASA-TViM/Run7/HANDOFF.md`；
> 判据唯一规范形式 `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json`。

**要回答的问题**：把编码器 stage0（1/4，HF 占 75%）与 stage1（1/8）频率分解的**高频分支**
（`local_conv(x_high)`）接出来做「变化边缘检测」**辅助监督**，让边缘梯度**回流进高频分支**，
能否在受控消融下带来四库一致的 F1 提升——而不是又一次"判别信息已经在 head 里"（Run6 的教训）。

**为什么这是新机制**：Run5 的**无监督** HF tap 已被否证、F14 手工边缘通道已被否证、
Run2/Run4 的**末端**旁路两次 `MECHANISM-FAIL`、Run6 的**读出层**门控 K4b FAIL。
HFAE 的承重部件是**监督**（去掉即退化为已否证的无监督注入），且注入点在
**DCR 的 1/4 输入**（`decoder(feats)` 之前），不是末端。

**关键文件**：预注册
`docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md`（含 §7 实施澄清与 §7.7 缺陷披露）。

---

# Run7 结果（已完成，`STATUS = MECHANISM-PASS`；正式训练次数 = 12）

**Run7 的唯一问题**：把编码器 stage0（1/4，HF 占 75%）与 stage1（1/8）频率分解的**高频分支**
（`local_conv(x_high)`）接出来做「变化边缘检测」**辅助监督**，让边缘梯度**回流进高频分支**，
能否在受控消融下带来四库一致的 F1 提升——而不是又一次"判别信息已经在 head 里"（Run6 的教训）。

**受控消融说明**（`finish_run7.sh`；3 变体 × 4 库；各**严格 80,000** `optimizer.step`；
协议 `run7_hf_edge_exact80k_v1`；唯一变量 `--hf_edge/--hf_edge_form`）

| 变体 | `hf_edge`/`form` | 说明 | deploy 参数 | FLOPs |
|---|---|---|---|---|
| `M1_R7CTRL` | 0 / none | 同期同代码对照，先跑 | 4,880,190 | 2.6747 G |
| `E9_HFAE_A` | 1 / aux_only | +边缘辅助监督（train-only） | **4,880,190** | 2.6747 G |
| `E10_HFAE_B` | 1 / aux_inject | +边缘辅助监督 **+零初始化注入 DCR 1/4 输入** | 4,890,366 | 2.7170 G |

`λ=0.2` 预注册固定、**禁止搜索**。波次：波一 CTRL → 波二 E9 → 波三 E10，2×RTX 5090 每卡 2 run、
三波串行（督导自动接续）。**收口链本次实测**：`gate_rc=0 / check_run_fail=0 / object_rc=0 /
verdict_rc=0 / audit_rc=0`；**独立审计 10 checks / 0 FAILED**（A1–A10，含 A5 deploy 参数冻结门、
**A8 12-run 单一代码身份** `dc6d9f6e55a1fabd`、A9 接受规则独立复算）；逐 run 硬门 12/12 全过
（CTRL 32 项 / E9 45 项 / E10 46 项，`n_failed=0`）。`models/` 改动仅**新增** `str_hf_edge.py`
与三处接线（`casa_tvim_str_net.py`/`train.py`/`eval.py`），既有算子未改。

**接受标准与守门（预注册冻结）**：`SELECTION_LOCK.json::acceptance_criteria_frozen_numeric`
（冻结于 2026-10-11T09:35:00+08:00，**见结果之前**）——≥3/4 库 F1 ≥ 同期 CTRL 且 SYSU F1 严格 > CTRL，
且机制自检全过；守门 recall / ObjRecall 降幅 ≤0.02、F1 降幅 ≤0.002（CDD/WHU）。

**① 四库 F1 表**（各 run 最后一个完整 TEST 区块；`_RESULTS/summary.json`）

| dataset | CTRL F1 | E9_HFAE_A | ΔE9 | E10_HFAE_B | ΔE10 |
|---|---:|---:|---:|---:|---:|
| CDD-CD-256 | 0.9717 | 0.9705 | −0.0012 | 0.9702 | −0.0015 |
| LEVIR-CD-256 | 0.9115 | **0.9134** | +0.0019 | 0.9123 | +0.0008 |
| SYSU-CD-256 | 0.8268 | **0.8331** | **+0.0063** | 0.8319 | +0.0051 |
| WHU-CD-256 | 0.9484 | **0.9507** | +0.0023 | **0.9518** | +0.0034 |
| **≥ CTRL 库数 / SYSU 是否提升** | — | — | **3/4 · 是** | — | **3/4 · 是** |

> 两个实验臂**都**满足预注册的 `primary_ok`；`winner` 按 `EXPERIMENTS` 顺序取首个 ⇒ **`E9_HFAE_A`**。

**② 对象级小目标表**（`_OBJECT_METRICS/object_metrics_*.json`；**只作归因，不参与判定**）

| variant | dataset | small pooledR (CTRL→EXP) | Δ | Hit@25 Δ | 未匹配小预测 (CTRL→EXP) | ×CTRL | 中大 pooledR Δ (med/large) |
|---|---|---|---:|---:|---|---:|---|
| E9_HFAE_A | CDD-CD-256 | 0.7344→0.7167 | −0.0176 | −0.0120 | 1576→1591 | 1.01 | −0.0036 / +0.0000 |
| E9_HFAE_A | LEVIR-CD-256 | 0.6878→0.6717 | −0.0161 | −0.0083 | 552→478 | 0.87 | +0.0032 / +0.0076 |
| E9_HFAE_A | SYSU-CD-256 | 0.2267→**0.2411** | **+0.0144** | +0.0055 | 2038→1861 | 0.91 | −0.0126 / +0.0071 |
| E9_HFAE_A | WHU-CD-256 | 0.5771→**0.6617** | **+0.0846** | +0.0380 | 67→67 | 1.00 | +0.0370 / +0.0022 |
| E10_HFAE_B | CDD-CD-256 | 0.7344→0.7259 | −0.0085 | −0.0040 | 1576→1590 | 1.01 | −0.0016 / −0.0000 |
| E10_HFAE_B | LEVIR-CD-256 | 0.6878→0.7107 | +0.0229 | +0.0204 | 552→653 | 1.18 | +0.0070 / +0.0016 |
| E10_HFAE_B | SYSU-CD-256 | 0.2267→0.1916 | −0.0351 | −0.0192 | 2038→2813 | 1.38 | −0.0140 / +0.0151 |
| E10_HFAE_B | WHU-CD-256 | 0.5771→**0.6814** | **+0.1043** | +0.0633 | 67→65 | 0.97 | +0.0374 / +0.0044 |

**③ 机制自检原始读数表**（**只作诊断/引用，不参与判定**）

| variant | dataset | form | deploy params | fold disagree | HF tap grad norm | `[HFAE-LOSS]` epochs | aux first→last | all_ok |
|---|---|---|---:|---:|---:|---:|---|---|
| E9_HFAE_A | CDD | aux_only | 4,880,190 | 0 | 0.0891 | 256 | 0.3089→0.1231 | ✅ |
| E9_HFAE_A | LEVIR | aux_only | 4,880,190 | 0 | 0.0736 | 359 | 0.3209→0.0564 | ✅ |
| E9_HFAE_A | SYSU | aux_only | 4,880,190 | 0 | 0.0681 | 214 | 0.3067→0.1337 | ✅ |
| E9_HFAE_A | WHU | aux_only | 4,880,190 | 0 | 0.0655 | 431 | 0.3311→0.0592 | ✅ |
| E10_HFAE_B | CDD | aux_inject | 4,890,366 | 0 | 0.0895 | 256 | 0.3021→0.1229 | ✅ |
| E10_HFAE_B | LEVIR | aux_inject | 4,890,366 | 0 | 0.0735 | 359 | 0.3138→0.0567 | ✅ |
| E10_HFAE_B | SYSU | aux_inject | 4,890,366 | 0 | 0.0676 | 214 | 0.2999→0.1344 | ✅ |
| E10_HFAE_B | WHU | aux_inject | 4,890,366 | 0 | 0.0656 | 431 | 0.3238→0.0593 | ✅ |

> 每 run 的 `fold_real_max_abs_error` 为 1.8e-06 ~ 9.4e-06，但**二值化 disagreement 全为 0**
> ⇒ 预注册 §7.6 的「刀锋像素」例外本次**未被触发**（`fold_knife_edge_accepted = []`）。

**判定与机制归因**

- `verdict = MECHANISM-PASS`；`winner = E9_HFAE_A`；`reasons = []`；两变体 `guardrail_fails` 均为空。
- 逐变体：E9 / E10 均 `n_datasets_ge_ctrl = 3`、`sysu_improved = true`、`mech_all_ok = true`。
- **归因（逐字摘自 `verdict.json::attribution`）**：E9_HFAE_A 在 ≥3/4 库 F1 ≥ 同期 CTRL 且 SYSU F1
  提升，机制自检全过，守门未触发。SYSU small pooledR 0.2267→0.2411（Δ+0.0144）、Hit@25
  0.2198→0.2253、未匹配小预测 2038→1861（×0.91）；WHU small pooledR 0.5771→0.6617（Δ+0.0846）；
  CDD/LEVIR 的 small pooledR 分别 −0.0176 / −0.0161，但中大对象未受损（−0.0036/+0.0000、+0.0032/+0.0076）。
- **机制方向与预注册 §5 一致**：收益落在**小目标召回/定位**（SYSU、WHU 的 small pooledR 与 Hit@25
  同向上升、未匹配小预测下降），而**不是**靠放大已在过 firing 的信号（这正是 Run5 否证的老路）。

**结论**

1. 在 ≤5,000,000 参数与冻结协议下，把前两层频率分解的**高频分支**接出来做边缘辅助监督，
   **确实**能带来**受控**增益：`E9_HFAE_A` 在 **3/4 库 ≥ 同期 CTRL**，其中 **SYSU（小目标瓶颈库）
   F1 +0.0063**、其**小目标对象级召回 +0.0144**；机制自检全过（边缘梯度**真的**回流到高频分支、
   aux loss 逐 epoch 单调下降、train↔deploy 折叠**逐位一致**、deploy 参数**与 CTRL 同值**）。
   这是在 Run4 `MECHANISM-FAIL`、Run5/Run6 `NO_80K` 之后，本主线**首次 `MECHANISM-PASS`**。
2. **但增益幅度小**（SYSU +0.63 F1 点；CDD 轻微回落 −0.12 点），四库均**未达**论文硬目标
   （97.02 / 91.34 / 83.31 / 95.07 vs 98.00 / 92.50 / 85.00 / 95.00）——按修订后的纪律，
   硬目标是**努力方向而非必达门**，**绝不可**据此回调 λ/阈值/seed。
3. **E10（额外零初始化注入）不优于 E9**（3/4 库达标但 SYSU 增益更小、且 SYSU 小目标对象级召回
   反而下降 0.0351、未匹配小预测 ×1.38）⇒ 按预注册取 **E9 = winner**。这条对照说明：
   承重部件是**监督**本身（把梯度压回高频分支），而往 DCR 里**再注入**一路边缘特征没有额外收益。

**产物索引**：`…/Run7/_RESULTS/summary.json` · `verdict.json`（+ `verdict.md`）·
`INDEPENDENT_AUDIT.json` · `check_run_*.json` × 12 · `…/Run7/_OBJECT_METRICS/object_metrics_*.json` × 12。
指标已并入 `docs/experiment_metrics.xlsx`（**78 → 90 行，0 丢失**）；
代码+结果快照 `docs/temporary/models_and_metrics_CASA-TViM_Run7.txt`（49 个代码文件 + 90 行指标）。

**过程记录（诚实披露，论文须写）**

1. **§7.7 hook 注册非幂等缺陷 ⇒ 整轮 12 runs 重跑。** 波二发现单 epoch 耗时随累计 forward 数
   **线性上升**（CDD 65→282 s，CTRL 恒 56~63 s）：`_ensure_hf_hooks()` 把幂等标记打在
   `RemovableHandle` 上、却到 `mod._forward_hooks` 的 **hook 函数**里查（两者不是同一对象），
   于是每次 forward 重复注册 1 个 hook/模块。用独立实测的每 hook 成本（≈2.0 μs/forward）**无自由
   参数**预测四库斜率 `2cF²`，与实测吻合到 10~15%，并预测 CDD epoch 171 = 247 s（实测 245 s）。
   修复后**数值逐位中性**（T12(c)：加 40 个重复 hook 前后输出与 aux logits `max|Δ| = 0`），
   `hf_edge=0` 的 CTRL 更是走不到该函数。处置：前一轮产物**全部封存不删除**
   （`_SUPERSEDED_Run7_20261011_2324_prehookfix/`），修复后**整轮 12 runs 从零重跑**以保持
   A8「12 run 单一代码身份」；判据/阈值/λ/seed/80K/协议版本**一字未改**。
   详见预注册 §7.7 与 `train_scripts/CASA-TViM/Run7/README.md` §7.16。
2. **波三显存事故与恢复。** `E10_HFAE_B/LEVIR-CD-256` 在 GPU0 上连续 `CUDA OOM` 三次被编排脚本判失败
   （`CDD` 亦 OOM 两次、第三次成功）；从自身 `last.pth` 续训（`_load_resume` 恢复 model/optimizer/
   **actual_steps**/**四项 RNG**，`_write_casa_tvim_manifest` **逐字段严格校验通过** ⇒ 80K 预算与
   seed 16 轨迹连续），并因 GPU0 只剩 ~10 GiB（同用户另一项目 C3L-Z 占 11.4 GiB）把该 run 的**剩余
   epoch 迁移到 GPU1** 执行。**披露**：CUDA 设备号不属于冻结协议字段（`run_manifest.json` 无此字段，
   其严格校验通过即为证），该改动只影响浮点累加顺序，不涉及任何超参/阈值/seed/预算。
3. **预注册 §7.6「刀锋像素」澄清本次未被触发**（12 run 的真实 batch 折叠 disagreement 全为 0），
   故本轮结论不依赖该澄清；它仍需在论文中按 §7.6 披露。

> **硬目标（98.00/92.50/85.00/95.00）是努力方向而非必达门**（见 §1）；
> 接受标准只看**相对同期 CTRL**（≥3/4 库 F1 ≥ CTRL 且 SYSU F1 > CTRL，且机制自检全过）。
> 未达硬目标**不等于**失败，但**绝不可**因此回调 λ/阈值/seed。

---

## Run8（CASA-VMamba-STRNet · **换主干为 VMamba-Tiny** + CAACP-HQKV + HGSL，**进行中**）

> **状态（2026-10-15 更新）：代码定稿并开跑。** 4 wave × 4 库、各**严格 80,000** `optimizer.step`，
> 协议 `run8_vmamba_hqkv_exact80k_v1`；双卡并行、波次按"有空位才接续"自动推进。
> 执行索引 `train_scripts/CASA-TViM/Run8/README.md`；会话交接 `…/Run8/HANDOFF.md`；
> 判据唯一规范形式 `…/Run8/SELECTION_LOCK.json`（**在任何 Run8 训练之前冻结**）。
> **结果表见「# Run8 结果」——四波跑完后补齐。**

**要回答的两个问题**

1. 把主干从 TinyViM-S-Slim 换成 **VMamba-Tiny**，并把它 **4 个 stage 的双时相输出**映射到
   **双曲（Lorentz）流形**，按「局部离散度半径」与「全局偏离半径」分成**两个分支**、各做
   **特化 SAT 式 QKV**（K/V 压到 `k·N`、**Q 保持 `N`**），再注入解码器——
   能否相对**同期 CTRL** 取得受控提升？
2. 上述收益里，有多少来自**双曲曲率**本身、有多少只来自**局部径向结构信息**？
   （必须用**同式欧氏**对照回答；若两者持平，就**不得**主张"负曲率必要"。）

**参数预算（用户 2026-10-15 裁定：解除 `deploy ≤5M`，改为"除主干外尽量小"）**

| 组件 | 参数量 |
|---|---:|
| VMamba-Tiny 主干 | **29,938,176（29.94 M）** |
| CAACP-HQKV（新模块） | **304,131** |
| TAR | 1,351,688 |
| DCR | 686,987 |
| head | 193 |
| **合计** | **31,977,044**（CTRL）/ **32,281,175**（+HQKV） |
| **非主干合计** | **2,038,868（CTRL）/ 2,342,999（+HQKV）** |

**主干口径（⚠️ 必须连同配置一起引用，否则失真）**：`vssm_tiny_0230_ckpt_epoch_262.pth`
（ImageNet-1k 1000 类，**30,705,832** 含分类头 / **29,935,296** 纯骨干）——
`depths=[2,2,5,2]`、`dims=[96,192,384,768]`、`ssm_ratio=2.0`、`ssm_d_state=1`、
**无 z 门控（noz）**、`ssm_conv_bias=False`、patch_embed v2（3→48→96）、downsample v3（k3/s2）、
`norm_layer=ln2d`、`mlp_ratio=4.0`。
> ⚠️ 不得引用 `others/VMamba/vmamba.py::VSSM.__init__` 的**类默认值**（`[2,2,9,2]`/`d_state16`/v1/v2）
> ——那会得到 42.95M 且**根本加载不了这枚权重**（本项目首版正是这么错的，smoke 直接 shape mismatch）。

**关键设计取舍（逐条对应"已知陷阱"）**

1. **不用 `rank_normalize_2d`。** 这是本轮最重要的取舍。`rank(·)` 只保留序、丢弃一切幅度，
   **任何保序变换**（包括固定半径的双曲距离）在其之后都得到**逐位相同**的权重
   ⇒ "把 `1−cos` 换成双曲距离"不构成任何新增信息。该退化定理已在真实张量上**复现**
   （固定半径双曲 vs 余弦：`frac_rank_diff=0.0, Kendall τ=1.000000`）。
   Run8 改为让 Q/K/V **直接消费数值分数**（softmax 对幅度敏感）。
2. **半径非恒定**：`r = r0 + a·q`，`q` 来自同**时相**的局部离散度（detail 分支）或全图偏离
   （global 分支）；`r0=0.5`、`a=0.5` **预注册冻结、禁止搜索**。
   非退化性实测：`detail vs 余弦` `frac_rank_diff=0.9785`；两分支之间 `0.9746`。
3. **压缩必须落在规则二维网格上**（detail 2×2 ⇒ `k=1/4`；global 4×4 ⇒ `k=1/16`）：
   不做 TopK、不做聚类、不做随机采样——SSM 的四向 `CrossScan/CrossMerge` 依赖同一 H×W 网格，
   spatially-agnostic 的 token reduction 会让结构化 Mamba 崩溃（STORM, ICML 2026）。
   这也同时消掉了 CASAA（Run1/Run3，已终止）引入的 `randperm/rand_like` 随机性。
4. **QKV 便宜但保留在部署图里**：`V` 用 per-channel depthwise 1×1（`groups=rank`），
   单分支参数从 `2·dim²` 降到 `3·dim·rank`（`dim=768, rank=32` 时 1.18M → 74K）。
   ⚠️ CAACP-HQKV **是推理期真实算子**（不像 Run7 的 `aux_only` 那样可整头删除）
   ⇒ **不得**声称它零推理代价。FLOPs 实测 **14.6841 G**（input 2×3×256×256，`unsupported_ops=11`）。
5. **新损失 HGSL（λ=0.2 固定）**：监督的是**模块自己的双曲变化分数**（参数自由的量），
   目标用**变化掩码**（`adaptive_max_pool2d` 保底，避免 1–15px 小目标在 1/32 上被平均掉）。
   与 Run7 HFAE 的机制差异：HFAE 监督的是**新增的有参边缘头** + GT 轮廓带；
   而 CAACP 的 score 在 **Run1–Run7 一直走 `no_grad`、从未接收过梯度**。
   `hgsl_detach_score` 是**独立实验臂**（梯度是否进入几何量本身），不得混为一句"加了新损失"。
6. **注入零初始化**（`γ_detail/γ_global` 在 `apply` 之后置零）⇒ `hqkv=1` 与 `hqkv=0`
   在 **epoch-0 逐位一致**（真实数据 batch 实测 `torch.equal=True, max|d|=0.0`）。

**消融设计（每阶段唯一变量 + 每阶段同期 CTRL）**

| 波 | 变体 | `hqkv` | `geometry` | `hgsl` | 注入 | 唯一变量 / 作用 |
|---|---|---|---|---|---|---|
| 1 | **`CTRL_R8`** | 0 | — | 0 | none | — / **同期对照**（VMamba + TAR/DCR） |
| 2 | **`H1_HYP`** | 1 | hyperbolic | 0 | dcr1 | 整个 HQKV 模块 / **主实验** |
| 3 | **`H2_EUC`** | 1 | euclidean | 0 | dcr1 | **仅几何** / **曲率必要性对照**（同式欧氏，用完全相同的 `r_A,r_B,u_A,u_B`） |
| 4 | **`H3_HYP_GS`** | 1 | hyperbolic | 1 | dcr1 | **仅 HGSL** / 新损失的增量 |

**接受标准（预注册冻结，`SELECTION_LOCK.json::acceptance_criteria_frozen_numeric`，冻结于
2026-10-15T02:00:00+08:00，见结果之前）**：`H1_HYP` 在 **≥3/4 库 F1 ≥ 同期 CTRL** 且
**SYSU F1 严格 > CTRL**，且机制自检全过；**曲率门**：`H1` 必须在 **≥3/4 库**优于 `H2`，
否则结论只能写成"局部径向结构信息有用"，**不许**写"负曲率必要"；**非退化门**：两分支在真实特征上的
排序分歧 > 0.05；守门沿用 Run6/Run7 已冻结的「中大对象 pooled Recall 降幅 ≤0.02、
CDD/WHU F1 降幅 ≤0.002」。

**服务器预检（实测，全部通过）**

| 项 | 读数 |
|---|---|
| VMamba 官方权重加载 | `retained=212/222`、**`worst_diff=0.0`**（逐位） |
| CPU 预检（数学/形状/预算/HGSL） | **26/26 PASS** |
| HQKV 非退化性（真实 backbone 特征，4 stage） | `deg_frac = 0.7344 ~ 0.9963`（门 0.05） |
| epoch-0 恒等（真实 batch，四库 × 2 变体） | `torch.equal=True`、`max|d|=0.0` |
| train↔deploy 折叠门 | 翻转像素 1–3 个、margin ≤ 3e-7、`max_abs_diff ~1e-6` ⇒ 全部判为 **`KNIFE_EDGE_ARTIFACT`**（与 Run7 预注册 §7.6 **同一定义**，报告须写明走的是这条判据） |
| batch32 峰值显存 | **26.9–27.7 GB**（32.6 GB 卡）⇒ **每卡只能放 1 个 run** |
| 排卡 | 双卡并行（GPU0 / GPU1 各 1 run），波次自动接续；对**别人的进程只检测、从不抢占** |

**过程记录（诚实披露，论文须写）**

1. **首版按 `VSSM` 类默认值搭主干 ⇒ 42.95M 且加载不了权重**（smoke 报
   `patch_embed.0.weight: ckpt (48,3,3,3) vs model (96,3,4,4)`）。
   已按权重 shape **反推**真实配置（见上「主干口径」），修正后 29.94M、`worst_diff=0.0`。
2. **channel-first 下不能用 `nn.Linear`**：`LayerNorm2d` 的输出是 (B,C,H,W)，
   最后一维是 W 不是 C ⇒ `F.linear` 直接 shape 报错。`in_proj`/`out_proj`/`Mlp`
   三处改用官方 `Linear2d`（权重仍是 2D `(out,in)`，只在 forward 里视图成 conv2d 权重）。
3. **`run_all.sh` 由 `setsid nohup` 拉起时不继承 conda 激活** ⇒ `python: command not found`
   （四库各失败 3 次）。已显式 source+activate+导出环境变量，并新增 `_r8_preflight.sh`
   （真实 train.py + 真实数据跑 3 步）把这类错误压成 30 秒检查。
4. **调度器层面的 5 个真实缺陷**（见 `train_scripts/CASA-TViM/Run8/HANDOFF.md` 与 git 提交信息）：
   `own_pids()` 认不出自己的 GPU 进程（`nvidia-smi` 的 `pid, mem` 未按逗号切分 +
   后台起的 `train.py` 匹配不到）⇒ 自己的 run 被判成 foreign；`pgrep -f '_r8_supervisor\.sh'`
   **子串匹配**到并行督导 ⇒ 误杀；"非空目录"守卫不看文件名 ⇒ 连 `*SUPERSEDED*` 也当非空而拒绝启动；
   断点续训缺口；重复督导实例。**全部已修，且全程未打断任何正在跑的 `train.py`。**

**产物**：`models/model/{vmamba_tiny,casa_vmamba_net,hgsl_loss}.py` ·
`models/model/layers/{vmamba_kernel,hyperbolic,caacp_hqkv}.py` ·
`models/{run8_dry_run,test_run8_cpu_precheck}.py` ·
`train_scripts/CASA-TViM/Run8/`（`run_all.sh`、`_r8_run_one_ds.sh`、`_r8_pipeline_supervisor.sh`、
`SELECTION_LOCK.json`、`SOURCE_FILES.txt`、`SOURCE_SHA256.txt`、`check_mapping.py`、
`README.md`、`HANDOFF.md` 及 `_r8_*.sh` 探针/启动/诊断脚本）。

---

# Run8 结果（**已停止**；判定 = `NO-OP`（无效对照），**不是**机制否证）

> ## ⚠️ 2026-10-17 14:00 结论：Run8 作废、由 **Run8b** 接续
>
> **Run8 的 H1_HYP 与 H2_EUC 在四库全部自锁**：`[HQKV-GAMMA] ≈ ±5e-41 / 0 / 0`、
> `[HQKV-DIAG-*] contrib_norm = 0.0`（四个 stage）、`branch_out_ratio_db = 0.0`
> ⇒ **CAACP-HQKV 从未进入网络**，H1/H2 与 CTRL 的 F1 差（−0.0016~+0.0001）是数值噪声。
>
> **根因三条（全部实现级，不是"双曲没用"）**
> 1. **`γ` 零初始化 + 乘性门 = 自锁**：`r = γ·Δ`，`γ=0 ⇒ r=0 ⇒ ∂L/∂γ = ⟨∂L/∂r, Δ⟩ = 0` 恒成立；
> 2. `gamma_lite` **从未进入前向**（stage0 的 `lite_fuse` 输出被乘的是 `gamma_detail`）；
> 3. `gamma_global` **从未进入前向**（detail/global 先 `o=a+b` 求和，再乘单一 `gamma_detail`）。
>
> ⇒ 本轮四个 run 全部记为 **`NO-OP`**，**只有 H1 与 H2 的"无效"这一条结论可用**：
> "零初始化门 + 无归一化 = 自锁"。**不能**据此说"双曲/双分支几何无效"——
> `H2_EUC` 同样自锁，**曲率门无法被检验**。
>
> **接续**：`Run8b`（`arch=casa_vmamba_str_v2`，协议 `run8b_vmamba_hqkv_antilock_exact80k_v2`）
> 已修复并开跑，见 `train_scripts/CASA-TViM/Run8b/README.md`。

> 以下 Run8 的数字**仅作过程记录**，不进入任何结论表。正式结果只取每库
> `train_log.txt` **最后一个完整** `=== TEST RESULTS === … === END TEST RESULTS ===` 区块。

**① 进度（2026-10-17 14:02 停止时）**

| 变体 | CDD | LEVIR | SYSU | WHU | 状态 |
|---|---|---|---|---|---|
| `CTRL_R8` | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | **4/4 完成**（有效对照） |
| `H1_HYP` | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | **4/4 完成**（NO-OP） |
| `H2_EUC` | ✅ 80,000 | 5,798 / 80,000 | — | — | 停止（CDD 也确认自锁） |
| `H3_HYP_GS` | — | — | — | — | 未启动（无意义，已停） |

**①b 排卡与 ETA（实测口径，**2026-10-15 22:04 起改为单卡串行**）**

- ⚠️ **GPU1 已被同门占用**（`bqwang`，2 × 12,362 MiB `3dinvnet/study.py`，利用率 100%）。
  本项目的单 run 峰值显存 **26.9–27.7 GB**，GPU1 只剩 ~7.8 GB ⇒ **放不下，也不该抢**。
  ⇒ 用户裁定（2026-10-15）：**等 GPU0 上的 WHU 跑完后，Run8 其余全部训练都在 GPU0 上串行完成**。
- 现在的督导是 **`_r8_pipeline_supervisor_single.sh`（v3，单卡串行，`GPU_ID=0`）**；
  双卡版 `_r8_pipeline_supervisor.sh`（v2）保留在仓库里备将来复用。两者对 `python train.py`
  的调用**参数完全相同**，差别只有设备号与并发度（设备号不是 `run_manifest.json` 的冻结字段）。
- **硬纪律**：对 GPU1 上同门的作业**只统计、绝不触碰**（不 kill、不抢占）；
  开跑前用 `nvidia-smi` 实测 GPU0 余量。
- 实测 **step = 0.307–0.310 s @ batch32**（进程口径 3.2303 step/s ＋ 日志口径 3.2632 step/s 互证）
  ⇒ **单库 80K steps ≈ 6.9 h**；**单卡单波（4 库串行）≈ 27.5 h**。
- **四波全部完成 ≈ 2026-10-19 10:00**（自 10-15 22:04 起算约 **84 h**）：
  WHU ≈ 10-15 23:30 → 波二 ≈ 10-17 03:00 → 波三 ≈ 10-18 06:30 → 波四 ≈ 10-19 10:00。

> **代价与理由（如实记录）**：双卡改单卡把总时长从 ~46 h 拉长到 ~84 h。
> 这是"不打扰同门作业"的必要代价。**没有**采用"跨波并行抢 GPU1 空档"这类做法——
> 抢占会与项目前 7 轮的核心纪律（单变量可归因 + 不打扰他人）冲突。

> **⚠️ 2026-10-15 22:07–22:08 的实测教训（这条**支持**单卡决定，务必留档）**：
> 22:07:20 快照显示同门 `bqwang` 的作业**已完全退出**（GPU1 = 15 MiB / 0% / 无 compute 进程）；
> 但 **22:08:13 再测，它又回来了**（2 × 2,092 MiB `FOREIGN(user=bqwang)`，利用率 93%）。
> ⇒ 一次快照里的"卡空着"**根本不能推断"这张卡归我了"**：那是同门两次 run 之间的间隙。
> 若当时据此改回双卡，本项目的 run 会与同门的作业同卡争抢（27.5 GB + 24.7 GB > 32.6 GB）
> ⇒ 双方都 OOM。**因此维持单卡串行**（用户 2026-10-15 的明确指示：其余训练都在卡0 上完成）。
>
> 若将来确需改回双卡（总时长 ~84 h → ~46 h），一条命令即可（可逆）：
> ```bash
> pkill -f _r8_pipeline_supervisor_single.sh
> setsid nohup bash train_scripts/CASA-TViM/Run8/_r8_pipeline_supervisor.sh \
>     > /tmp/run8/pipe.stdout 2>&1 < /dev/null &
> ```
> 采用时必须意识到：同门一旦在跑，本项目 run 会 OOM 并从自身 `last.pth` 续训（有损失但可恢复）。
- 排卡纪律：**对同门/别人的进程只检测、从不抢占**（`own_pids()` 用三重并集把"我们自己的残留"
  与"别人的作业"分开；开跑前用 `nvidia-smi` 实测余量；不 kill 任何非本项目进程）。

**② 四库 F1 表**（各 run 最后一个完整 TEST 区块）

| dataset | CTRL_R8 | H1_HYP（双曲） | ΔH1 | H2_EUC（同式欧氏） | ΔH2 | H3_HYP_GS（+HGSL） | ΔH3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDD-CD-256 | 0.9776 | 0.9777 | **+0.0001** | 训练中 | — | 未启动 | — |
| LEVIR-CD-256 | 0.9121 | 0.9116 | **−0.0005** | 未启动 | — | 未启动 | — |
| SYSU-CD-256 | 0.8188 | 0.8173 | **−0.0015** | 未启动 | — | 未启动 | — |
| WHU-CD-256 | 0.9434 | 0.9418 | **−0.0016** | 未启动 | — | 未启动 | — |
| **≥ CTRL 库数 / SYSU 是否提升** | — | — | **1/4 · 否** | — | — | — | — |
| **曲率门（H1 在 ≥3/4 库 > H2）** | — | — | — | **待 H2 跑完** | — | — | — |

四库完整指标（同一 TEST 区块）：

| run | dataset | Recall | Precision | OA | F1 | IoU | Kappa | deploy params | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CTRL_R8 | CDD | 0.9779 | 0.9773 | 0.9945 | 0.9776 | 0.9562 | 0.9745 | 31.026 M | 13.6715 G |
| CTRL_R8 | LEVIR | 0.9043 | 0.9201 | 0.9911 | 0.9121 | 0.8384 | 0.9074 | 31.026 M | 13.6715 G |
| CTRL_R8 | SYSU | 0.7966 | 0.8422 | 0.9169 | 0.8188 | 0.6932 | 0.7649 | 31.026 M | 13.6715 G |
| CTRL_R8 | WHU | 0.9355 | 0.9514 | 0.9955 | 0.9434 | 0.8928 | 0.9411 | 31.026 M | 13.6715 G |
| H1_HYP | CDD | 0.9778 | 0.9776 | 0.9945 | 0.9777 | 0.9563 | 0.9745 | 31.330 M | 14.1244 G |
| H1_HYP | LEVIR | 0.9078 | 0.9154 | 0.9910 | 0.9116 | 0.8375 | 0.9068 | 31.330 M | 14.1244 G |
| H1_HYP | SYSU | 0.7925 | 0.8436 | 0.9164 | 0.8173 | 0.6910 | 0.7632 | 31.330 M | 14.1244 G |
| H1_HYP | WHU | 0.9271 | 0.9571 | 0.9955 | 0.9418 | 0.8901 | 0.9395 | 31.330 M | 14.1244 G |

**②b ⚠️ 决定性机制读数：CAACP-HQKV 在整个训练过程中**从未**产生非零贡献**

| 读数 | CTRL_R8 | H1_HYP（四库全部一致） | 含义 |
|---|---|---|---|
| `[HQKV-GAMMA] detail` | — | **5.2e-41 / −5.2e-41 / −5.9e-41 / −5.9e-41** | 零初始化后**再未离开 0**（float32 的次正规下界量级） |
| `[HQKV-GAMMA] global` | — | **0.000000e+00** | 恒为 0 |
| `[HQKV-GAMMA] lite` | — | **0.000000e+00** | 恒为 0 |
| `[HQKV-DIAG-*] contrib_norm` | — | **0.000000e+00（四个 stage 全部）** | 模块输出恒为零张量 |
| `[HQKV-DIAG-*] branch_out_ratio_db` | — | **0.000000e+00** | 两分支输出都是 0 ⇒ 比值无定义 |
| Δdeploy params | — | +304,131（= HQKV） | **纯增参数、零有效贡献** |
| ΔFLOPs | — | +3.31% | **纯增算力、零有效贡献** |

⇒ **H1_HYP 的 logits 与 CTRL_R8 事实上是同一个模型**；四库 F1 的 −0.0016 ~ +0.0001 是
**数值噪声**（`[REPARAM-REAL-ARGMAX-DISAGREE]` 也在 0 / 9.5e-7 之间），**不能**当作"两分支几何无效"的证据。

**为什么门没学起来（[I] 级推断，不是事实）**：`γ` 零初始化 ⇒ epoch-0 时 `dr/dγ = ‖Δ‖` 很小；
`γ` 一旦停在 ~1e-41，模块输出恒为 0，**梯度只能经由 `γ` 自身回流**（`∂L/∂γ = ‖Δ‖ = 0`）
⇒ 形成**自锁（self-locking）**：出不了零。
这是**实现级设计缺陷**，不是"双曲/双分支无效"的科学结论。

**因此本轮已产出的四库对比只能记为 `NO-OP`（无效对照）**，
而不是 `MECHANISM-FAIL`（机制被否证）——两者性质完全不同，必须分清。

**②c 机制门核对（预注册冻结项）**

| 门 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| `[PRETRAIN-LOAD] worst_diff` | == 0 | **0.000e+00**（8 个 run 全部） | ✅ PASS |
| `fold_disagreement`（真实 batch） | == 0 | 0 或 9.5e-7（刀锋像素，margin < 1e-6） | ✅ PASS（按 Run7 §7.6 判据） |
| `[ACTUAL-OPT-STEPS]` | == 80000 | **80000**（8 个已完成 run 全部） | ✅ PASS |
| `[HQKV] gamma_zero_init` | true | true | ✅ PASS（但见 ②b） |
| **`γ` 在训练中离开 0** | 应≫0 | **≈1e-41 / 0** | ❌ **FAIL（自锁）** |
| **非退化门** `deg_d_vs_g_frac_rank_diff > 0.05` | > 0.05 | stage0–2: **0.95–0.997**；stage3: **0.73–0.84** | ✅ PASS |

> 注意：**非退化门是通过的**（两分支的分数排序确实差异很大），
> 但因为 `γ≈0`，这个差异**从未进入网络输出**。这也说明"分数层面非退化"不等于"机制生效"。

**③ 归因与结论（截至 2026-10-17 13:10，**不是最终结论**；H2/H3 未跑完）**

1. **`H1_HYP` 相对同期 `CTRL_R8`：1/4 库 ≥ CTRL、SYSU 未提升（−0.0015）**
   ⇒ 按预注册接受标准（≥3/4 且 SYSU 严格提升）**未通过**。
2. **但该"未通过"不能解读为"双曲双分支机制无效"**，因为 `[HQKV-GAMMA] ≈ 1e-41 / 0` 与
   `contrib_norm = 0` 证明**模块输出恒为零、从未进入网络**。这是**实现级自锁**：
   `γ` 零初始化 ⇒ 模块输出 0 ⇒ `∂L/∂γ = ‖Δ‖ = 0` ⇒ `γ` 永远出不了 0。
   **本轮 H1 实验因此是 `NO-OP`（无效对照），必须按无效对照记录，不得当作机制否证。**
3. **代价可以精确量化**：`H1_HYP` 比 `CTRL_R8` 多 **304,131 参数**（+0.98%）、多 **3.31% FLOPs**，
   换来的有效贡献是 **0**。这条读数本身就是有价值的负结果（"零初始化门 + 无归一化的模块输出
   会自锁"），且与 Run7 的 `aux_inject`（E10 不优于 E9）方向一致：**"再注入"很难有增益**。
4. **非退化门通过**（两分支排序分歧 0.73–0.997 ≫ 0.05）⇒ **几何量本身是有区分度的**，
   失效点在"如何把它送进网络"，不在"几何有没有信息"。这为下一轮指明了唯一的修复点。
5. **曲率门待定**：`H2_EUC` 刚跑完 CDD（76,998/80,000），四库跑完才可判"曲率是否必要"。
   若 `H2 ≈ CTRL`（与 `H1` 同样自锁），则曲率门**无法被检验**，必须在论文中如实说明。

**⚠️ 下一轮必须先修的三个实现问题（不是新方法，是让机制真的能生效）**

1. **去掉 `γ` 的裸零初始化**：改为 `γ` 在 **LayerNorm / 单位方差归一化之后**的残差上（使 `‖Δ‖≈1`），
   或给 `γ` 单独的 **大学习率参数组**（如 ×100），或直接用 sigmoid/softplus 门并初始化为 0.5。
   **判据**：训练 1000 步后 `|γ|` 必须 > 1e-3，否则立即停止该 run（fail-fast）。
2. **`γ_global` / `γ_lite` 恒为 0 的结构性原因**：`lite` 分支根本没有被乘进输出（`stage0` 走
   `lite_fuse` 但输出乘的是 `gamma_detail`）；`global` 门的输出没有独立路径。
   ⇒ 需要**每个分支一套独立门 + 独立残差**，否则这两个门永远收不到梯度。
3. **加一条进程内的 `contrib_norm > 0` 硬断言**（epoch 0 之后），让"模块零贡献"在训练早期
   就 fail-fast，而不是跑满 80K 后才发现。

**仍需补充的证据**

- [x] ~~修复 `γ` 自锁后的重跑~~ ⇒ **已由 `Run8b` 执行**（见下一节）；`B1_HYP/CDD` 已实测封口抗自锁
- [ ] Run8b 的四库结果（**5/16 已完成 + 1 在跑**，ETA ≈ 10-22 下午）
- [ ] 收口链（`finish_run8.sh` / `check_run.py` / `run8_verdict.py` / `independent_audit.py`）尚未移植
- [ ] `models/eval.py` 补 `--arch casa_vmamba_str_v2`（正式结果不依赖，独立审计前必须补）
- [ ] 对象级小目标复算（`analyse/tvim_object_metrics.py` 口径：4 连通、native 256² mask、按 GT 面积分组）

---

# Run8b 结果（**已停训 2026-10-20 11:49**；8/16 封口；判定 = 双曲 HQKV **无正增益**）

> **2026-10-20 后续：双曲几何已按用户指令拆除**，见下一节 `# Run8c`。
> 拆除前的实现字节归档在
> `train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/`（6 个文件，SHA256 与
> `SOURCE_SHA256.txt` 逐项相等）。
>
> **停训决策依据**：`B1_HYP` 四库跑完后 ΔF1 = {+0.0001, −0.0009, −0.0012, −0.0014}、均值 **−0.00085**，
> 且多付 +0.304 M 参数 / +0.46 G FLOPs / 约 +9% 时间 ⇒ 「加了双曲几何的 HQKV 不比不加更好」。
> 冻结记录（含未跑项、可复现命令、仍有效资产清单）：`train_scripts/CASA-TViM/Run8b/STOPPED_2026-10-20.md`

> 执行索引与预注册：`train_scripts/CASA-TViM/Run8b/README.md` · `…/Run8b/SELECTION_LOCK.json`
> 协议 `run8b_vmamba_hqkv_antilock_exact80k_v2` · `arch=casa_vmamba_str_v2` · `γ` 初值 **0.5**

**① 进度（停训时刻 2026-10-20 11:49 实测）**

| 变体 | CDD | LEVIR | SYSU | WHU | 状态 |
|---|---|---|---|---|---|
| `CTRL_R8B` | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | **4/4 封口** |
| `B1_HYP` | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | ✅ 80,000 | **4/4 封口** |
| `B2_EUC` | 4,069 / 80,000（5.1%，已 KILL） | — | — | — | **不算结果** |
| `B3_HYP_GS` | — | — | — | — | 从未启动 |

> **完成度 8/16 封口；全程 0 次 OOM、0 次 resume（`[RESUME]` 计数 = 0）。**
> `B2_EUC` 的 5.1% 没有 TEST 区块 ⇒ **不得引用**；因此「双曲 vs 欧氏」这一步**没有测完**（见 ② 末）。
> 产物一份未删；GPU0 已让出（166 MiB / 0%）。

**② 已产出的正式结果（各 `train_log.txt` 最后一个完整 TEST 区块）**

> 来源口径：Run8b 的正式结果由 **`train.py` 自带的最终 TEST 阶段**产出（`=== TEST RESULTS ===` 区块），
> **不经过 `models/eval.py`**。结构真值以 ckpt 目录下的 `arch.json` / `run_manifest.json` 为准（已核对，见 ⑦）。

`CTRL_R8B`（`hqkv=0`；**31.977 M / 14.6841 G**）：

| dataset | Recall | Precision | OA | F1 | IoU | Kappa | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDD-CD-256 | 0.9776 | 0.9775 | 0.9945 | **0.9776** | 0.9561 | 0.9744 | 255 |
| LEVIR-CD-256 | 0.8997 | 0.9248 | 0.9912 | **0.9121** | 0.8384 | 0.9074 | 314 |
| SYSU-CD-256 | 0.8040 | 0.8323 | 0.9156 | **0.8179** | 0.6919 | 0.7630 | 58 |
| WHU-CD-256 | 0.9319 | 0.9496 | 0.9953 | **0.9407** | 0.8881 | 0.9383 | 286 |

`B1_HYP`（`hqkv_geometry=hyperbolic`、`inject=dcr1`、`rank=32`、`γ_init=0.5`；**32.281 M / 15.1464 G**）：

| dataset | Recall | Precision | OA | F1 | IoU | Kappa | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDD-CD-256 | 0.9779 | 0.9775 | 0.9945 | **0.9777** | 0.9564 | 0.9746 | 255 |
| LEVIR-CD-256 | 0.9006 | 0.9220 | 0.9911 | **0.9112** | 0.8369 | 0.9065 | 352 |
| SYSU-CD-256 | 0.8004 | 0.8336 | 0.9153 | **0.8167** | 0.6902 | 0.7616 | 24 |
| WHU-CD-256 | 0.9209 | 0.9586 | 0.9953 | **0.9393** | 0.8856 | 0.9369 | 263 |

**`B1_HYP` − `CTRL_R8B`（同期同协议同 seed，唯一变量 = CAACP-HQKV 开启且几何 = 双曲）**

| dataset | CTRL F1 | B1_HYP F1 | Δ F1 | Δ IoU | Δ Recall | Δ Precision |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 0.9776 | 0.9777 | **+0.0001** | +0.0003 | +0.0003 | +0.0000 |
| LEVIR | 0.9121 | 0.9112 | **−0.0009** | −0.0015 | +0.0009 | −0.0028 |
| SYSU | 0.8179 | 0.8167 | **−0.0012** | −0.0017 | −0.0036 | +0.0013 |
| WHU | 0.9407 | 0.9393 | **−0.0014** | −0.0025 | −0.0110 | +0.0090 |
| **均值** | — | — | **−0.00085** | −0.00135 | −0.00335 | +0.001875 |

> **判定：`B1_HYP` 在四个库上 1 平 3 负，均值 −0.00085 F1，同时多付 +0.304 M 参数 /
> +0.46 G FLOPs / 约 +9% 训练时间 ⇒ 双曲几何的 HQKV「不划算」。**
> 其中 WHU 是"召回换精度"（ΔR −0.0110 / ΔP +0.0090），净 F1 仍为负；
> SYSU 上 `best epoch = 24`（对照 58）说明该臂很早就过拟合。
>
> **必须守住的结论边界（不要外推）**：以上只支持「**双曲版** HQKV 不划算」。
> `B2_EUC`（欧氏反事实）只跑到 5.1% 就被停训，**无 TEST 区块** ⇒
> 「欧氏版 HQKV 是否也不划算」「整个跨层 QKV 注入是否无效」**都没有答案**，
> 禁止用这两个库的结果去支持这两句话。

> **轻量性读数（直接由 TEST 区块读出）**：`B1_HYP` 比 `CTRL_R8B` **只多 0.304 M 参数、+0.462 G FLOPs**
> （31.977 M → 32.281 M；14.6841 G → 15.1464 G）。其中 VMamba-Tiny 主干本身 29.94 M，
> 即 STR 头 + CAACP-HQKV 合计仅 ≈ 2.34 M ⇒ 符合"主干之外总体参数尽量小"的裁决。
> 注：TEST 区块头行的 `[MODEL] ChangeViT-TINY baseline` / `[VIT-DEPTH] 12` 是**模板遗留标签**，
> 不代表真实结构（真实结构 = `casa_vmamba_str_v2`，见 `arch.json`），此处如实披露、不做美化。

**③ 关键的"唯一变量"验证：`CTRL_R8B`（v2，hqkv=0）vs `CTRL_R8`（v1，hqkv=0）**

| dataset | Run8 `CTRL_R8` | Run8b `CTRL_R8B` | Δ |
|---|---:|---:|---:|
| CDD | 0.9776 | 0.9776 | **+0.0000** |
| LEVIR | 0.9121 | 0.9121 | **+0.0000** |
| SYSU | 0.8188 | 0.8179 | −0.0009 |
| WHU | 0.9434 | 0.9407 | −0.0027 |

> **这条对照很重要**：`hqkv=0` 时 v2 与 v1 **逐库几乎逐位一致**（CDD/LEVIR 完全相同；SYSU −0.0009、
> WHU −0.0027）。它支持「抗自锁改动在关掉模块时近似零影响」这一判断 ⇒ Run8b 与 Run8 的差异
> 主要落在模块内部，基本满足"唯一变量"要求；同时这也是一次**跨协议的可复现性交叉验证**。
>
> **诚实标注（不夸大）**：这**不是** bit-exact。WHU 的 −0.0027 是本轮"零影响"主张里**唯一一处非平凡偏差**，
> 成因尚未单独归因（候选：v2 多出的参数注册/初始化顺序改变了 RNG 消耗序列）。
> 收口时必须以「`CTRL_R8B` 自身 16/16 同协议对照 + `B1_HYP` 的同期 Δ」为主判据，
> **不能**把 −0.0027 当成"噪声"直接抹掉，也不能据此宣布任何机制结论。

**④ 抗自锁门（已由真实 80K run 实测封口，不再是"期望"）**

来源：`/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b/B1_HYP/CDD-CD-256/train_log.txt`

| 门 / 读数 | 实测 | 判读 |
|---|---|---|
| step-1 `[HQKV-GRAD]` `gamma_grad` | detail **0.014124** / global **0.027942** / lite **0.010388** | 三支**全非零** ⇒ v1 的自锁**已解除** |
| `[FAILFAST-PASS]` | step=1252 `contrib_norm=2.209e3`（阈值 1e-3） | 贡献非零，未触发 fail-fast |
| 255 个 epoch 的 `contrib_norm` | 1.518e3 (e1) → 1.874e3 (e41 峰) → **1.601e3** (e255)，全程 > 0 | v1 是**恒 0**；v2 从未归零 |
| `gamma_detail` | 0.5086 → 0.5603 (e41) → **0.5292** | 风险 R4「反向跑大」**未触发**（峰值 0.56 < 1） |
| `gamma_global` | 0.5131 → **0.5294**（与 detail 近同轨迹） | 独立参数、同步收敛 |
| `gamma_lite` | 0.5176 → **0.2211**（单调下行，−57%） | **首次可观测量**：stage-0 `lite` 支被优化器主动压低 |
| `shares` | 三支各 ≈ **1/3**，256 个 epoch 恒定 | 与固定结构缩放 `1/√(n_branch·n_stage)` 一致（结构性常数，非学到的） |

> **两个必须带进论文/答辩的诚实结论**：
> 1. `shares` 恒定 1/3 是**设计使然**（固定结构缩放），不构成"学到了平衡"的证据；
>    真正学到东西的读数是 **`gamma_*` 的漂移方向**，尤其 `gamma_lite → 0.221`。
> 2. `gamma_lite` 单调下行说明 stage-0 注入支在 CDD 上被**主动抑制**但仍非零（0.22 ≫ 0）。
>    若另三个库也呈同向单调下行，则需在收口时回答"该支是否等价于被关掉"——
>    这条已登记为 Run8b 的开放问题，不在本轮擅自改设计。

**⑤ 开跑前的机制证据（已实测，非推断）**

| 项 | 读数 |
|---|---|
| CPU 预检 | **26/26 PASS**；注入 RMS **0.5054**（期望 0.5） |
| **负对照 γ=0** | `contrib_norm = 0` **且三支 `gamma_grad` 全为 0** ⇒ 精确复现 v1 自锁 |
| 主实验 γ=0.5 | 三支 share 各 **1/3**；`gamma_grad = {0.3333, 0.3375, 0.3261}` |
| 服务器 smoke（真实特征） | `worst_diff=0.0`；γ=0.5 ⇒ `contrib_norm=442.7`、梯度全 >0；γ=0 ⇒ 全 0 |
| dry-run（四库 × 2 变体） | **8/8 全绿**；`B1_HYP` `contrib_norm≈2.6e3`、**RMS≈0.489–0.497**；`CTRL` 正确为 0 |
| `check_mapping.py` / `sha256sum -c` | **PASS** / **22/22 OK**（开跑时快照；08:10 改督导后该行已过期，见 ⑦） |

**⑥ 实测耗时与 ETA（2026-10-20 09:32 重算，全部为 born→mtime 实测）**

| run | 起 | 止 | 耗时 |
|---|---|---|---:|
| `CTRL_R8B/CDD` | 10-17 14:10 | 10-17 21:24 | 7.23 h |
| `CTRL_R8B/LEVIR` | 10-17 21:24 | 10-18 04:38 | 7.24 h |
| `CTRL_R8B/SYSU` | 10-18 04:39 | 10-18 11:56 | 7.29 h |
| `CTRL_R8B/WHU` | 10-18 11:56 | 10-18 18:49 | 6.88 h |
| `B1_HYP/CDD` | 10-18 18:50 | 10-19 04:56 | 10.11 h |
| `B1_HYP/LEVIR` | 10-19 08:11 | 10-19 18:15 | 10.06 h |
| `B1_HYP/SYSU` | 10-19 18:15 | 10-20 03:50 | **9.58 h** |
| `B1_HYP/WHU` | 10-20 03:50 | 在跑（75%，**2.934 step/s** ⇒ 约 11:25 完） | ≈ 7.56 h |

**关键读数**：`hqkv=0` 时 **3.23 step/s**（7.2 h/库）；`hqkv=1` 时近期 **2.93 step/s**（7.6 h/库），
早前三库因同门抢卡掉到 **2.2–2.3 step/s**（10.1 h/库）。⇒ **HQKV 自身的吞吐代价约 9%**，
其余是邻居干扰。

**（原 ETA 表已作废——2026-10-20 11:49 停训，`B2_EUC`/`B3_HYP_GS` 不再跑。）**
若将来续跑，参考值：`hqkv=1` 单库 7.6–10.1 h，剩余 7 库（`B2_EUC` 3 + `B3_HYP_GS` 4，
`B2_EUC/CDD` 的 5.1% 需整库重跑）约 **53–71 h**。

**⑦ 排卡（硬约束，实测依据；2026-10-19 有变更，如实披露）**

- 只用 **GPU0**（单卡串行）；**GPU1 上是另一会话的探索线**（pid 507695，`train_stgeo.py`；
  08:11 实测 27.6 GB、08:14 掉到 3.3 GB —— **占用会自己变**），督导**只统计、绝不触碰**；
- 单 run 峰值显存 26.9–28.4 GB（32.6 GB 卡）⇒ 每卡只能放 1 个 run；
- 「GPU1 空了」不可信：2026-10-15 22:07 实测同门退出、**22:08 就回来了**（故不据此改回双卡）；
- **2026-10-19 08:04 事件**：GPU0 出现同门 `bqwang` 的 1,938 MiB 作业（pid 520316），
  旧判据要求「卡上无任何别人的进程」，导致督导**空转约 6 h 未启动 `B1_HYP/LEVIR`**。
  已把判据改为「**只看真实余量**（`NEED_MB`，默认由 30500 → 30000）」，允许与同门**共存但不触碰**；
  每拍决策写入 `/tmp/run8b/pipeline.log`（`GPU0-READY(shared)` + `foreign=[...]`）。
  实测 `free=30,495 MB`、`B1_HYP` 峰值 28,371 MB ⇒ 余量 **2,124 MB**；
  重启脚本：`bash train_scripts/CASA-TViM/Run8b/_r8b_restart_supervisor.sh 30000`。
  **残余风险（2026-10-19 08:13 实测，已升级为 watch item）**：`B1_HYP/LEVIR` 起跑后
  GPU0 上 **我们 29,022 MiB + 同门 1,938 MiB = 30,960 MiB**，`free` 只剩 **962 MiB**。
  危险窗口有两个：① `epoch=1` 的首次 `val()`（新分配模式，约起跑后 6 min）；
  ② 收尾的正式 `TEST` 阶段（约 10 h 后）。若 OOM，督导会**自动 resume**（见下条），
  但该库必须按 `RESUME-DEVIATION` 标注。
  **后续（10-20 09:32 复查）**：同门进程已退出（`GPU0-READY free=32441MB foreign=none`），
  两个危险窗口都已安全通过，7 封口 run **零 OOM / 零 resume**。风险状态：**已解除，继续监控**。
- **崩溃恢复的协议偏离（必须披露的真实缺陷）**：`models/train.py::_save_last()` 只对
  `arch == casa_tvim_str` 写入 `actual_steps` 与 `rng_state`（`train.py:1142`），
  **`casa_vmamba_str_v2` 的 `last.pth` 不含这两项** ⇒ 一旦 resume：
  `cur_iter` 只能回退到 `start_epoch × iters_per_epoch`（epoch 粒度），
  且 **RNG 流不连续** ⇒ 该库不再是"同一 seed 的连续 exact-80K"。
  留痕机制已存在（督导会把旧日志改名为 `…_before_resume`，且日志里有 `[RESUME]` 行），
  因此**任何出现 `_before_resume` 的库都必须标 `RESUME-DEVIATION`**；
  时间允许时应用干净重跑替换，而不是默默接受。
- **由此产生的披露**：`_r8b_pipeline_supervisor_single.sh` 于 08:10 被改动，
  与 `Run8b/SOURCE_SHA256.txt` 登记的旧哈希 `e8495cda…` **已不一致**。
  该改动**只涉及调度判据**，不触及模型前向 / 损失 / 优化器 / RNG 序列。
- **可复现性硬证据（未被上述改动污染）**：`CTRL_R8B` 与 `B1_HYP` 的 `arch.json` 中
  `protocol_version = run8b_vmamba_hqkv_antilock_exact80k_v2`，且 `source_code_sha256` 的 17 项与
  `Run8b/SOURCE_SHA256.txt` **逐项相符**（含 `train.py = 536b149d…`、`eval.py = 2886b278…`、
  `caacp_hqkv_v2.py = b6f30bec…`、`casa_vmamba_net_v2.py = c6b3de16…`）；
  `B1_HYP/CDD` 的 `γ_init=0.5 / r0=0.5 / a=0.5 / rank=32 / detail_window=2 / global_window=4 /
  attn_stages=1,2,3 / inject=dcr1 / str_dim=192` 与预注册锁定值**完全一致**。

**⑧ 停训后的未完成项（不得宣称已完成）**

- `B2_EUC` 只跑了 CDD 的 5.1%（已 KILL，无 TEST 区块，**不得引用**）、`B3_HYP_GS` 0/4
  ⇒ **「双曲 vs 欧氏」的反事实缺位**，这是本轮结论的边界；
- 收口链（`finish_run8.sh` / `check_run.py` / `run8_verdict.py` / `independent_audit.py` /
  `collect_results.py`）**未移植到 Run8b**（停训后不再需要，保留记录备查）；
- `models/eval.py` **未支持** `--arch casa_vmamba_str_v2`（正式结果由 `train.py` 自带 TEST 产出，
  不依赖它；若要做独立复算则必须补，并同步登记 `SOURCE_SHA256.txt` 的新旧哈希）；
- `docs/experiment_metrics.xlsx`（当前 98 行，含探索线 Exp1 的 8 行）**尚未写入 Run8b 的 8 行**；
  快照（`analyse/models_to_txt.py`）同理 —— 是否入库待拆除方案确定后再定；
- 对象级小目标复算（`analyse/tvim_object_metrics.py` 口径：4 连通 / native 256² mask / 按 GT 面积分组）
  未做。

---

# Run8c：拆除双曲几何（2026-10-20，**代码已改完，未训练**）

> 用户指令：停训（"现在的设计没什么用"）→ **拆除双曲几何结构**，范围 **B** =
> 拆「流形映射 + 半径分解」，**保留跨层 QKV 注入**；拆完只做清理 + 归档，**先不训练**。
> 完整记录：`train_scripts/CASA-TViM/Run8b/TEARDOWN_R8C_2026-10-20.md`

**① 拆了什么**（`CAACP-HQKV` 的整条几何链）

| 删除 | 保留 |
|---|---|
| Lorentz 双曲映射、变化分数 `r = r0 + a·q`、半径 `r0`/`a` | 跨层 QKV 注入（Q 全 `N`，K/V 压到 `k·N`） |
| detail（局部离散度）/ global（全局半径）**双分支**及其双窗口 | SAT 式非对称注意力（Q 长 `N`、K/V 长 `n`） |
| 分数驱动的加权池化 | **固定格点**压缩（`window=2`，无参数、无 TopK/聚类/随机） |
| HGSL 的分数输入口 | 反自锁资产：每支独立 γ + `rms_normalize` + 非零初值 + `contrib_norm` fail-fast |

- 删除文件：`models/model/layers/caacp_hqkv_v2.py`、`models/test_run8b_antilock_precheck.py`、
  `models/run8b_dry_run.py`（**均已归档**，不丢）；
- 新增：`models/model/layers/caacp_hqkv_lattice.py`、`models/test_hqkv_lattice_cpu.py`；
- 改写：`models/model/casa_vmamba_net_v2.py`（传 `hyperbolic`/`euclidean` ⇒ **`SystemExit`**；
  `hqkv=1 且 hgsl=1` 也 ⇒ `SystemExit`，不静默跳过）；
- `models/train.py`：协议串 bump 为 **`run8c_vmamba_lattice_qkv_exact80k_v1`**、
  新增 `--hqkv_window`、v2 的侧车字段改为 `hqkv_form=hqkv_v3_lattice` +
  三个恒为 `"none"` 的**墓碑字段**（`hqkv_manifold` / `hqkv_radius_decomposition` /
  `hqkv_score_weighted_pooling`）。

**② 参数量变化**

| 版本 | 模块参数 | 说明 |
|---|---:|---|
| v2（双曲，三支） | ≈ 0.62 M | 拆除前 |
| **lattice（单支 + 格点）** | **198,340（0.198 M）** | 拆除后，**↓68%** |

**③ 验证（服务器实测，`_r8c_verify.sh`）**

| 组 | 结果 |
|---|---|
| A 拆除完整性（10 项） | **PASS** — 无 hyperbolic import、无 r0/radius 参数、三个几何字段全 `none` |
| B 反自锁不回退（4 项） | **PASS** — 4 支 γ 梯度全非零 `{2.44,2.45,2.23,2.52}×1e-1`、`contrib_norm=615.7`；负对照 γ=0 ⇒ contrib=0 且梯度全 0 |
| C 结构（6 项） | **PASS** — Q=1024 / KV=256、窗口不整除报错、净网络 GPU 前向 `contrib=2.40e+02` |
| D 参数量（3 项） | **PASS** — `report == real == 198,340`，在结构性上界内 |
| E fail-closed（4 项） | **PASS** — `hyperbolic`/`euclidean`/`hgsl=1` 三种非法组合全部 `SystemExit` |
| **合计** | **27/27 PASS**（2026-10-20 12:02） |

**④ 边界（不许外推）**

lattice 版**没有任何训练结果**。27 项 PASS 只证明「代码对、能跑、有梯度、fail-closed」，
**不证明**它优于 CTRL。双曲臂的「不划算」结论也不能推广成「任何跨层 QKV 注入都无效」——
`B2_EUC`（欧氏反事实）只有 5.1%、无 TEST 区块。

**⑤ 停训时踩到的坑（已修进脚本）**

只杀 `python train.py` 不够：`_r8b_run_one_ds.sh` 里有 **10 s 重试循环**，
父进程活着就会把训练**复活**（实测 11:48 杀掉 → 11:50 自己回来，新 pid 720375 重新吃 GPU0）。
`_r8b_stop.sh` 已改为「**先杀重试宿主 → 再杀 train.py → 再观察 20 s**」，实测复活检查 4 拍全 0。

---

# Run9：HSAM（Poincaré 球双曲相似度注意力，**零参数**，进行中）

> 用户指令（2026-10-20）：按 JSTARS 2024 论文 Fig.2 的 HSAM，把编码器 3、4 层输出映射到双曲空间
> 算双曲距离、构造注意力矩阵，再经注意力聚合 + 残差增强双时相特征。**就加这个模块。Run9。**
> 预注册与冻结方案：`train_scripts/CASA-TViM/Run9/README.md` · `…/Run9/SELECTION_LOCK.json`
> 协议 `run9_hsam_poincare_exact80k_v1` · 变体 `S0_CTRL` / `S1_HSAM` · 8 个 run

**① 开跑前发现并修掉的两个**必然退化**（这是本轮最重要的技术结果）**

`outnorm{i}=LayerNorm2d(dim)` 在通道维把每个空间位置标准化 ⇒ **`‖v‖²=C` 精确成立**
（实测 `‖v‖/√C=1.0000±5e-7`，`1−‖p‖²` 对所有 token 恒为 `0.4200`）。于是所有点同半径：

```
‖p−q‖²=2r₀²(1−cosθ)，(1−‖p‖²)(1−‖q‖²)=sech⁴(1)=常数
⇒ HD = arcosh(1+K(1−cosθ))     ← 余弦距离的严格单调函数
```

实测秩相关 **`rank_corr(HD,1−cos)=1.000001/1.000000`** ⇒ **"双曲"不携带任何信息**，
模块等价于余弦注意力。若不开跑前查出来，60 小时 GPU 会去回答一个错误的问题。

| 修法 | 依据 | 效果（实测 rank_corr） |
|---|---|---|
| ① 改取 `outnorm` **之前**的激活（新增 `encoder.forward_pre_norm()`） | `o` 的范数随内容变化（cv 4.9e-2 / 1.1e-1） | 1.000000 → **0.83 / 0.29** |
| ② exp 映射前除以**全局 RMS 标量**（保序、零参数） | 不处理时 stage3 `1−‖p‖²` 触底 8.5e-6、半径压倒角度（rank_corr 掉到 0.0047） | 维持 **0.83 / 0.29** |
| ③ logit 逐样本标准化 | 论文原式 `tanh(HD/√C)` 注意力熵 = **1.0000**、`agg/feat=1.9e-4`（等于没加） | 熵 0.91 / 0.86 |

> **② 的注**：HSAM 在 `outnorm` **之前**混合、之后照常归一化 ⇒ **CTRL 路径一个字节都不变**。

**② 硬门（每个 run 都查，任一不过 ⇒ `SystemExit(4)`，不重试、终止整轮）**

`finite` · `agg_over_feat>1e-6` · `radius_cv>1e-3` · `attn_entropy_norm<0.995` ·
**`rank_corr_hd_vs_cos<0.999`**（本轮的核心门：挡住"以为在测双曲、其实在测余弦"）

**③ 开跑前实测（服务器，CDD 真实特征）**

| 项 | `hsam=0` | `hsam=1` |
|---|---:|---:|
| 预训练 | `212/222, worst_diff=0.0` | 同 |
| 总参数 | 31,977,044 | **31,977,044（HSAM 贡献 0）** |
| step/s | 3.691 | 3.581（**−3.08%**） |
| 峰值显存 | 26,798 MiB | 26,941 MiB |
| stage2 | — | `r_cv 5.7e-2` `rank_corr 0.829` `ent 0.909` `agg/feat 1.64` |
| stage3 | — | `r_cv 2.3e-1` `rank_corr 0.290` `ent 0.864` `agg/feat 1.21` |

CPU 回归门 `models/test_hsam_cpu.py` **32/32 PASS**，其中两条是**固定断言**：
"论文原式必然均匀（熵>0.9999）"、"post_norm 负对照 rank_corr≈1"。

**④ 预注册早停**：CDD+LEVIR 两臂都封口后，若两库 `ΔF1` 都 ≤ 0 ⇒ STOP（不再跑 SYSU/WHU），
结果按实际完成的报、不挑选。脚本 `_r9_delta.py`。

**⑤ 保留但本轮不跑**：`--hsam_radius local_dispersion`（半径只由 3×3 局部离散度决定、
与 token 能量解耦；代码已就绪并通过 CPU 检验），留作后续对照臂。

**⑥ 监控**

```bash
bash train_scripts/CASA-TViM/Run9/_r9_state.sh      # 进度 + HSAM 读数 + Δ 表
bash train_scripts/CASA-TViM/Run9/_r9_probe.sh      # 当前 stage 日志 + 督导决策
tail -20 /tmp/run9/pipeline.log
bash train_scripts/CASA-TViM/Run9/_r9_stop.sh --kill-run
```

---

## 探索线 Exp1：SPD(2)/H² 结构张量测地变化证据（2026-10-16 → 10-19，已完成）

> 独立于 Run8：代码在 **`models_v1/`**（`models/` 未改动，Run8 campaign 的身份链保持冻结），
> 只在 **LEVIR / SYSU** 两库训练（用户 2026-10-16 指令：这两库是硬目标缺口的所在）。
> 设计与预注册：`docs/temporary/CASA-CD_Exp1_SPD2结构张量测地证据_设计与预注册_2026-10-16.md`、
> `…_Exp1v2_鲁棒池化与跨尺度门_…_2026-10-17.md`、`…_Exp1B_特征域几何监督_…_2026-10-18.md`。
> 结果与归因：`…_Exp1_结果与机制归因_2026-10-17.md`、`…_Exp1v2_结果与三臂对比_2026-10-18.md`。
> 执行索引：`train_scripts/Exploration/Exp1/`；代码+指标快照：`docs/temporary/models_and_metrics_CASA-TViM_Exp1.txt`。

**方法（一句话）**：把每个像素的局部结构张量 J（Di Zenzo 矢量版）当作 **SPD(2)** 流形上的点
（det=1 切片 ≅ **曲率 −1/2 的双曲平面**），用仿射不变黎曼度量的**闭式测地距离**度量双时相
「结构变化」。数学结论（均已在 `models_v1/test_st_geodesic.py` 数值验收，23 项全过）：

* 拉回度量 `ds² = 2du² + 8sinh²u·dθ²`；测地距离 `cosh(d/√2) = 1 + 2W`，
  `W = sinh²(Δu/2) + sinh u₁sinh u₂·sin²(Δθ)` ⇒ **无 eigh、无半径参数**；
* **辐射不变性**：`I_B = c·I_A ⇒ d_shape = 0`（精确）、`I_B = I_A + d ⇒ 两项均为 0`（精确）
  ——在含强各向异性（κ→1）的最坏合成图上仍恒为 0；
* 尺度-形状正交分解 `d²_AIM = d²_shape + 2(log σ₁/σ₂)²`（数值对拍 <1e-4）；
* 单测同时守住了三个真实实现缺陷：误把 `z=tanh(u)e^{2iθ}` 当等距盘坐标（偏差 1.25）、
  `(1−cosΔφ)` 的 FP32 灾难性抵消（假距离 2.6e-3）、有理化时误用 `|cos|`（系统性偏小）。

### 三个臂（受控单变量；对照 = Run8 `CTRL_R8`，**epoch-0 逐位相同**，已实测 `torch.equal=True`）

| 臂 | 形态 | 与 CTRL 的差别 | 参数量（训练图 / deploy） |
|---|---|---|---|
| `STG_INJ`（v1） | 注入：`feats[0] += γ·Stem(证据)`，max 池化 15 通道 | 推理期真实算子 | +6,977 / +6,977 |
| `STG_V2` | 注入 v2：**top-4 均值池化** + **跨尺度一致性门**（16 通道） | 推理期真实算子 | +7,021 / +7,021 |
| `STG_SUP`（B） | **监督**：几何分数在 **stage0 特征**上算，GT 掩码监督（λ=0.2 固定），`aux_only` | **只改梯度方向** | +1,538 / **0（整支删除）** |

### 结果（同源最后一个完整 TEST 区块；LEVIR / SYSU）

| 臂 | LEVIR F1 | ΔvsCTRL | SYSU F1 | ΔvsCTRL |
|---|---:|---:|---:|---:|
| `CTRL_R8` | **0.9121** | — | 0.8188 | — |
| `STG_INJ`（v1） | 0.9114 | −0.08pp | 0.8168 | −0.20pp |
| `STG_V2` | 0.9118 | −0.03pp | 0.8172 | −0.16pp |
| `STG_SUP`（B） | 0.9109 | −0.12pp | **0.8201** | **+0.13pp** |

**判定**：三臂按预注册接受标准（两库均 ≥ CTRL 且 SYSU 严格提升）均 **未通过**
⇒ 记为受控负结果；但**机制读数全部按设计生效**，且给出了本项目在小目标上的**首个稳定正向证据**。

### 机制读数（这是本轮真正的产出）

* **γ 被完全采纳、绝非休眠**：v1 `+0.382/+0.201`、v2 `−0.398/−0.484`（Run4 FET1 是 ~1e-5）；
  B 的 `[STGEO-LOSS]` 0.3197→0.1707 单调下降，且**只回传辅助损失**时
  `encoder stage0` 梯度 = 3.21e-4 > 0（支配条件成立，已写进预检门）。
* **小目标召回在两库都稳定改善**（相对 CTRL）：
  LEVIR small pooled R **v2 +1.80pp**；SYSU small pooled R **v2 +2.77pp / B +2.81pp**、
  small Hit@25 **v2 +2.47pp / B +3.29pp**。
* **碎片化可被压回**：SYSU 未匹配 <256px 预测 CTRL 2950 → v1 **4393(+49%)** →
  v2 **3152(−28% vs v1)** → B 3360；LEVIR 585 → 551/563/**512(B，最优)**。
* **v2 的三条预注册预测全中**（碎片 −28.3% ≥ 20%、ObjP +5.87pp ≥ 3pp、LEVIR small R +1.80pp ≥ 0.5pp）
  ⇒ 「碎片化来自 max 池化 + 单尺度证据」这一归因**被数据证实**。
* **B 在 SYSU 上净胜 CTRL（+0.13pp）且推理零代价**：Recall **+1.62pp**（0.7966→0.8128）、
  large pooled R **+1.63pp**（v2 曾 −0.34pp）、small pooled R **+2.81pp**；
  SYSU 的 ObjP 也回到 0.4768（v2 0.4776，v1 0.4189）。
* **代价集中在 Precision**：B 在 LEVIR/SYSU 的 Precision 分别 −0.45/−1.48pp（Recall 反而最高）
  ⇒ 这是一个**工作点偏移**，而非能力缺失。

### 复用与工具

| 新增 | 作用 |
|---|---|
| `models_v1/model/layers/st_geodesic.py` + `test_st_geodesic.py` | 数学核心 + 23 项验收（含 3 个真实缺陷的回归守卫） |
| `models_v1/model/layers/stgeo_sup.py` | B 形态（特征域几何 + 监督头） |
| `models_v1/{train_stgeo,eval_stgeo_tb,epoch0_check,check_sup_wiring,eval_object_metrics}.py` | 训练 / 正式区块补算 / epoch-0 对拍 / 接线自检 / 对象级复算 |
| `train_scripts/Exploration/Exp1/` | 预检、启动、等卡督导（连续 2 次空卡才启动）、状态/曲线/对比脚本 |
| `.claude/_deploy_v1.py` | 只上传 `models_v1/` 与 `Exploration/`，**绝不触碰服务器 `models/`** |

### 结论与下一步

1. **有效**：几何证据确实携带小目标与抗碎片化信息（两库 small R +1.8~2.8pp、碎片 −28%），
   且 B 形态**首个在 SYSU 上净胜同期 CTRL**、推理零代价。
2. **不足**：三臂的净像素 F1 都没能同时超过 CTRL（B 在 LEVIR −0.12pp）。差异量级 0.03–0.20pp，
   属本项目单 seed 噪声，但**不得**因此回调 γ/阈值/σ/seed。
3. **下一步（按证据强度排序）**
   * **C（Beltrami / 各向异性扩散）**：把证据场先做边缘保持的 PDE 平滑再注入/监督，
     针对 B 的 Precision 代价（即"多报"），是最后一个算子级尝试；
   * **B + 工作点校准**：B 的 Recall 最高、Precision 最低 ⇒ 若能与 v2 的抑制特性**级联**
     （B 监督表征 + v2 注入抑制），有机会同时拿到两端；但这属于组合臂，须重新预注册；
   * **收束**：把三臂证据写成「负结果 + 机制归因」一节（含可复现的门槛、三条预注册预测、
     以及"碎片化归因被证实"这一正向结论）。

---

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
models/                        # 全部代码（两条主线：casa_tvim_str / casa_vmamba_str）
  train.py                     #   训练入口（--arch casa_tvim_str | casa_vmamba_str | casa_vmamba_str_v2）
  eval.py                      #   独立测试入口（同款架构参数，arch.json 侧车 fail-closed 校验）
  smoke_test.py                #   冒烟测试（--mode casa_tvim_str，T-CA-1..10）
  run8_dry_run.py              #   ★Run8 真实数据 dry-run（6 道门 + 折叠刀锋定位）
  test_run8_cpu_precheck.py    #   ★Run8 CPU 预检 26 项（退化定理复现 / 非退化性 / 预算 / HGSL）
  test_hqkv_lattice_cpu.py     #   ★Run8c 拆除回归门 27 项（无几何 / 反自锁 / fail-closed）
  model/
    tinyvim_s_slim.py          #   TinyViM-S-Slim（Stage4 裁剪 + 预训练 key 重映射）
    casa_tvim_str_net.py       #   CASA-TViM-STRNet（Run1–Run7 主线主架构）
    casa_vmamba_net.py         #   ★CASA-VMamba-STRNet（Run8 主线主架构）
    casa_vmamba_net_v2.py      #   ★Run8b 血统；**2026-10-20 已拆双曲**，现走 lattice QKV 注入
    vmamba_tiny.py             #   ★VMamba-Tiny 自包含主干（vssm_tiny_0230 口径，权重逐位加载）
    hgsl_loss.py               #   ★HGSL 双曲几何监督损失（实现保留；v2 已无分数输入口 ⇒ fail-closed）
    str_tar.py                 #   TAR（TemporalRep1x1 / TemporalRepFine3x3）
    str_dcr.py                 #   DCRDecoder + RepLocalBlock/RepPW1x1
    str_reparam.py             #   折叠原语（fold_conv_bn / RepDW3 等）
    str_fine_head.py           #   STRFineHead（Run2 FRH，可选）
    str_hf_edge.py             #   Run7 HFAE（高频边缘辅助监督）
    layers/ss2d.py             #   SS2D 自包含移植（oflex/mamba-ssm/Triton 三后端）
    layers/caacp_ss2d.py       #   CAACP-SS2D（Run1–Run7 创新一：score_mode / residual_mode）
    layers/caacp_hqkv.py       #   ★CAACP-HQKV v1（Run8，含自锁缺陷，仅作历史复现）
    layers/caacp_hqkv_lattice.py # ★Run8c：无几何的格点压缩跨层 SAT-QKV（当前 live 实现）
    layers/hyperbolic.py       #   ★Lorentz 双曲算子（v1 仍用；v2 已不引用，拆除记录见 Run8c）
    layers/vmamba_kernel.py    #   ★selective-scan 后端适配（绕过 casacd 里 ABI 已坏的 .so）
    layers/mamba_scan_triton.py#   vendored Triton 兜底 kernel
  dataset/                     #   DataLoader（A/B/label + list 格式）
train_scripts/
  CASA-TViM/Run1/              #   4 变体 × 4 数据集（A0/M1/A1/A2，已完成）
  CASA-TViM/Run2/              #   CP-CAACP / FRH / 组合 × 4（E1/E2/E3，已完成）
  CASA-TViM/Run3/              #   RA-CAACP / FS-TAR × 4（E4/E5，已完成）
  CASA-TViM/Run4/              #   R4-FET1 1×1 可折叠旁路 × 8 run（已完成，MECHANISM-FAIL）
  CASA-TViM/Run5/              #   原生高频证据复用前测（已完成，NO_80K；正式训练 0 次）
  CASA-TViM/Run6/              #   变化判别瓶颈可分辨性前测（已完成，NO_80K；正式训练 0 次）
  CASA-TViM/Run7/              #   HFAE 高频→边缘辅助监督消融（3 变体 × 4 库，已完成，MECHANISM-PASS；含收口 runbook §8/§9）
  CASA-TViM/Run8/              #   ★CASA-VMamba 换主干 + CAACP-HQKV + HGSL（4 变体 × 4 库，进行中）
                               #     双卡并行调度：_r8_run_one_ds.sh / _r8_pipeline_supervisor.sh
                               #     只读诊断：_r8_state.sh / _r8_eta.sh / _r8_gpu_survey.sh / _r8_preflight.sh
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
  tests/test_tvim_*.py         #   P0 单测（对象指标 + 日志解析器逐字对拍 + Run5/Run6 前测；合计 142 项）
  run2_zero_cost_diag.py       #   ⚠️ 历史诊断（component_pr 的 FP 恒为 0，口径已被 tvim_object_metrics 取代）
  run2_report.py               #   Run2 自动对比报告
  run3_report.py               #   Run3 自动对比报告（含 §13 预注册裁决）
  run7_object_metrics.py       #   ★Run7 三变体对象级只读复算（--all-variants/--compare；CPU 上拒绝运行）
others/                        # 参考实现（非本仓库模型代码）
  TinyViM-main/                #   TinyViM(ICCV2025) 官方核心（tinyvim.py + tvimblock.py）
  VMamba/                      #   VMamba(NeurIPS2024) 官方 vmamba.py（Run8 主干的结构来源）
  STR-RepNet/                  #   STR-RepNet 的 TAR/DCR/Mamba_backbone 参考实现
  SAT/                         #   SAT(CVPR2026) 官方 SAA 抽取版（CASAA 历史机制来源）
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

> **硬目标（≥98 / ≥92.5 / ≥85 / ≥95）是努力方向，不是必达门**（见上文「研究目标与实验纪律」§1）：
> 未达**不等于**失败；每个 Run 的有效性以其**预注册判据**为准（Run7 尤其只看**相对同期 CTRL**）。
> **绝不可**为了凑硬目标回调已冻结的 λ / 阈值 / seed。

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
# Run7（HFAE）：收口一条命令 + 交付状态自检 + 会话交接
bash train_scripts/CASA-TViM/Run7/finish_run7.sh          # 6 步收口；rc 含义见 Run7/README.md §8.2
#   交付状态一条命令核对（身份 26/26 / 四个工具自检 / 督导与进度）见 Run7/README.md §9
#   新接手者请从 train_scripts/CASA-TViM/Run7/HANDOFF.md §0 开始
# Run8（CASA-VMamba v1，已停止）—— 保留只读诊断脚本备查
bash train_scripts/CASA-TViM/Run8/_r8_state.sh        # 权威状态快照
bash train_scripts/CASA-TViM/Run8/_r8_eta.sh CTRL_R8  # 实测步速与剩余时间
# Run8b（CASA-VMamba v2 抗自锁，**当前主线**）
bash train_scripts/CASA-TViM/Run8b/_r8b_state.sh            # 进度矩阵 + 抗自锁证据 + 正式 TEST 块
bash train_scripts/CASA-TViM/Run8b/_r8b_timeline.sh         # 各库起止时间 + 实测速率 + ETA
bash train_scripts/CASA-TViM/Run8b/_r8b_key_readings.sh B1_HYP CDD-CD-256
                                                           # fail-fast / gamma_grad / HQKV-DIAG
tail -8 /tmp/run8b/pipeline.log                            # Run8b 督导决策日志
#   开跑：bash …/Run8b/_r8b_launch.sh
#   预注册与设计说明：train_scripts/CASA-TViM/Run8b/README.md
```

> ⚠️ **远端命令一律写进 `.sh` 文件再上传执行**：PowerShell 会吃掉内联命令里的
> `2>/dev/null`、`\$()` 与引号（会变成 `F:\dev\null` 之类），本项目已因此反复踩坑。


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
- **`setsid nohup bash <script>` 拉起的脚本不继承 conda 激活** ⇒ 脚本内必须显式
  `source conda.sh && conda activate casacd` + `export PATH/PYTHONPATH/LD_LIBRARY_PATH`
  （Run8 首启即因此四库各失败 3 次）。开跑前建议先跑 `_r8_preflight.sh`。
- **`nvidia-smi --query-compute-apps --format=csv,noheader` 的输出是 `pid, mem`（带逗号）**；
  用 `read -r pid _rest` 会把整行读进第一个字段，导致进程归属判断永远失配
  （Run8 曾因此把自己正在跑的 run 判成"别人的作业"）⇒ 必须显式 `IFS=,`。
- **`pgrep -f 'xxx.sh'` 是子串匹配**：`_r8_supervisor.sh` 会同时匹配
  `_r8_pipeline_supervisor.sh`（Run8 曾因此误杀刚拉起的督导）⇒ 要用锚定正则并在脚本内显式排除。
- **"ckpt 目录非空即拒绝启动"的守卫必须忽略封存残留**（`*SUPERSEDED*`）：
  否则按纪律"只改名不删除"的产物反而会把后续自动重启全部挡死。

