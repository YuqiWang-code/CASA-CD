## 代码与文档更新（默认**自动**）

**默认全自动**：本项目由 agent 在每次改完后直接 `git commit` + `git push`，
正常情况下你不需要做任何事。只有两种情形需要你手动介入，命令备查：

```bash
cd f:/Code_Repositories_2/CursorCode/CASA-CD
git add -A && git commit -m "<描述性提交信息>" && git push origin main
```

- 预训练权重 `pretrained_weight/*.pth` 不进 git（`.gitignore` 已忽略），
  需要时上传到 GitHub Releases（仓库页 → Releases → Draft a new release → 上传附件 `.pth`）。
- `docs/参考文献/` 的调研 PDF 随 `docs/` 一并提交（`.gitignore` 未忽略 PDF）。

---

# CASA-CD

遥感二值变化检测（变化感知上下文聚合 + 结构重参数化）。**当前主线只有一条**：

| 主线 | 骨干 | 状态 | 约束口径 |
|---|---|---|---|
| **CASA-VMamba-STRNet**（Run9） | **VMamba-Tiny**（NeurIPS 2024，`vssm_tiny_0230`） | **进行中**（`S0_CTRL` / `S1_HSAM` × 4 库） | 主干 **29.94M** + 非主干 **≈2.04M**（无硬上限） |

硕士课题：遥感二值变化检测。当前一轮（Run9）在 VMamba-Tiny 主干上验证
**HSAM（Poincaré 球双曲相似度注意力，零可学习参数）** 是否带来增益。

历史版本（Run1–Run8c、探索线 Exp1、Diag1）的设计、指标、判定与教训
**全部移到** [`docs/temporary/历史版本结论与指标_2026-10-21.md`](docs/temporary/历史版本结论与指标_2026-10-21.md)，
README 只留一张速览表。早期路线（ChangeViT baseline / CASAA / UltraLight / STR-Fusion /
CASA-STR/SHViT）已全部归档，见 git 历史、`docs/temporary/过去的想法/` 与
`docs/experiment_metrics.xlsx`。服务器与数据规范见
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

# 当前结果：Run9 · HSAM（Poincaré 球双曲相似度注意力，**零参数**，进行中）

> 用户指令（2026-10-20）：按 JSTARS 2024 论文 Fig.2 的 HSAM，把编码器 3、4 层输出映射到双曲空间
> 算双曲距离、构造注意力矩阵，再经注意力聚合 + 残差增强双时相特征。**就加这个模块。Run9。**
> 预注册与冻结方案：`train_scripts/CASA-TViM/Run9/README.md` · `…/Run9/SELECTION_LOCK.json`
> 协议 `run9_hsam_poincare_exact80k_v1` · 变体 `S0_CTRL` / `S1_HSAM` · 8 个 run

**① 开跑前发现的两个必然退化 + 对应的三处修正（本轮最重要的技术结果）**

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

> **修法 ① 的关键性质**：HSAM 在 `outnorm` **之前**混合、之后照常归一化
> ⇒ **CTRL 路径一个字节都不变**（`hsam=0` 时走的还是原来的 `forward_features`）。

**② 硬门（每个 run 都查，任一不过 ⇒ `SystemExit(4)`，不重试、终止整轮）**

`finite` · `agg_over_feat>1e-6` · `radius_cv>1e-3` · `attn_entropy_norm<0.995` ·
**`rank_corr_hd_vs_cos<0.999`**（本轮的核心门：挡住"以为在测双曲、其实在测余弦"）

**③ 开跑前实测（服务器，CDD 真实特征）**

| 项 | `hsam=0` | `hsam=1` |
|---|---:|---:|
| 预训练 | `212/222, worst_diff=0.0` | 同 |
| 总参数 | 31,977,044 | **31,977,044（HSAM 贡献 0）** |
| smoke step/s（6 步） | 3.691 | 3.581（−3.08%） |
| **campaign 实测墙钟（CDD 80K）** | **7h15m** | **7h24m（+2.1%）** |
| 峰值显存 | 26,798 MiB | 26,941 MiB |
| FLOPs（TEST 区块） | 14.6841 G | 14.7690 G（+0.58%） |
| stage2 | — | `r_cv 5.7e-2` `rank_corr 0.829` `ent 0.909` `agg/feat 1.64` |
| stage3 | — | `r_cv 2.3e-1` `rank_corr 0.290` `ent 0.864` `agg/feat 1.21` |

> 注：smoke 的 −3.08% 与 campaign 的 +2.1% 是**两种不同测量**（6 步 vs 满 80K 含 359 次 val），
> 两者并列，不要只引一个。真实代价以 **+2.1% 墙钟 / +0.58% FLOPs** 为准。

CPU 回归门 `models/test_hsam_cpu.py` **32/32 PASS**，其中两条是**固定断言**：
"论文原式必然均匀（熵>0.9999）"、"post_norm 负对照 rank_corr≈1"。

**④ 进度与结果（2026-10-21 11:25 实测）**

| run | 状态 | 耗时 |
|---|---|---|
| `S0_CTRL/CDD` | ✅ 封口 | 7h15m |
| `S1_HSAM/CDD` | ✅ 封口 | 7h24m |
| `S0_CTRL/LEVIR` | ✅ 封口 | 7h15m |
| `S1_HSAM/LEVIR` | 🔄 9,143 / 80,000（11.4%，2.95 step/s） | 预计 10-21 ≈18:05 完 |
| SYSU / WHU 各两臂 | 未开始 | — |

**封口进度 3/8**（CDD 两臂 + LEVIR 对照臂）。

| 库 / 变体 | Recall | Precision | OA | **F1** | IoU | Kappa | best ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDD `S0_CTRL` | 0.9777 | 0.9777 | 0.9945 | **0.9777** | 0.9565 | 0.9746 | 255 |
| CDD `S1_HSAM` | 0.9766 | 0.9765 | 0.9942 | **0.9765** | 0.9542 | 0.9732 | 255 |
| LEVIR `S0_CTRL` | 0.8979 | 0.9253 | 0.9911 | **0.9114** | 0.8372 | 0.9067 | 278 |
| LEVIR `S1_HSAM` | 在跑 | — | — | — | — | — | — |

**CDD 的 Δ（HSAM − CTRL）**：ΔF1 **−0.0012**、ΔIoU −0.0023、ΔRecall −0.0011、
ΔPrecision −0.0012、ΔOA −0.0003、ΔKappa −0.0014。

**判断标尺：跨轮 CTRL 可复现性**（同库同 80K 同 seed，只是协议串不同）

| 库 | 本轮 `S0_CTRL` | Run8b `CTRL_R8B` | 差 |
|---|---:|---:|---:|
| CDD | 0.9777 | 0.9776 | **+0.0001** |
| LEVIR | 0.9114 | 0.9121 | **−0.0007** |

> CDD 的 −0.0012 是 CDD 自身 CTRL 波动的 **12 倍**、是 LEVIR 波动尺度的 1.7 倍
> ⇒ 不是噪声能解释的量级；但绝对量仍只是**小负**，且**只有一个库**，
> 按预注册**不下结论**，等 LEVIR 对照臂。

**训练后的机制读数**（证明"非退化"贯穿训练，不只在初始化时成立）

| | CDD epoch 255 | LEVIR epoch 40（在跑） |
|---|---:|---:|
| `[HSAM-FAILFAST-PASS]` | step=626 | step=446 |
| stage2 `rank_corr(HD,1−cos)` | **0.7414** | **0.7932** |
| stage3 `rank_corr` | **0.7588** | **0.7309** |
| `r_cv` | 6.9e-2 / 1.4e-1 | 1.0e-1 / 2.2e-1 |
| `at_eps`（半径触底比例） | 0 | 0 |
| `agg/feat` | 1.04 / 1.37 | 1.26 / 0.96 |

> 两库全程 `rank_corr` 稳定在 **0.73–0.79**（离 0.999 的退化线很远）⇒
> 双曲成分在整个 80K 里都真实存在，这次不是"以为在测双曲、其实在测余弦"。
> **但机制成立 ≠ 性能成立**：CDD 上是小负。

**⑤ 预注册早停**

CDD+LEVIR 两臂都封口后（预计 **10-21 ≈18:05**），若两库 `ΔF1` 都 ≤ 0 ⇒ STOP
（不再跑 SYSU/WHU，省约 28.6 h）；结果按实际完成的报、不挑选。脚本 `_r9_delta.py`。
当前：CDD 已 −0.0012（≤0），**只等 LEVIR 一个判决**。
不触发则跑满四库，预计 **10-22 ≈22:40** 全部结束。

**⑥ 保留但本轮不跑**：`--hsam_radius local_dispersion`（半径只由 3×3 局部离散度决定、
与 token 能量解耦；代码已就绪并通过 CPU 检验），留作后续对照臂。

**⑦ 监控**

```bash
bash train_scripts/CASA-TViM/Run9/_r9_state.sh      # 进度 + HSAM 读数 + Δ 表
bash train_scripts/CASA-TViM/Run9/_r9_timeline.sh   # 逐库耗时 + 当前速率 + 不截断读数
bash train_scripts/CASA-TViM/Run9/_r9_probe.sh      # 当前 stage 日志 + 督导决策
tail -20 /tmp/run9/pipeline.log
bash train_scripts/CASA-TViM/Run9/_r9_stop.sh --kill-run
```

---

## 历史版本速览

> 全量叙述、逐 run 指标、机制读数与判定理由见
> [`docs/temporary/历史版本结论与指标_2026-10-21.md`](docs/temporary/历史版本结论与指标_2026-10-21.md)。
> 全部六指标在 `docs/experiment_metrics.xlsx`（98 行，已合并三个历史副本）。

| 版本 | 做了什么 | 判定 |
|---|---|---|
| Run1 | CAACP-SS2D + STR（TinyViM-S-Slim）主轴建立 | 4 变体 × 4 库完成 |
| Run2 | LEVIR/SYSU 定向提升（CP-CAACP / FRH） | 完成 |
| Run3 | SYSU 小目标瓶颈定向（RA-CAACP / FS-TAR） | E4/E5 均未过预注册条件 |
| Diag1 | SYSU 小目标瓶颈**只读**分阶段诊断 | 完成，未训练 |
| Run4 | R4-FET1 1/4 细尺度可折叠旁路 | `MECHANISM-FAIL` |
| Run5 | 原生高频证据复用**前测** | `NO_80K`（正式训练 0 次） |
| Run6 | 变化判别瓶颈可分辨性**前测** | `NO_80K`（正式训练 0 次） |
| Run7 | HFAE 高频边缘辅助监督 | `MECHANISM-PASS`（12 run，赢家 `E9_HFAE_A`） |
| Run8 | **换主干 VMamba-Tiny** + CAACP-HQKV（Lorentz 双曲双分支）+ HGSL | **`NO-OP`**：`γ` 零初始化自锁，模块恒零，是无效对照 |
| Run8b | 抗自锁重跑（`γ=0.5` + 每支独立门 + RMS 归一化） | **双曲臂 4/4 无正增益**（ΔF1 均值 −0.00085），8/16 后停训 |
| Run8c | **拆除**双曲几何与半径分解，只留格点压缩跨层 QKV | 代码改完并 27/27 PASS，未训练 |
| Exp1 | SPD(2)/H² 结构张量测地证据（`models_v1/` 独立线） | 三臂均未过门，已收束 |

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
models/                        # ★主线代码（当前 Run9）
  train.py                     #   训练入口（--arch casa_tvim_str | casa_vmamba_str | casa_vmamba_str_v2）
  eval.py                      #   独立测试入口（arch.json 侧车 fail-closed 校验）
  main.py                      #   上游 ChangeViT 原版入口（仅参考，未改动）
  smoke_test.py                #   冒烟测试（T-CA-1..10）
  test_hsam_cpu.py             #   ★Run9 回归门 32 项（HSAM 数值/退化证书/零参数/fail-closed）
  test_hqkv_lattice_cpu.py     #   ★Run8c 回归门 27 项（无几何/反自锁/fail-closed）
  test_str_reparam_equivalence.py  # 折叠原语等价性测试
  run9_smoke.py                #   ★Run9 真实数据 smoke（hsam 0/1 对比：读数/吞吐/显存）
  run9_degeneracy_cert.py      #   ★非退化证书（rank_corr(HD,1-cos)，pre/post_norm 对照）
  run9_radius_probe.py         #   ★半径诊断（outnorm 前/后 的范数与半径分布）
  _legacy/                     #   past-run 的 dry-run 与预检（Run4/7/8），保留备查
  model/                       #   网络与算子
    vmamba_tiny.py             #   VMamba-Tiny 自包含主干（vssm_tiny_0230 口径，权重逐位加载）
    casa_vmamba_net_v2.py      #   ★当前主线网络（CAACP-HQKV lattice + HSAM 双时相注入）
    casa_vmamba_net.py         #   Run8 v1 网络（保留，历史复现用）
    casa_tvim_str_net.py       #   Run1–Run7 主架构（TinyViM-S-Slim + CAACP-SS2D）
    tinyvim_s_slim.py          #   TinyViM-S-Slim（Stage4 裁剪 + 预训练 key 重映射）
    hgsl_loss.py               #   HGSL（实现保留；当前无分数输入口 ⇒ fail-closed）
    str_tar.py / str_dcr.py / str_reparam.py / str_fine_head.py / str_hf_edge.py
    layers/hsam.py             #   ★Run9：Poincaré 球双曲相似度注意力（零参数）
    layers/caacp_hqkv_lattice.py   # ★Run8c：格点压缩跨层 SAT-QKV（无几何）
    layers/caacp_hqkv.py       #   Run8 v1（含自锁缺陷，仅历史复现）
    layers/hyperbolic.py       #   Lorentz 双曲算子（v1 仍用）
    layers/caacp_ss2d.py       #   CAACP-SS2D（Run1–Run7）
    layers/ss2d.py             #   SS2D 移植（oflex / mamba-ssm / Triton 三后端）
    layers/vmamba_kernel.py    #   selective-scan 后端适配
  dataset/                     #   DataLoader（A/B/label + list 格式）
models_v1/                     # ★探索/额外任务区（自包含快照，不动主线）
  train_stgeo.py               #   Exp1 几何证据线（STG 系列）
  eval_object_metrics.py / eval_stgeo_tb.py / test_st_geodesic.py
  check_sup_wiring.py / epoch0_check.py
  train.py / eval.py / model/ / dataset/   # v1 快照（供上面脚本 import）
tools/                         # ★通用分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标
  param_breakdown.py           #   参数量/FLOPs 分解
  tvim_object_metrics.py       #   ★正确口径：GT 面积分组 / 对象级 / 边界带
  tvim_diag_common.py          #   诊断基础设施 + D0 对拍审计
  tvim_stage_raw_stats.py / tvim_stage_recoverability.py / tvim_linear_probe.py
  tvim_probe_stratified.py / tvim_diag_report.py / vit_pretrain_audit.py
  tests/                       #   通用工具单测（对象指标 + 日志解析器逐字对拍）
  archive/                     #   run 专用脚本（Run1–Run11 / Exp1），保留备查
train_scripts/
  CASA-TViM/Run1..Run8/        #   历史各轮（设计 + 执行 + 收口脚本）
  CASA-TViM/Run8b/             #   抗自锁重跑（已停训；含停训与拆除记录）
  CASA-TViM/Run9/              #   ★当前主线：预注册 / 督导 / 状态 / 停训 / 探针
  CASA-TViM/Diag1/             #   小目标瓶颈分阶段诊断流水线（只读）
  Exploration/Exp1/            #   探索线 Exp1 执行索引
others/                        #   参考实现（非本仓库模型代码）
  TinyViM-main/ VMamba/ STR-RepNet/ SAT/
outputs/                       #   训练日志（训练结束后下载到这里）
docs/                          #   文档：temporary/（历史舱 + 过去的想法）、参考文献/、服务器说明
.claude/                       #   部署与 SSH 辅助脚本（不进 git）
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

1. 本地改代码 → `python .claude/_deploy.py` 同步到服务器（models/ + models_v1/ + train_scripts/ + tools/）。
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
# ---- 当前主线 Run9 ----
bash train_scripts/CASA-TViM/Run9/_r9_state.sh      # 进度矩阵 + HSAM 读数 + 正式 TEST 块 + Δ 表
bash train_scripts/CASA-TViM/Run9/_r9_timeline.sh   # 逐库起止耗时 + 当前速率 + 不截断读数
bash train_scripts/CASA-TViM/Run9/_r9_probe.sh      # 当前 stage 日志 + 督导决策
tail -20 /tmp/run9/pipeline.log                     # 督导每拍决策
bash train_scripts/CASA-TViM/Run9/_r9_stop.sh --kill-run   # 停（顺序已修正，不会被重试循环复活）

# ---- 通用工具 ----
python .claude/_ssh.py '<cmd>'                     # 通用 SSH 执行（远端命令请写成 .sh 再上传）
python tools/extract_metrics_to_excel.py           # outputs → docs/experiment_metrics.xlsx
python tools/models_to_txt.py --tag CASA-TViM --run Run9   # models 代码快照 + 指标
python tools/tvim_object_metrics.py ...            # 对象级小目标复算（4 连通 / native 256²）
```

> Run1–Run8b 的监控/收口脚本仍在各自 `train_scripts/CASA-TViM/Run*/` 下，属历史备查；
> run 专用分析脚本在 `tools/archive/`。

> ⚠️ **远端命令一律写进 `.sh` 文件再上传执行**：PowerShell 会吃掉内联命令里的
> `2>/dev/null`、`\$()` 与引号（会变成 `F:\dev\null` 之类），本项目已因此反复踩坑。


## 注意事项

- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（train.py 已处理）。
- `.sh` 脚本需 LF 行尾（Windows 编辑后由 `_deploy.py` 上传，本地已确认 LF）。
- 训练脚本必须导出 `LD_LIBRARY_PATH=.../torch/lib:.../env/lib`（oflex kernel 依赖）。
- `run_all.sh` 的 `wait` 必须传 `$pid`（数值），直接传变量名会被 bash 当作 job 名。
- **`setsid nohup bash <script>` 拉起的脚本不继承 conda 激活** ⇒ 脚本内必须显式
  `source conda.sh && conda activate casacd` + `export PATH/PYTHONPATH/LD_LIBRARY_PATH`
  （Run8 首启即因此四库各失败 3 次）。开跑前先跑 `_r9_preflight.sh`。
- **停训必须先杀"重试宿主"**：`_r9_run_one_ds.sh` 内有重试循环，只杀 `python train.py`
  会让它在 10 s 后**自动复活**（Run8b 实测踩到）。`_r9_stop.sh` 的顺序是
  「督导 → run_one_ds → train.py → 观察 20 s」。
- **`nvidia-smi --query-compute-apps --format=csv,noheader` 的输出是 `pid, mem`（带逗号）**；
  用 `read -r pid _rest` 会把整行读进第一个字段，导致进程归属判断永远失配
  （Run8 曾因此把自己正在跑的 run 判成"别人的作业"）⇒ 必须显式 `IFS=,`。
- **`pgrep -f 'xxx.sh'` 是子串匹配**：`_r8_supervisor.sh` 会同时匹配
  `_r8_pipeline_supervisor.sh`（Run8 曾因此误杀刚拉起的督导）⇒ 要用锚定正则并在脚本内显式排除。
- **"ckpt 目录非空即拒绝启动"的守卫必须忽略封存残留**（`*SUPERSEDED*`）：
  否则按纪律"只改名不删除"的产物反而会把后续自动重启全部挡死。
- **几何模块必须先验"非退化"**：LayerNorm 之后的特征范数恒为 `√C`，会让 Poincaré 半径恒定、
  双曲距离退化成余弦距离的单调变换（Run9 实测秩相关 = **1.000000**）。
  现行硬门是 `rank_corr(HD,1−cos) < 0.999`（`models/test_hsam_cpu.py` + `train.py` 的 fail-fast）。

