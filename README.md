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
| **CASA-VMamba-STRNet**（Run10） | **VMamba-Tiny**（NeurIPS 2024，`vssm_tiny_0230`） | **训练中**（SYSU→LEVIR，各 CTRL→HSAM local_dispersion，共 4 项） | 主干 **29.94M** + 非主干 **≈2.04M**（无硬上限） |

硕士课题：遥感二值变化检测。Run10 于 2026-10-21 17:19（CST）在 GPU0 启动，
按 SYSU 对照→SYSU HSAM→LEVIR 对照→LEVIR HSAM 串行跑满各 80K step。
HSAM 使用 `local_dispersion` 半径；冻结协议与停训命令见
[`train_scripts/CASA-TViM/Run10/README.md`](train_scripts/CASA-TViM/Run10/README.md)。

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




## 当前方法：CASA-VMamba-STRNet

共享 VMamba-Tiny（`vssm_tiny_0230`）编码双时相图像，四级特征经 TAR 融合与 DCR 解码后输出变化概率图。当前配置 `str_dim=192`、`rep_mode=full`，总参数 **31,977,044**：主干 **29,938,176**、TAR **1,351,688**、DCR **686,987**、预测头 **193**；HSAM 为零参数。

```text
[A;B] (2B) → shared VMamba-Tiny
    → F1(1/4,96), F2(1/8,192), F3(1/16,384), F4(1/32,768)
    → HSAM on pre-outnorm F3/F4（实验臂）→ outnorm
per-time [F1,F2,F3,F4]
    → MultiScaleTAR(D=192) → DCRDecoder(D=192)
    → head Conv1x1 → bilinear → sigmoid
```

- **HSAM**：在编码器 stage2/stage3（从 0 编号）的 `outnorm` 之前，用双曲距离构造跨时相注意力，双向聚合后残差增强特征，再通过原 `outnorm`。对照臂关闭 HSAM；本轮 `hqkv=0`、`hgsl=0`、`fs_tar=0`。
- **TAR/DCR 结构重参数化**：训练时使用多分支融合，部署时将可重参数化分支折叠到对应卷积；HSAM 保留为推理算子。
- **训练入口**：`models/train.py --arch casa_vmamba_str_v2`。主损失 BCE+Dice，80,000 optimizer step，batch 32，256×256，seed 16，Adam lr=2e-4、poly 调度，主干学习率比例 0.1。
- **预训练加载**：VMamba 权重保留 212/222 键，`worst_diff=0.0`。旧 TinyViM 主线的结构、4.88M 参数口径与实验结果见历史版本文档。

## Run10 当前训练：局部离散度半径

协议 **`run10_hsam_local_dispersion_exact80k_v1`**，代码基线 `d3e250e`。2026-10-21 **17:19 CST** 在 GPU0 启动，顺序为 **SYSU CTRL → SYSU HSAM → LEVIR CTRL → LEVIR HSAM**，共 4 项，每项从同一预训练权重与 seed 开始训练 80K step。

| 顺序 | 数据集 / 变体 | 状态（2026-10-21 17:35 CST） | 已记录 optimizer step |
|---|---|---|---:|
| 1 | SYSU / `S0_CTRL` | 训练中，已完成 epoch 6 | 2,625 / 80,000（3.28%） |
| 2 | SYSU / `S1_HSAM_LD` | 待启动 | — |
| 3 | LEVIR / `S0_CTRL` | 待启动 | — |
| 4 | LEVIR / `S1_HSAM_LD` | 待启动 | — |

**半径定义**：HSAM 实验臂使用 `local_dispersion`：计算每个 token 的 3×3 邻域方差，在通道上取均值再开方，以总体均值归一化得到 `rho`，映射后的球内半径为 `tanh(rho)`。半径由局部离散度决定，方向来自原特征。

**数值与退化检查**：局部方差采用 replicate padding 和正下界；双曲距离在重合点附近使用 float32 数值保护。每个验证 epoch 都检查有限输出、聚合贡献、半径 CV、注意力熵及 `rank_corr(HD,1−cos)<0.999`。任一训练报错或退化触发即停止队列，保留日志与检查点供分析。

**验证结果**：HSAM CPU 回归 **32/32 PASS**；后期硬门、重合点反向梯度、常量特征局部方差的新增回归通过。两库各做了 batch 8、2 step 的真实数据短测，HSAM 读数如下：

| 数据集 | stage2 / stage3 半径 CV | stage2 / stage3 秩相关 | 有限输出 |
|---|---:|---:|---|
| SYSU | 0.1546 / 0.2407 | 0.667442 / 0.654187 | 是 |
| LEVIR | 0.1164 / 0.2435 | 0.562803 / 0.711559 | 是 |

协议、冻结参数与执行脚本见 [`train_scripts/CASA-TViM/Run10/README.md`](train_scripts/CASA-TViM/Run10/README.md) 和 [`SELECTION_LOCK.json`](train_scripts/CASA-TViM/Run10/SELECTION_LOCK.json)。Feature Clipping 的论文公式及简化参考源码见 [`others/Tree-Changes-Detection-with-Siamese-Hyperbolic-network/README.md`](others/Tree-Changes-Detection-with-Siamese-Hyperbolic-network/README.md)：其上限裁剪用于限制嵌入靠近球边界，不能恢复 Run9 LEVIR 已消失的半径差异。

**已知限制**：短测只确认链路与初始机制可用，后期稳定性和性能提升由本轮完整训练判定。目前尚无 Run10 正式 TEST 结果。

# Run9 已停训结果：HSAM（Poincaré 球双曲相似度注意力，**零参数**，已停训）

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

**② Run9 预注册硬门（当时实现仅在首次诊断时检查）**

`finite` · `agg_over_feat>1e-6` · `radius_cv>1e-3` · `attn_entropy_norm<0.995` ·
**`rank_corr_hd_vs_cos<0.999`**（本轮的核心门：挡住"以为在测双曲、其实在测余弦"）

> Run9 的 `_log_and_check_hsam()` 在首次通过后由 `_hsam_ff_done` 提前返回，后续 epoch 仅记录读数。LEVIR epoch 154 的 stage3 `rank_corr=0.999206` 因而没有触发中止。Run10 已改为每个验证 epoch 执行硬门。

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

**④ 进度与结果（2026-10-21 14:41 停训）**

| run | 状态 | 正式结果 |
|---|---|---|
| `S0_CTRL/CDD` | 完成 80,000 step | 有完整 TEST 区块 |
| `S1_HSAM/CDD` | 完成 80,000 step | 有完整 TEST 区块 |
| `S0_CTRL/LEVIR` | 完成 80,000 step | 有完整 TEST 区块 |
| `S1_HSAM/LEVIR` | 34,565/80,000 后 NaN，3 次尝试均失败 | **无 TEST 区块，不得引用为指标** |
| `S0_CTRL/SYSU` | 用户停训时 9,375/80,000 | 无 TEST 区块 |
| `S1_HSAM/SYSU`、WHU 两臂 | 未启动 | — |

**有效封口 3/8**。正式结果仅取训练日志最后一个完整的 TEST 区块；未完成和失败日志只作取证。

| 库 / 变体 | Recall | Precision | OA | **F1** | IoU | Kappa | best ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDD `S0_CTRL` | 0.9777 | 0.9777 | 0.9945 | **0.9777** | 0.9565 | 0.9746 | 255 |
| CDD `S1_HSAM` | 0.9766 | 0.9765 | 0.9942 | **0.9765** | 0.9542 | 0.9732 | 255 |
| LEVIR `S0_CTRL` | 0.8979 | 0.9253 | 0.9911 | **0.9114** | 0.8372 | 0.9067 | 278 |

**CDD 同期差（HSAM − CTRL）**：ΔF1 **−0.0012**、ΔIoU −0.0023。LEVIR HSAM 无有效结果，不能计算同期差，也不能把本轮外推为四库结论。

**失败前最后一次机制读数（LEVIR epoch 154）**：stage3 `r_cv=4.133e-3`、`rank_corr=0.999206`（越过 0.999 退化门）、`agg/feat=2.3953`、`out/in=2.9588`；其后发生 NaN 与 BCE CUDA 断言错误。CDD 完成时 stage2/stage3 `rank_corr=0.7414/0.7588`，不能据此声称 LEVIR 全程非退化。

**⑤ 预注册早停与停训**

原早停规则要求 CDD 与 LEVIR 两库的两臂都有效封口，再判断两库 ΔF1 是否均 ≤ 0。LEVIR HSAM 没有正式 TEST 区块，**早停前提未满足**。2026-10-21 用户另行要求停训，督导、重试宿主与当前训练进程均已停止；不自动恢复队列。

服务器代码受控同步与运行身份记录见 [`docs/temporary/Run9_服务器代码同步_2026-10-21.md`](docs/temporary/Run9_服务器代码同步_2026-10-21.md)。

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
> 全部六指标在 `docs/experiment_metrics.xlsx`（101 行：98 条历史结果 + Run9 三条正式结果）。

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
models/                        # ★主线代码（当前 Run10）
  train.py                     #   训练入口（--arch casa_tvim_str | casa_vmamba_str | casa_vmamba_str_v2）
  eval.py                      #   独立测试入口（arch.json 侧车 fail-closed 校验）
  main.py                      #   上游 ChangeViT 原版入口（仅参考，未改动）
  smoke_test.py                #   冒烟测试（T-CA-1..10）
  test_hsam_cpu.py             #   HSAM 回归门 32 项（数值/退化证书/零参数/fail-closed）
  test_run10_cpu.py            #   Run10 后期硬门与数值梯度回归
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
    layers/hsam.py             #   ★Run10：Poincaré 球双曲相似度注意力（零参数，局部离散度半径）
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
  CASA-TViM/Run9/              #   已停训历史轮次，保留全部记录
  CASA-TViM/Run10/             #   ★当前主线：四项训练协议 / 串行督导 / 停训
  CASA-TViM/Diag1/             #   小目标瓶颈分阶段诊断流水线（只读）
  Exploration/Exp1/            #   探索线 Exp1 执行索引
others/                        #   参考实现（非本仓库模型代码）
  TinyViM-main/ VMamba/ STR-RepNet/ SAT/
  Tree-Changes-Detection-with-Siamese-Hyperbolic-network/  # Feature Clipping 参考
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
  - 当前 VMamba 预训练权重 `/home/yqwang/projects/CASA-CD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth`
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

1. 在没有本项目训练进程时同步模型与训练脚本：`python .claude/_deploy.py`。训练期间保持当前代码版本固定。
2. Run10 启动入口：
   ```bash
   cd /home/yqwang/projects/CASA-CD
   bash train_scripts/CASA-TViM/Run10/_r10_launch.sh
   ```
   队列在 GPU0 串行执行四项训练；每项使用独立的 `Run10/<variant>/<dataset>/` 日志与检查点目录。任一项失败即停止队列。
3. 日志开头记录全部配置、参数量和 FLOPs；每个 epoch 记录 Loss 与六项指标；完整正式测试区块以 `=== TEST RESULTS ===` 开始、`=== END TEST RESULTS ===` 结束。
4. checkpoint 包括 `last.pth`（含优化器）与 `best_F1=xxx.pth`。正式实验从预训练权重开始；Run10 启动脚本拒绝复用已有训练目录。
5. 训练完成后把 `train_log.txt` 下载到本地 `outputs/CASA-TViM/Run10/<variant>/<dataset>/`，提取完整 TEST 区块更新指标表，再生成代码与指标快照。

## 运行监控 / 分析

```bash
# ---- 当前主线 Run10 ----
tail -20 /tmp/run10/pipeline.log                    # 四项串行队列进度
tail -20 /home/yqwang/outputs/CASA-CD/CASA-TViM/Run10/S0_CTRL/SYSU-CD-256/train_log.txt
bash train_scripts/CASA-TViM/Run10/_r10_stop.sh    # 停止 Run10 进程组

# ---- 通用工具 ----
python .claude/_ssh.py '<cmd>'                     # 通用 SSH 执行（远端命令请写成 .sh 再上传）
python tools/extract_metrics_to_excel.py           # outputs → docs/experiment_metrics.xlsx
python tools/models_to_txt.py --tag CASA-TViM --run Run10  # 本轮完成后生成 models 代码快照 + 指标
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
  （Run8 首启即因此四库各失败 3 次）。Run10 开跑前预检入口为 `train_scripts/CASA-TViM/Run10/_r10_preflight.sh`。
- **历史重试队列的停训顺序**：`_r9_run_one_ds.sh` 内有重试循环，只杀 `python train.py`
  会让它在 10 s 后**自动复活**（Run8b 实测踩到）。`_r9_stop.sh` 的顺序是
  「督导 → run_one_ds → train.py → 观察 20 s」。Run10 使用 `_r10_stop.sh` 终止自己的进程组。
- **`nvidia-smi --query-compute-apps --format=csv,noheader` 的输出是 `pid, mem`（带逗号）**；
  用 `read -r pid _rest` 会把整行读进第一个字段，导致进程归属判断永远失配
  （Run8 曾因此把自己正在跑的 run 判成"别人的作业"）⇒ 必须显式 `IFS=,`。
- **`pgrep -f 'xxx.sh'` 是子串匹配**：`_r8_supervisor.sh` 会同时匹配
  `_r8_pipeline_supervisor.sh`（Run8 曾因此误杀刚拉起的督导）⇒ 要用锚定正则并在脚本内显式排除。
- **历史轮次的目录复用规则**：允许同轮恢复的脚本需要区分真实检查点与封存残留（`*SUPERSEDED*`）；
  Run10 的四项实验使用全新目录，不自动续训。
- **几何模块必须先验"非退化"**：LayerNorm 之后的特征范数恒为 `√C`，会让 Poincaré 半径恒定、
  双曲距离退化成余弦距离的单调变换（Run9 实测秩相关 = **1.000000**）。
  当前硬门为 `rank_corr(HD,1−cos) < 0.999`，Run10 在每个验证 epoch 执行检查。

