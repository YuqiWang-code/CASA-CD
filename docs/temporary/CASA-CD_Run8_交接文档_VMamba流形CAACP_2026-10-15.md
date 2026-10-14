# CASA-CD Run8 交接文档：换 VMamba-tiny 主干 + 流形空间双分支 SAT-QKV 重构

> **生成**：2026-10-15（服务器实测时间 2026-10-15 00:09）
> **上游**：Run7 已于 2026-10-12 12:36 完成（12/12，`STATUS = MECHANISM-PASS`，winner = `E9_HFAE_A`），
> 归档提交 `23eb250`（`HEAD == origin/main`）。
> **读者**：新会话的 agent。**默认你没有上一轮对话的任何记忆**，本文是唯一权威交接。
> **建议阅读顺序**：§1（用户原话）→ §2（三个 P0，不裁定不要写代码）→ §4（环境实测事实）
> → §5/§6/§7（改哪里、怎么传、怎么收口）→ §8（迭代纪律）→ §9（已证明/已否证）。

---

## 0. 三十秒速览

| 项 | 状态 |
|---|---|
| 项目 | 极轻量全监督遥感二值变化检测（CDD / LEVIR / SYSU / WHU，256×256，A/B/label，`gray>=128`） |
| 当前主线 | **CASA-TViM-STRNet** = TinyViM-S-Slim 主干 + CAACP-SS2D + TAR/DCR，deploy **4,880,190 ≤ 5M** |
| 已完成 | Run1–Run7；Run7（HFAE 高频→边缘辅助监督）**首次 `MECHANISM-PASS`**，四库 0.9705/0.9134/0.8331/0.9507 |
| 本轮任务 | **换主干为 VMamba-tiny** + **重做 CAACP**（流形空间 + 细节/全局双分支特化 QKV + SAT 式 K/V 压缩 + 注入 + 高频损失） |
| ⚠️ 最大风险 | **VMamba-Tiny 官方约 30M 参数，与冻结的 deploy ≤5M 硬约束直接冲突**；且服务器目前**没有 VMamba 的 CUDA kernel、没有 VMamba 代码、没有 VMamba 权重** |
| 第一件事 | 读 §2，请用户裁定「≤5M 是否解除」。**在这之前不要新建任何 `models/` 代码** |

---

## 1. 用户原始指令（**逐字保留，不得改写或弱化**）

> 我现在要开个新对话框，还是做这个项目，但是要做换 backbone（换成 VMamba-tiny），重做 CAACP 模块
> （对 VMamba 编码器每层的输出都进行 QKV 或者跨层 QKV 或者 Q 和 K 和 V 来自 T1 或 T2 或者交叉，
> 然后融合或者 concat，注入给跳层连接或者解码器。这个"VMamba 编码器每层的输出"在做 QKV 之前也要处理，
> 把它们映射到流形空间中（双曲流形空间或者黎曼流形空间），然后在空间中做提取和分解和处理，
> 分出细节局部纹理边缘相关性比较高的分支和全局整体语义相关性比较高的分支，这两个分支要分别做各自
> 特化的 QKV，然后再注入到后面。所有的 QKV 都要用 SAT 那个文章中的思想，把 V 和 K 用流形空间中的
> 双时相差异感知方式聚合为 k*C 的维度，但是 Q 必须还是 N*C 的维度。（这个流形空间应用于遥感变化检测
> 的调研请看我上传的文档），然后对于高频的处理除了要注入主特征流，还要做损失加到损失函数里。）

**拆成可执行条目（我的转写，不是用户原话；有歧义处已在 §11 列出待确认）**

| # | 要求 | 确定性 |
|---|---|---|
| R1 | 主干换成 **VMamba-tiny** | 明确 |
| R2 | **重做 CAACP**：对 VMamba 编码器**每层**输出做 QKV | 明确（"每层"范围待确认，§11-Q2） |
| R3 | QKV 的 Q/K/V 来源可为：同时相 / 跨时相（T1、T2）/ 交叉；之后融合或 concat | 明确但**组合空间很大**，需收敛 |
| R4 | 注入目标：**跳层连接或解码器** | 明确（二者选一待确认，§11-Q3） |
| R5 | 做 QKV **之前**，先把层输出**映射到流形空间**（双曲或黎曼） | 明确（用哪个几何待确认，§11-Q4） |
| R6 | 在流形空间内做**提取/分解**，分出**细节局部纹理边缘分支** 与 **全局整体语义分支** | 明确，但"如何分解"是核心未定问题 |
| R7 | 两个分支**各自做特化 QKV** | 明确 |
| R8 | 所有 QKV 按 **SAT 思想**：**V、K 用"流形空间中的双时相差异感知"方式聚合到 `k*C`**；**Q 保持 `N*C`** | 明确（这是全案最硬的技术约束） |
| R9 | 高频处理：**既注入主特征流，又把损失加进损失函数** | 明确 —— ⚠️ 见 §2-P0-3 与 §9：**Run7 已实测"注入"无增益** |
| R10 | 流形空间应用于遥感 CD 的调研 = 用户上传的两份文档（已入库，见 §3） | 明确 |

---

## 2. ⚠️ 必须先裁定的三个 P0（**不裁定就不要写代码**）

### P0-1　deploy ≤5M 硬约束是否解除？——这一步决定整个方案是否可行

**已核验的事实（不要重新论证，直接引用）**

| 事实 | 来源 |
|---|---|
| 项目级硬约束：**有效推理（deploy）参数量 ≤5M**；且"方案若显著增加推理参数量/FLOPs、破坏预训练权重加载或引入不可折叠分支，**默认否决，除非用户明确改变约束**" | `docs/ChatGPT_Project_Settings.md`【硬约束】段 |
| 官方 **VMamba-T ≈ 30M 参数 / 4.9G FLOPs / ImageNet Top-1 82.6%**（v2 主表；v1 会看到 22–28M，引用时必须注明版本） | `docs/temporary/过去的想法/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md` §4.1 |
| 用户自己的 STR-RepNet 已实测：**VMamba-Tiny + TAR/DCR ≈ 29.57M → 98.42 / 91.44 / 83.45 / 95.14**，即**放宽到 30M 也仍然 LEVIR < 92.5、SYSU < 85** | 同上 §0.2 / §(a) |
| 把 30M 硬裁到 ≤5M 需去掉约 80% 参数；**宽度裁剪会破坏预训练 shape，深度裁剪后无法假设 82.6% 表征仍存在** | 同上 §0.2 |
| 该调研当时的结论：**首选 TinyViM-S，而非完整 VMamba-Tiny**——正因如此 Run1–Run7 全建在 TinyViM-S-Slim 上 | 同上 §0.2 / §4.4 |

**⇒ 三个互斥选项，请用户明确选一个（这是本轮第一个动作）**

| 选项 | 含义 | 代价 |
|---|---|---|
| **(a) 解除约束到 ~30M** | 可上完整 VMamba-Tiny。但论文"极轻量 / ultra-light"定位随之改变，需要导师签字（项目纪律里"骨干级/预训练级重立项需导师签字"）；且须自行解决 §4.2 的三个阻塞 | 定位变化 + 依赖代价；且已实证 LEVIR/SYSU 仍不达标 |
| **(b) 保持 ≤5M** | **VMamba-tiny 不可行**。退路：更小的 Mamba 主干（EfficientVMamba-T 6M / MSVMamba-Nano ~7M / MobileMamba-T2 8.8M，均仍>5M）或**继续用 TinyViM-S-Slim**，只做 R2–R9 的 CAACP/流形改造 | 与用户 R1 冲突，必须回问 |
| **(c) ≤5M + 截断版 VMamba-Tiny** | 深度/宽度裁剪到 5M 以内 | 调研明确判为**高风险**（预训练表征不可假设保留） |

> **不允许的做法**：不裁定就开始写 VMamba 代码；或悄悄把 deploy 预算写大；或在论文里把 30M 说成"轻量"。

### P0-2　"一次改四件事" vs 唯一变量分阶段

用户 R1–R9 同时改变了 **①主干 ②压缩机制 ③几何空间 ④注入点与损失**。而项目前 7 轮的全部教训
（§9）是：**多变量同改则无法归因**，且每轮都是"看起来合理 → 80K 跑完 → 无法说明是哪一项起作用"。

**建议（需用户同意）**：把 R1–R9 拆成**互不混淆的阶段**，每一阶段只有一个新变量、各配同期 CTRL：

```
Stage A  R1         : VMamba-tiny 主干 + **原封不动的 CAACP/STR**        → 只回答"换主干值多少"
Stage B  R5+R6      : 流形映射 + 细节/全局分解（**仍用原 QKV 形式**）     → 只回答"流形分解值多少"
Stage C  R8         : SAT 式 K/V 压缩（k*C）替换原 QKV（**流形+分解已固定**）→ 只回答"压缩值多少"
Stage D  R7         : 两分支特化 QKV                                     → 只回答"特化值多少"
Stage E  R3+R4+R9   : QKV 来源组合 / 注入点 / 高频损失                   → 只回答"注入与损失值多少"
```

**并且 Stage A 必须先出一个"新主干同预算 CTRL"**，因为主干一换，Run7 的任何数字都不再是同期对照
（两份调研文档都明确写了这一点：`不使用历史 Run7 当同期对照`）。

### P0-3　三条"已经有前例 / 已经否证"的硬提醒

1. **SAT 式 K/V 压缩在本项目已经做过两轮，并被终止。** 历史上 CASAA（`layers/casaa.py`）就是
   "Full Q + 压缩 K/V"，Run1/Run3 一路做到 A4；终止结论是 **"deployable routing ranking 提升不等于 F1 提升"**。
   ⇒ R8 必须交代：与 CASAA 的实质差异是什么？（新的差异只有"流形空间 + 双时相差异感知聚合 + 双分支特化"，
   这三条都还没有实验证据）
2. **Run7 已实测：把高频证据"注入"主特征流没有增益。** Run7 的 E10（`aux_inject`，零初始化注入 DCR 1/4 输入）
   在 3/4 库 ≥ CTRL、但 **不优于 E9（`aux_only`，只加损失不注入）**；且 E10 在 SYSU 上
   **小目标对象级召回 −0.0351、未匹配小预测 ×1.38**。⇒ 用户 R9 的"注入 + 损失"里，
   **"损失"是已验证有效的承重部件（Run7 首次 MECHANISM-PASS 靠的就是它），"注入"是已验证无增益的部件**。
   再引入注入必须先给出与 E10 不同的机制理由（例如注入位置/尺度/门控方式不同），否则会重复 E10。
3. **几何退化陷阱（固定半径双曲距离 ≡ 余弦排序）。** 两份调研文档 §2.3 / §5.2 都给出了同一个论证：
   若把单位特征以**相同固定半径**映射到双曲面，则双曲测地距离与原余弦差异**严格单调对应**，
   `rank_normalize_2d` 随即产生**相同排序** ⇒ 双曲版与余弦版给出同一组 cell 权重，
   "把 1−cos 换成双曲距离"**不构成任何新增信息**，也不构成创新。
   ⇒ 任何流形映射的**首要 smoke 测试**必须是"非退化性"证明：在固定方向相似度下，新几何**改变了重要性排序**。

---

## 3. 必读材料清单（按优先级，全部已在本地仓库内）

| 优先级 | 文件 | 为什么读 / 关键结论 |
|---|---|---|
| ★★★ | `docs/temporary/CASA-CD_双曲流形与变化感知_CAACP_文献调研及创新实验方案_2026-10-13.md` | **用户上传的流形调研（667 KB→67 KB 正文）**。含：现有 CAACP 的代码级审计；Poincaré 球公式；**§2.3 固定半径退化反例**；2024–26 文献地图（HES-Net / HyperCLIP-CD / HSN 是直接先例，**不能主张"首次"**）；§5 候选机制横评；§6 首选 **CRH-CAACP** 完整数学定义（参数自由、A/B 严格对称、epoch-0 逐位恢复）；§8 可证伪假设与失败判据；§9 P0/P1/P2 风险；§10 smoke 清单 |
| ★★★ | `docs/temporary/CASA-CD_双曲几何区分真实伪变化与小目标检测_深度文献核验及可证伪方案_2026-10-14.md` | **用户上传的第二份**。结论：**现有论文没有证明"负曲率本身能可靠识别小目标或伪变化"**；§6 **GTRC**（跨时相局部测地关系形状变化）；§7.1 **Geometry-Conditioned HFAE**（训练期高频，deploy 全删）；§7.2 测地可靠性 CAACP；§9 双曲空间**不能突破空间信息损失**；§11.2 **G0–G4 正式消融矩阵**（G0=E9 同期 CTRL，G4=同式欧氏反事实）；§11.3 验收读数 |
| ★★★ | `docs/ChatGPT_Project_Settings.md` | **"gpt 相关文件"** = ChatGPT 网页版项目指令（<8000 字符）。含：任务范围、研究定位、**证据纪律**、**硬约束 ≤5M**、项目环境、文献调研要求、工作方式与交付格式。新会话应把同一纪律搬过来 |
| ★★★ | `docs/temporary/过去的想法/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md` | **VMamba 可行性已被评估过**：VMamba-T/S/B 参数与 FLOPs；TinyViM-S vs EfficientVMamba-T；≤5M 预算重分配（逐层参数估算、slim trunk 4.644M + STR head 0.235M ≈ 4.88M）；**§6.1/§6.3：SSM 比 Attention 更怕 token 重排**（CrossMerge 依赖同一 H×W 网格；STORM ICML 2026 指出 spatially-agnostic token reduction 会让结构化 Mamba 崩溃）；§7.3 CAM-CD / §7.4 Mamba-CD novelty 碰撞；§15 selective_scan kernel 注意事项 |
| ★★ | `others/SAT/saa.py`（+`others/SAT/README.md`） | **SAT 参考实现已在仓库**。SAT = *Selective Aggregation Transformer for Image Super-Resolution*, Dinh Phu Tran et al., **CVPR 2026 Findings / arXiv:2604.07994**（venue 按项目纪律需**再核验**一次）。机制：**full-resolution Query + 压缩 K/V**，密度引导选中心 + 相似度加权聚合 K/V，宣称 ~97% K/V token reduction。⚠️ SAT 是**单图、超分、内容密度**驱动，**不是双时相变化感知**——R8 的"双时相差异感知聚合"是与它的实质差异所在 |
| ★★ | `docs/参考文献/baseline/SAT(CVPR2026).pdf` | 用户提供的 SAT 原文 PDF |
| ★★ | `train_scripts/CASA-TViM/Run7/README.md` | **Run7 runbook**：§7 全部事前/事后验证（含 §7.7 hook 缺陷、§7.16）；**§8 收口 runbook（8.1 一条命令、8.2 退出码表、8.3 核验、8.4 五种 verdict 写法、8.5 三张表、8.5b 根 README 结果节骨架、8.6 清单、8.7 禁做、8.8 恢复、8.9 可重入）**；§9 交付状态一条命令核对；§10 结果机器真值。**Run8 的 runbook 直接从这份改** |
| ★★ | `train_scripts/CASA-TViM/Run7/HANDOFF.md` | 上一轮交接（状态/ETA/接手顺序/恢复流程/冻结项），**格式即本文的模板** |
| ★★ | `docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md` | Run7 预注册（§1–§6 冻结 + §7 实施澄清 + **§7.6 刀锋像素澄清** + **§7.7 hook 缺陷与整轮重跑披露**）。Run8 预注册照此结构写 |
| ★ | `README.md`（根，L615 起） | Run7 索引节 + 「# Run7 结果」节（三张表 + 归因 + 结论 + 两条事故披露）；`## 目录结构`、`## 数据集`、`## 服务器环境`、`## 训练/测试` |
| ★ | `docs/RSML-3_服务器环境与变化检测数据统一说明.md` | 服务器与数据集的权威说明 |
| ★ | `docs/参考文献/文献索引.md` | 2024–26 文献总索引（新增文献先在此登记） |
| ★ | `docs/temporary/过去的想法/CASA-CD_CASAA_Run1复盘与下一步实验决策.md`、`CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md`、`CASA-CD_下一步方案_Run11_TASS_设计与预注册.md` §12.7 | **SAT/CASAA 的历史**：为什么终止（"deployable routing ranking 提升 ≠ F1 提升"）。R8 必须读，避免第三次重做同一件事 |

---

## 4. 环境事实（**2026-10-15 00:09 实测，不要凭记忆**）

### 4.1 服务器（RSML-3，用户 `yqwang`）

| 项 | 值 |
|---|---|
| 项目目录 | `/home/yqwang/projects/CASA-CD`（**非 git 工作副本**，只做运行） |
| conda | `source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate casacd` |
| 运行时 | Python **3.10.21**、**torch 2.14.0+cu132**、CUDA 13.2、GPU 2× **RTX 5090**（32 GB，`sm_120`） |
| 必须导出的环境变量 | `export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}`；`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 数据集 | `/share_datasets/CD/{CDD,LEVIR,SYSU,WHU}-CD-256`，list 长度实测：CDD 10000/3000、LEVIR 7120/2048、SYSU 12000/4000、WHU 5947/744（train/test） |
| checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/<Run>/<variant>/<dataset>/` |
| 日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/<Run>/<variant>/<dataset>/train_log.txt` |
| 磁盘 | 根分区 3.7 T，已用 958 G，**2.6 T 空闲** |

### 4.2 ⚠️ VMamba 的三个硬阻塞（未解决前 R1 无法开工）

| 依赖 | 实测结果 | 影响 |
|---|---|---|
| `selective_scan_cuda`（**VMamba 官方用的 kernel**） | **ImportError（缺失）** | VMamba 的 `VSSBlock` 直接 import 它 ⇒ **代码跑不起来** |
| `selective_scan_cuda_oflex`（TinyViM 用的） | **OK**（`.so` 在 site-packages） | 现有 TinyViM-S 能跑；**不能假定可平替 VMamba 的接口**（入参/返回值/梯度签名不同） |
| `mamba_ssm` | pip 显示 `2.3.2.post1` 已装，但 **`import mamba_ssm` 失败** | 同时 `causal_conv1d` **缺失**（mamba_ssm 的依赖）⇒ 需要重新编译或修版本 |
| `vmamba` python 包 | **缺失** | 需引入官方 `vmamba.py`/`vss.py`（本地 `others/` **也没有** VMamba 参考代码） |
| VMamba 预训练权重 | `pretrained_weight/` 只有 `deit_tiny_patch16_224-a1311bcf.pth`、`mobilenet_v3_small-047dcff4.pth`、`shvit_s1.pth`、`tinyvim_s_1000e.pth` | **没有 `vssm_tiny` 权重** ⇒ 需下载并上传（服务器能否联网需先测） |
| `others/`（服务器） | **空目录**（本地有 `others/TinyViM-main`、`others/SHViT`、`others/SAT` 等） | 服务器上没有参考实现；`_r7_stage.py` 的清单里也不含 `others/` |
| `openpyxl`（服务器） | **缺失** | 服务器上 `analyse/tests` 有 2 项 skip（属正常）；**xlsx 重建必须在本机做**（本机有 openpyxl） |

> **行动建议（Stage A 第 0 步）**：先写一个只读探针回答三个问题——
> ① 服务器能否联网/有无 pip 源；② 能否编出 VMamba 要的 `selective_scan_cuda`（torch 2.14 + CUDA 13.2 + sm_120
> 当年官方**没有可用 wheel**，TinyViM 就是用 oflex 绕过的）；③ 若编不出来，是否接受"用 oflex 自己写适配层"。
> **在 kernel 与权重两件事落定之前，不要写模型代码。**

### 4.3 GPU 现状与纪律

- 实测时 **GPU0 空闲**（166 MiB / 0%），**GPU1 被子进程占用**：两个**同用户但不同项目**的作业
  （`models.scripts.train --experiment_id C3*-Z --dca_mode moe128 --teacher_package dinov3_sat`，
  各 ~12 GB，其中一个 100% 利用率）。
- **纪律**：
  1. **绝不动别人的进程**（哪怕是同用户另一项目）；
  2. 开跑前先 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` 记账；
  3. 本项目历史上有"每卡 2 run"的打包方式，**在 GPU0 上同时压两个最重的 run 已经接近 31.35 GB 上限**，
     本次 Run7 波三就是因此 OOM（§8.3 坑 #9）⇒ **Run8 排卡时必须留余量**，最重的库不要与另一个重库同卡。

### 4.4 本地环境

- 工作目录 `F:\Code_Repositories_2\CursorCode\CASA-CD`（git 仓库，`main` 分支，remote `origin` = GitHub）。
- **本地没有 GPU / 没有 torch 的 CUDA**：所有训练、验收测试（需要 `selective_scan` kernel 的）、
  对象级复算都在**服务器**上跑；本地只做：编辑代码、hash/账本、上传、下载、xlsx 重建、快照、git。
- 本地有 Python（可跑 `py_compile`、hash、`_r7_finalize.py`）与 PowerShell（**没有 bash**）。
- `.gitattributes`：`* text=auto eol=lf`，**例外** `models/model/utils.py text eol=crlf`
  （因为它的 CRLF 字节已经被写进**每一个 run 的 `arch.json`**，中途归一化会破坏身份核验）。**不要改这两条。**

---

## 5. 代码地图：改哪里

### 5.1 `models/` 拓扑（Run7 结束时的真实状态）

```
models/
  train.py                     # 训练入口；PROTOCOL_VERSION；SOURCE_IDENTITY_FILES；exact-80K；
                               #   模型构造（arch == casa_tvim_str 分支）在 :273 起；
                               #   fold 等价（train↔deploy）检查在 :884-911；epoch 循环 :750
  eval.py                      # 独立评估入口（部署图；--batch_size，没有 --test_batch_size）
  model/
    casa_tvim_str_net.py       # ★当前主线 CASA-TViM-STRNet（Run8 大概率要新建同构文件）
    tinyvim_s_slim.py          # TinyViM-S-Slim 主干（widths 48/64/168/224, depths 3/3/9/6）
    str_tar.py  str_dcr.py  str_reparam.py   # TAR / DCR / 重参数化（可折叠）
    str_fine_tap.py            # Run4 FET1（1/4 细尺度 1×1 可折叠旁路）
    str_hf_edge.py             # ★Run7 HFAE（高频边缘辅助监督：taps / edge target / aux loss / HFEdgeHead）
    layers/
      ss2d.py                  # ★SS2D（四向 CrossScan/CrossMerge + local_conv 高低频分解）
      caacp_ss2d.py            # ★当前 CAACP 实现（Run8 要重做的就是这个）
      casaa.py                 # ★历史 CASAA（SAT 式 Full Q + 压缩 K/V），已终止
      mamba_scan_triton.py     # selective_scan 的 triton 兜底
      attention.py mlp.py patch_embed.py drop_path.py layer_scale.py block.py swiglu_ffn.py ...
    dataset/  utils.py  metric_tool.py
  test_run7_hf_edge.py         # ★T0–T12 验收测试（含 T12 hook 幂等回归）
  run7_dry_run.py              # ★真实数据 3-step dry-run（Run8 应仿写 run8_dry_run.py）
  smoke_test.py                # 小 smoke
```

### 5.2 预训练兼容三条铁律（**Run1–Run7 全靠它们保住"epoch-0 逐位一致"**）

1. **新模块最后构造** + `fork_rng` 局部种子 ⇒ 开关新机制时，既有模块的初始化/预训练加载**逐位相同**。
2. **门控/旁路零初始化**（`β=0`、`γ=0`、inject 零初始化）⇒ epoch-0 输出与"关掉该机制"**逐位一致**。
3. **重参数化分支必须可折叠为单卷积**，且 `switch_to_deploy()` 后实测参数量必须 ≤ 预算；
   `train↔deploy` 折叠等价性以 **0.5 二值 disagreement = 0** 为门。

### 5.3 `train.py` 的几个必改/必看接线点

| 位置 | 作用 | Run8 注意 |
|---|---|---|
| `PROTOCOL_VERSION`（:51） | 写入 `arch.json`/`run_manifest.json`/`last.pth`，resume 时严格校验，防跨协议续训 | **改协议/实现必须 bump 版本号** |
| `SOURCE_IDENTITY_FILES`（:54-64） | 9 个文件的 sha256 写进**每个 run** 的侧车 | 若新增 `models/model/*.py` 主干/模块文件，**要加进来**，否则身份链不完整 |
| `arch == "casa_tvim_str"` 分支（:273-343） | 模型构造 + optimizer 双 lr 组 + `_backbone_ref`（BACKBONE-ADAPT 审计） | 新主干要新增分支或参数化；**新主干没有 TinyViM 的 pretrained 键时，drift 审计口径要重新定义** |
| `_load_resume`（:434-461） | 恢复 model / optimizer / **`actual_steps`** / **torch+numpy+python+cuda RNG** | 崩溃恢复靠它；**它同时是"人工续训是否忠实"的机器保证** |
| `_write_casa_tvim_manifest`（:463+） | **resume 时对 `run_manifest.json` 逐字段严格比对** | ⇒ 手工续训命令行写错一个参数就会立刻中止（Run7 靠它验证过两次） |
| fold 等价检查（:884-911） | 真实数据 batch 上 train-graph vs deploy-graph 的 `max_abs` 与 0.5 二值 disagreement | 新算子（流形/几何）**必须**纳入这里 |

### 5.4 变体定义必须在三处同步（Run7 的 `check_mapping.py` 就是查这个）

`train_scripts/CASA-TViM/Run7/run_all.sh`（`variant_args()`）·
`models/run7_dry_run.py`（`VARIANTS`）· `train_scripts/CASA-TViM/Run7/check_mapping.py`。
Run8 新建目录后，这三处 + `SELECTION_LOCK.json` + `SOURCE_FILES.txt` 一起建。

### 5.5 新建一个 Run 的最小脚手架（**直接照抄 Run7**）

```bash
# 本地
cp -r train_scripts/CASA-TViM/Run7 train_scripts/CASA-TViM/Run8
# 然后逐项改：RUN=Run8、variant 名与 hf_edge→新开关、PROTOCOL、SOURCE_FILES.txt、SELECTION_LOCK.json
```
**不要从零写 runbook**：`Run7/README.md` §8/§9 与 `Run7/HANDOFF.md` 是打磨过的模板
（退出码表、五种 verdict 写法、恢复流程、可重入性、交付核对全部实测过）。

---

## 6. 服务器操作工具箱（本地 `.claude/`，**gitignored，但长期复用**）

### 6.1 核心工具

| 工具 | 用途 | 调用 |
|---|---|---|
| `_ssh.py` | 执行远端命令。**argv[1] 可以是"本地命令文件"路径**；argv[2] = 超时秒 | `python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash <scp>" 300` |
| `_r7_put2.py` | **单文件上传** + 双向 sha256 回显对比 | `python .claude/_r7_put2.py <本地相对路径> <远端相对路径>` |
| `_r7_stage.py` | **全量清单上传 + 对拍**：读 `SOURCE_FILES.txt` → 本地算 sha256 → 写 `SOURCE_SHA256.txt` → SFTP 上传 → 远端 `sha256sum -c` | `python .claude/_r7_stage.py`（上传+对拍）/ `--check`（只对拍） |
| `_download_tvim_logs_run7.py` | **带完整性门的日志下载器**（12 份齐全才返回 0） | `python .claude/_download_tvim_logs_run7.py` |
| `_r7_finalize.py` | **本地收口三步**：①下载并校验 12/12 → ②重建 `docs/experiment_metrics.xlsx`（**先备份 + 事后断言"没有行消失"**，丢行则 rc=3）→ ③生成 `docs/temporary/models_and_metrics_*.txt` 快照 | `python .claude/_r7_finalize.py`（`--allow-partial` 仅自测） |
| `_r7_watch2.ps1` | **12/12 完成看守**（每 10 min 探测，只在 12/12 时收工，**不写任何正式名产物**） | `& .\.claude\_r7_watch2.ps1`（后台 job） |
| `_r7_allstate.sh` | 12 run 状态一条命令（输出 `n_valid=/n_total=` + `ALL12_DONE`/`STILL_RUNNING`） | 远端执行 |
| `_r7_supervisor.sh` | **波次督导**（等波一过门 → 自动开波二 → 波三；pidfile 单实例守卫；25 min 停滞与 20 h 死线安全阀） | 远端 `setsid nohup bash ... &`（见 §6.3） |

### 6.2 完整命令序列（一轮的生命周期）

```powershell
# ── 0) 改完代码后：syntax check + 行尾检查 + hash
python -m py_compile models/model/<file>.py
python -c "import hashlib;b=open('models/model/<file>.py','rb').read();print(b.count(b'\r\n'), hashlib.sha256(b).hexdigest())"
#    ⇒ crlf 必须为 0（除非是 utils.py），否则远端 hash 对不上

# ── 1) 上传（两种粒度）
python .claude/_r7_put2.py models/model/<file>.py models/model/<file>.py      # 单文件
python .claude/_r7_stage.py                                                   # 全清单 + 对拍

# ── 2) 远端验收（务必用脚本文件，绝不在 ssh 里拼引号！见 §8.3 坑 #1）
python .claude/_r7_put2.py .claude/_r8_smoke.sh train_scripts/CASA-TViM/Run8/_r8_smoke.sh
python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash train_scripts/CASA-TViM/Run8/_r8_smoke.sh" 900

# ── 3) 启动（督导 + 波一；督导只等波一，不自己开波一！）
python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash train_scripts/CASA-TViM/Run8/_r8_launch_supervisor.sh" 300
python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash train_scripts/CASA-TViM/Run8/_r8_launch_wave1.sh" 300

# ── 4) 监控（本地后台看守 + 随时快照）
& .\.claude\_r7_watch2.ps1                       # 后台，完成后会通知
python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash train_scripts/CASA-TViM/Run8/_r8_progress.sh" 300

# ── 5) 12/12 之后：服务器收口链
python .claude/_ssh.py "cd /home/yqwang/projects/CASA-CD && bash train_scripts/CASA-TViM/Run8/finish_run8.sh" 1500

# ── 6) 本地归档三步（下载 + xlsx + 快照）
python .claude/_r7_finalize.py

# ── 7) 写结果节 + 提交推送
git add -A ; git commit -F .claude/_r8_commit_msg.txt ; git push origin main
```

### 6.3 启动/停止的正确姿势（Run7 血泪）

```bash
# 督导（远端脱离会话，必须重定向 stdin，否则 ssh 通道不关、调用超时）
setsid nohup bash .claude/_r8_supervisor.sh > /tmp/r8/sup.stdout 2>&1 < /dev/null &

# 停一个 campaign 的顺序（错一步就会自动重启）
pkill -f '<supervisor>.sh'      # 1) 先停督导（否则它会在 wait_wave 里重启）
pkill -f 'run_all\.sh'          # 2) 再停 run_all（否则它 3 次重试会拉起来）
# 3) 最后按数据集精确停 train.py（用锚定 pgrep，避免自我匹配）
for p in $(pgrep -f '^python train\.py'); do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q -- '--dataset CDD-CD-256' && kill -TERM $p
done
```

---

## 7. 收口 / 审计链（Run7 模板，Run8 应原样复用）

### 7.1 六个脚本

| 脚本 | 作用 |
|---|---|
| `run_all.sh --wave {1\|2\|3\|all}` | 自包含 runner：同波四库同一份 CLI，唯一变量只有开关；空目录预检（**非空 ckpt 目录需 `RESUME_SAME_RUN=1`**）；3 次重试自动从自身 `last.pth` 续训；内联 `check_test_block` 门 |
| `check_run.py` | **逐 run 硬门**（Run7：CTRL 32 项 / E9 45 项 / E10 46 项）：变体契约、机制证据、sidecar、deploy 参数、折叠门 |
| `run7_verdict.py` | 裁决引擎 + 机制归因（含 `--selftest`，Run7 为 **21 branches PASS**） |
| `independent_audit.py` | **独立最小复算审计**（不同代码路径，A1–A10；Run7 = 10 checks / 0 FAILED；`--selftest`） |
| `collect_results.py` | 判定每个 run 是否 VALID（只看**最后一个完整 TEST 区块** + `[ACTUAL-OPT-STEPS] 80000` + 协议串） |
| `finish_run7.sh` | **六步收口链**：门 →（刀锋证据）→ 逐 run 硬门 → 对象级复算 → 裁决 → 独立审计 → 跨变体表 |

### 7.2 退出码与 fail-closed 纪律（Run7 全部实测过）

| rc | 含义 | 动作 |
|---|---|---|
| 0 | 裁决已产出 | 进入写作 |
| 2 | 有 run 未产出完整 TEST 区块 | **等**；看训练日志，不看督导日志（波内静默） |
| 3 | 环境/脚本缺失 | 查 conda 与文件 |
| 4 | 逐 run 硬门有 FAIL | 查 `_RESULTS/check_run_*.json` 的 `failed`；**不得**用 `RUN7_ALLOW_CHECKRUN_FAIL=1` 蒙过 |
| 5 | 独立审计未全过 | **结论按 INVALID 处理** |

> 设计要点：**两道不同代码路径的门**。绕过第一道（硬门）时，第二道（独立审计）仍会抓到同一个篡改 ⇒ 最终仍是 rc=5。

### 7.3 结果归档工具（`analyse/`，本地跑）

| 工具 | 用途 |
|---|---|
| `extract_metrics_to_excel.py` | `outputs/**/train_log.txt` → **整表重建** `docs/experiment_metrics.xlsx`。⚠️ 会**静默跳过**没有 TEST 区块的日志 ⇒ 必须配"丢行断言"（`_r7_finalize.py` 已内置） |
| `models_to_txt.py` | 代码快照 + 指标 → `docs/temporary/models_and_metrics_<tag>_<run>.txt`（`--tag/--run` 只影响文件名） |
| `tvim_object_metrics.py` | 正确口径的**对象级**指标（GT 面积分组 Recall / ObjectHit / ObjP-ObjR-F1 / 边界带） |
| `run7_object_metrics.py` | Run7 三变体对象级只读复算（`--all-variants/--compare`；CPU 上拒绝运行） |
| `run7_fold_knife_edge.py` | 折叠门"刀锋像素"定位（真实 batch 复现 + 判 KNIFE_EDGE_ARTIFACT / NEEDS_REVIEW） |
| `analyse/tests/test_tvim_*.py` | **142 项单测**（对象指标、日志解析器逐字对拍、Run5/Run6 前测）：<br>`python -m unittest discover -s analyse/tests -p 'test_tvim_*.py'` |

---

## 8. 迭代方式（这是本项目的核心工作法，务必遵守）

### 8.1 一轮的标准节奏

1. **调研**（2024–2026 高水平工作；CCF-A 与 SCI 期刊分层要分清；只引用可核验主页/官方 GitHub）。
2. **代码级审计**：先读现有实现，画出**真实数据流**（不是概念图），指出梯度路径与压缩发生的位置。
3. **候选横评**：提 2–4 个候选，**只选一个主方案 + 必要对照**，并写出与既有工作的**实质差异**。
4. **写预注册**（见结果之前）：机制数学定义、数据流、逐文件修改清单、唯一变量、成功阈值、**失败判据**、
   smoke/dry-run 清单。冻结判据写进 `SELECTION_LOCK.json`（含冻结时间）。
5. **只读诊断/前测**（不需 80K）：先回答"这条信息是否存在""排序是否改变"。
   **纪律**：前测失败不能当作 80K 的替代判决，但**能决定值不值得投 80K**。
6. **smoke**：形状/参数量/FLOPs/epoch-0 逐位一致/梯度是否真的回流。
7. **dry-run**：真实数据 3–5 步（含增广同步、`gray>=128`、峰值显存）。
8. **fold 等价**：train↔deploy 0.5 二值 disagreement = 0。
9. **正式 80K**：同期 CTRL + 实验组，**唯一变量**，四库同预算同 seed。
10. **收口链**（§7）+ **独立审计**（不同代码路径）。
11. **复盘**：把结论（含**负结果**）写进根 `README.md` 结果节 + 该 Run 的 README + 预注册文档。
12. **归档**：xlsx + 快照 + 提交推送。**产物只改名（`_SUPERSEDED_*`/`_INVALID_*`），绝不删除。**

### 8.2 改代码的具体纪律

1. **只在本地改，再上传**；**绝不直接在服务器上编辑**（否则身份账本与实跑字节不一致）。
2. 任何改动涉及 `SOURCE_FILES.txt` 里的文件 ⇒ 重新算 sha256、更新 `SOURCE_SHA256.txt`、重跑
   `_r7_stage.py --check`，**必须全 OK**。
3. **不要在一次 campaign 中途改任何受哈希约束的文件**：`independent_audit.py` 的 **A8 要求 12 个 run
   代码身份唯一**（`len(unique_code_identity)==1`）。中途改文件会把 campaign 裂成两套身份，
   A8 会（正确地）判 FAIL。Run7 就是因为一个纯性能缺陷不得不**整轮重跑**（预注册 §7.7）。
   - 若必须改：**先停 campaign → 封存产物 → 改 → 重算账本 → 整轮重跑**。
4. **改模型必须 bump `PROTOCOL_VERSION`**（`models/train.py:51`）。
5. **新增的模型文件要加进 `SOURCE_IDENTITY_FILES`**。
6. 提交信息用 **`git commit -F <文件>`**；**不要**用 PowerShell `Set-Content -Encoding UTF8`（会加 BOM）。
7. 改完必须跑的验收（Run7 版，Run8 照做）：
   ```bash
   python -m py_compile <改动文件>
   # 服务器上：
   sha256sum -c train_scripts/CASA-TViM/Run8/SOURCE_SHA256.txt | grep -c ': OK'   # 期望 N/N
   python train_scripts/CASA-TViM/Run8/check_mapping.py ; python train_scripts/CASA-TViM/Run8/check_imports.py
   python train_scripts/CASA-TViM/Run8/run8_verdict.py --selftest
   python train_scripts/CASA-TViM/Run8/independent_audit.py --selftest
   cd models && CUDA_VISIBLE_DEVICES=0 python test_run8_<新模块>.py     # T0–Tn 全 PASS
   cd models && CUDA_VISIBLE_DEVICES=0 python run8_dry_run.py           # 真实数据 dry-run
   cd .. && python -m unittest discover -s analyse/tests -p 'test_tvim_*.py'   # 142 OK
   ```

### 8.3 我踩过的坑（**全部是实测踩过的，不是想象**）

| # | 坑 | 正确做法 |
|---|---|---|
| 1 | **PowerShell 会把内联远端命令的引号/`$()`/`\$?` 弄坏**（本项目反复发生，我在 Run7 收尾时又犯了两次） | **远端命令一律写进 `.sh` 文件 → 上传 → `bash <file>`**。绝不在 `_ssh.py` 的参数里拼复杂引号 |
| 2 | `pgrep -f`/`pkill -f` **自我匹配**（bash 自己的 cmdline 含该模式） | 用锚定模式 `^python train\.py`；在**脚本文件内**用普通模式（脚本 cmdline 不含被搜字符串） |
| 3 | 我写的正则 `levir_gua.rd\.sh` **匹配不到** `levir_guard.sh`，导致我误判"守卫进程不见了"（实际一直在跑） | 正则先在本地验一遍；或用 `pgrep -f '<完整子串>'` |
| 4 | `[ACTUAL-OPT-STEPS]` **每个 epoch 只打印一次** ⇒ 60 s 两次采样测步速会得到"0 步" | 用 `Epoch N/M ... t=Xs` 行测 epoch 耗时；或用跨 epoch 的累计差值 |
| 5 | 非空 ckpt 目录会被 `run_all.sh` 预检拦下；`RESUME_SAME_RUN=1` 是**续训同一 run** 的语义，**不是**"覆盖重跑" | 要重跑就先**改名封存**旧目录 |
| 6 | `extract_metrics_to_excel.py` **整表重建**且会静默跳过无 TEST 区块的日志 ⇒ 可能悄悄丢行 | 先备份 + 事后断言"没有行消失"（`_r7_finalize.py` 已内置，丢行 rc=3） |
| 7 | 把封存日志留在 `outputs/**/Run7/` 下会被 `extract_metrics_to_excel.py` 的 glob **当成额外 run** | 封存日志**移出 outputs 通配根**（Run7 移到了 `outputs/CASA-CD/CASA-TViM/_SUPERSEDED_Run7_*/`） |
| 8 | 督导日志在波内会**长时间静默**（它同步阻塞在 `run_all.sh` 里） | 判死活用 `kill -0 $(cat /tmp/run8_supervisor.pid)` + 子进程 + GPU 进程数，**不要**看日志 |
| 9 | GPU 显存：Run7 波三 LEVIR 在 GPU0 上 **连续 OOM 3 次被判失败**（该 run 自己就稳定滞留 ~10.8 GB，而 GPU0 上还有另一个重库 + 同用户别的项目占 11.4 GB） | 排卡留余量；开跑前记账；出 OOM 时**从自身 `last.pth` 续训**（`_load_resume` 会恢复步数与四项 RNG，`run_manifest.json` 逐字段校验保证你没改参数），必要时**换一张卡**（设备号不是冻结协议字段） |
| 10 | 服务器时钟随真实墙钟推进（我一度以为"几分钟前"，实际过了 2.7 h） | 每次探测**先 `date`**；ETA 用实测 `t=` 推 |
| 11 | `selective_scan` 系列 **import 失败是静默的大坑**（`mamba_ssm` pip 装了但 import 失败；`causal_conv1d` 缺失） | **写代码前先 import 探针**（本次已做，见 §4.2） |
| 12 | 只在 `tail -1` 上断言会漏掉关键行（例如 `[FAIL] ... attempts exhausted` 出现在中间、后面还有别的输出） | 断言用 `grep -c`/`grep -x` 精确匹配，不要只看最后一行 |

---

## 9. Run1–Run7 已证明 / 已否证（**避免第三次重做同一件事**）

| Run | 唯一变量 / 问题 | 结论 | 四库 F1（最好变体） |
|---|---|---|---|
| Run1 | CASA-TViM 主线：CAACP-SS2D + STR 落地 | 完成，成为新基线 | `M1_FULL` **97.18 / 91.07 / 83.47 / 95.07** |
| Run2 | LEVIR/SYSU 定向提升（CP-CAACP / FRH / 组合） | 无实质增益（最好 `E3` 97.18/91.29/82.85/94.98，SYSU 反降） | 97.18 / 91.29 / 83.13 / 95.18 |
| Run3 | SYSU 小目标定向（RA-CAACP / FS-TAR） | 无实质增益 | 97.13 / 91.18 / 83.24 / 95.02 |
| Diag1 | SYSU 小目标瓶颈分阶段**诊断**（D0–D5） | 定位到瓶颈性质（供 Run4–Run7 使用） | — |
| Run4 | R4-FET1（1/4 细尺度证据 1×1 可折叠旁路，8 runs） | **`MECHANISM-FAIL`** | — |
| Run5 | 原生高频证据复用前测 | **`NO_80K`**（前测否证；"edge map 救小目标"被数据否决） | 训练 0 次 |
| Run6 | 变化判别瓶颈可分辨性前测 | **`NO_80K`**（K4b：C2 只比 `baseline_full` 高 **+0.00801**，远低于要求的 +0.05 ⇒ 判别信息几乎全在模型自己的 head 置信度里） | 训练 0 次 |
| **Run7** | **HFAE：把 stage0/stage1 频率分解的高频分支接出来做边缘辅助监督** | **`MECHANISM-PASS`**（首次），winner `E9_HFAE_A`（`aux_only`，只加损失不注入）；**E10（加零初始化注入）不优于 E9** | CTRL 0.9717/0.9115/0.8268/0.9484 · **E9 0.9705/0.9134/0.8331/0.9507** · E10 0.9702/0.9123/0.8319/0.9518 |
| （更早归档线） | CASAA / STR-Fusion(Run2_TASS) / UltraLight(Run4–9) / Run11 | 均已终止；CASAA 的终止结论：**"deployable routing ranking 提升 ≠ F1 提升"** | 见 `docs/experiment_metrics.xlsx` |

**Run7 的三条可迁移经验**

1. **监督式辅助损失（把梯度压回表征）是有效的**；**把同一路证据"注入"主特征是无效的**（E10 vs E9）。
2. **增益很小且不均衡**：SYSU +0.0063（其小目标对象级召回 +0.0144）、WHU +0.0023；CDD −0.0012。
   四库**均未达**论文硬目标（98.00/92.50/85.00/95.00，仅 WHU 在 E9/E10 达 95.00+）。
3. **"改哪里"的判据已经收紧**：Run6 证明"信息已在 head 里"，Run5 证明"放大高频会优先放大假阳性"，
   Run7 证明"只有真正改变梯度流向的监督才有效" ⇒ **新机制必须先回答"它改变了哪条梯度路径"**。

---

## 10. 建议的 Run8 分阶段方案（**提案，待用户裁定；不要擅自开跑**）

| Stage | 内容 | 唯一变量 | 必要对照 | 出口判据（失败即停止，不靠加模块挽救） |
|---|---|---|---|---|
| **0** | 裁定 §2 的三个 P0；写 Run8 预注册 + `SELECTION_LOCK.json`；测服务器联网/编译 kernel/权重可得性 | — | — | P0-1 有明确答案；`import` 探针全绿 |
| **A** | 主干替换：VMamba-tiny（或其 ≤5M 可行替代）+ **原封不动的 CAACP/STR** | 主干 | 同期 `M1_R8CTRL`（新主干原机制） | 只回答"换主干值多少"；若 ≤5M 约束下主干不可行 ⇒ 回到用户 |
| **B** | 流形映射 + 细节/全局分解（**仍用原 QKV 形式**） | 几何 + 分解 | (i) 欧氏同式（同信息、无曲率）(ii) 固定半径双曲（**退化对照**） | **非退化性**：新几何必须改变重要性排序；且须优于"同式欧氏"才可谈曲率 |
| **C** | SAT 式 K/V 压缩（`k*C`）替换原 QKV，Q 保持 `N*C` | 压缩 | 原 QKV（不压缩） | 压缩率 vs F1 的权衡曲线；须交代与 **CASAA** 的差异 |
| **D** | 双分支各自特化 QKV | 特化 | 单分支共享 QKV | 特化是否真的带来分工（可诊断两分支的响应差异） |
| **E** | QKV 来源组合 / 注入点 / 高频损失 | 每项单独 | 同期 CTRL | 注入项必须交代与 **Run7 E10** 的差异；损失项沿用 `λ` 锁定不再搜索 |

**注意**：SSM 结构对 token 重排极其敏感（`CrossMerge` 依赖同一 `H×W` 网格；STORM ICML 2026 的结果）。
流形空间里的"压缩/分解/聚合"必须是**网格保持**的，或显式给出逆映射；**不要**引入任意 TopK/非规则聚类。

---

## 11. 待用户确认的问题清单（**新会话第一轮就把这些问清楚**）

| 编号 | 问题 |
|---|---|
| Q1 | **deploy ≤5M 是否解除？**（§2-P0-1 的 a/b/c 三选一）——这是全局前置 |
| Q2 | R2 的"每层"指哪些层？VMamba-tiny 的 4 个 stage 全部，还是只 stage3/4？（每层都做 ⇒ 训练图参数与显存会显著上升） |
| Q3 | R4 的注入目标是**跳层连接**还是**解码器**？（二者要分别做对照，还是只选一个？） |
| Q4 | R5 的几何选**双曲（Poincaré 球 / Lorentz）**还是**黎曼（SPD / Grassmann）**？两份调研文档都倾向"双曲 + 严格欧氏同式对照"，且都警告 SPD/Grassmann 在 2×2 cell 上样本太少 |
| Q5 | R6 的"细节/全局"分解用什么判据？（按频率带？按感受野？按注意力尺度？）——这决定是不是真的"分解"，还是只是"两个卷积头" |
| Q6 | R8 的 `k` 取多少？如何与 SAT 的 `c_ratio` 口径对齐（SAT 用 `c_ratio=0.5` 且是单图内容密度驱动）？ |
| Q7 | R9 的高频损失是否**沿用 Run7 的 HFAE（`λ=0.2` 锁定）**，还是要新设计？若沿用，注入部分与 E10 的差异是什么？ |
| Q8 | 本轮**是否仍要求四库同预算 80K + 同期 CTRL**？（调研文档 §11.2 明确要求"不使用历史 Run7 当同期对照"） |
| Q9 | 论文定位是否仍为"极轻量"？若是，(a) 选项会与之冲突，需要导师签字 |
| Q10 | 时间/算力预算：一轮 12×80K ≈ 13–15 h 墙钟（Run7 实测；含波次串行与收口）。本轮允许几个 Stage？ |

---

## 12. 名词与路径速查

| 名词 | 含义 |
|---|---|
| CAACP-SS2D | Change-Aware Asymmetric Context Pooling：把 SS2D 里 stage3 末个 TViM 的低频 2×2 AvgPool 换成双时相变化分数加权的 2×2 cell 聚合（A/B 共享权重、规则网格、β=0 精确继承预训练） |
| TAR / DCR | 多尺度双时相代数重参数化 / 可折叠解码器（训练多分支、部署折叠为单卷积） |
| HFAE | Run7 的高频边缘辅助监督（`str_hf_edge.py`）：HF tap → 边缘头 → aux loss（`aux_only`）/ 再加零初始化注入（`aux_inject`） |
| SAT | *Selective Aggregation Transformer for Image Super-Resolution*（CVPR 2026 Findings / arXiv:2604.07994）：**full Query + 压缩 K/V** |
| CASAA | 本项目历史的 SAT 式实现（`layers/casaa.py`），Run1/Run3 后终止 |
| GTRC | 调研文档 2 §6 提出的"跨时相局部测地关系形状变化" |
| fold / 折叠 | `switch_to_deploy()` 把训练多分支折叠为静态卷积；以 0.5 二值 disagreement=0 为门 |
| exact-80K | `--exact_max_steps 1`：第 80,000 次 `optimizer.step()` 严格停止，poly 分母固定为 `max_steps` |
| VALID | `collect_results.py` 的判定：**最后一个完整 TEST 区块**含 `[ACTUAL-OPT-STEPS] 80000` 与协议串 |

| 关键路径 | 值 |
|---|---|
| 服务器项目 | `/home/yqwang/projects/CASA-CD` |
| 督导 pidfile | `/tmp/run7_supervisor.pid`（Run8 换成 `/tmp/run8_supervisor.pid`） |
| Run7 结果 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_RESULTS/{summary,verdict,INDEPENDENT_AUDIT}.json` + `check_run_*.json`×12 |
| Run7 对象级 | `…/Run7/_OBJECT_METRICS/object_metrics_*.json`×12 |
| Run7 封存（**禁止删除**） | 日志 `…/CASA-TViM/_SUPERSEDED_Run7_20261011_2324_prehookfix/`；ckpt `…/checkpoints/…/Run7/*_SUPERSEDED_20261011_2324_prehookfix/` |
| 本地 xlsx | `docs/experiment_metrics.xlsx`（90 行 = Run1–Run7 全部正式 run） |
| 本地快照 | `docs/temporary/models_and_metrics_CASA-TViM_Run7.txt` |

---

## 13. 一句话交接

**Run7 已经证明"把梯度压回表征的监督"是本项目第一条真正有效的机制（幅度小但方向正确），
而"把证据注入主特征"无效。Run8 想在 VMamba-tiny 上用流形空间重新设计 CAACP 的压缩与 QKV——
方向可以谈，但有三件事必须先落定：`≤5M` 是否解除、VMamba 的 kernel/权重能否就位、
以及"流形映射不得退化成余弦的换皮"。请先读 §2 与两份调研文档，再动第一行代码。**
