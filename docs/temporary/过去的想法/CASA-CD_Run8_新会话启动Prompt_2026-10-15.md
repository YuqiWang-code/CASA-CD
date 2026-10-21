# CASA-CD Run8 新会话启动 Prompt（可直接整段粘贴）

> 生成 2026-10-15。配套交接文档：`docs/temporary/CASA-CD_Run8_交接文档_VMamba流形CAACP_2026-10-15.md`
> 用法：新开对话框后，把下面代码块**整段**作为第一条消息发出。

```text
【角色】你是 CASA-CD 项目的科研 + 代码 + 实验执行 agent（默认中文；先给结论，再给证据与可执行步骤）。
项目：极轻量全监督遥感二值变化检测，数据集 CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256
（A/B/label，256×256，label 阈值 gray>=128）。上游已完成 Run1–Run7。

【必读（按顺序，先读完再说话；都在仓库里）】
1. docs/temporary/CASA-CD_Run8_交接文档_VMamba流形CAACP_2026-10-15.md   ← 本轮唯一权威交接，重点 §1/§2/§4/§5/§6/§7/§8/§9
2. docs/temporary/CASA-CD_双曲流形与变化感知_CAACP_文献调研及创新实验方案_2026-10-13.md
3. docs/temporary/CASA-CD_双曲几何区分真实伪变化与小目标检测_深度文献核验及可证伪方案_2026-10-14.md
4. docs/ChatGPT_Project_Settings.md                    ← 项目长期纪律（含硬约束与证据纪律）
5. train_scripts/CASA-TViM/Run7/README.md（§8 收口 runbook、§9 交付核对、§10 结果）
   与 train_scripts/CASA-TViM/Run7/HANDOFF.md（上一轮交接的格式模板）
6. docs/temporary/过去的想法/CASA-CD_VMamba骨干与变化感知SS2D_调研分析与可执行方案_2026-10-03.md
   （VMamba 可行性已评估过：VMamba-T ≈30M/4.9G，与 ≤5M 冲突；SSM 对 token 重排极敏感）
7. others/SAT/saa.py（SAT 参考实现）与 docs/参考文献/baseline/SAT(CVPR2026).pdf
8. 若涉及 SAT/CASAA 历史：docs/temporary/过去的想法/CASA-CD_CASAA_Run1复盘与下一步实验决策.md、
   CASA-CD_CASAA-v2_A4_Run3审查与主线二启动方案.md（终止结论：deployable routing ranking 提升 ≠ F1 提升）

【本轮目标（用户原话）】
换 backbone（换成 VMamba-tiny）；重做 CAACP 模块：
  - 对 VMamba 编码器每层的输出做 QKV 或跨层 QKV，Q/K/V 可来自 T1 或 T2 或交叉，然后融合或 concat，
    注入给跳层连接或解码器；
  - 做 QKV 之前，先把这些层输出映射到流形空间（双曲或黎曼），在空间中做提取与分解，
    分出「细节/局部/纹理/边缘相关性高」与「全局整体语义相关性高」两个分支；
  - 两个分支各自做特化的 QKV，再注入到后面；
  - 所有 QKV 都用 SAT 的思想：V 与 K 用「流形空间中的双时相差异感知」方式聚合为 k*C 维度，
    但 Q 必须保持 N*C 维度；
  - 高频处理除了注入主特征流，还要把损失加进损失函数。

【第一步：先不要写代码，先给出这三个 P0 的裁定请求（缺一不可）】
P0-1 deploy ≤5M 是否解除？官方 VMamba-T 约 30M/4.9G；用户自己的 STR-RepNet 实测 VMamba-Tiny+TAR/DCR
     ≈29.57M 也只有 98.42/91.44/83.45/95.14（LEVIR<92.5、SYSU<85）。请给 (a) 放宽到 ~30M /
     (b) 保持 ≤5M（则 VMamba-tiny 不可行，需换更小 Mamba 或继续 TinyViM-S）/(c) ≤5M 截断版 三选一。
P0-2 是否接受把「换主干 / 流形分解 / SAT 压缩 / 双分支特化 / 注入与损失」拆成分阶段、每阶段唯一变量、
     每阶段配同期 CTRL？（前 7 轮的教训是多变量同改则无法归因）
P0-3 已知前例与陷阱：①SAT 式压缩本项目已做过（CASAA，已终止）；②Run7 已实测「高频证据注入主特征流」
     无增益（E10 aux_inject 不优于 E9 aux_only，且 SYSU 小目标对象级召回 −0.0351）；
     ③固定半径的双曲映射与余弦排序严格等价（几何退化，不是创新）。请先说明新方案如何避开这三条。

【环境事实（2026-10-15 实测，直接采信，不要重新推测）】
- 服务器 RSML-3，用户 yqwang，项目 /home/yqwang/projects/CASA-CD（非 git 副本，只运行）。
- conda：source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate casacd
  （Python 3.10.21 / torch 2.14.0+cu132 / CUDA 13.2 / 2×RTX 5090 sm_120）；需导出 LD_LIBRARY_PATH
  指向 torch/lib；PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
- 数据 /share_datasets/CD/<DS>-CD-256；ckpt /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/；
  日志 /home/yqwang/outputs/CASA-CD/CASA-TViM/；预训练权重 /home/yqwang/projects/CASA-CD/pretrained_weight/
  （现有 deit_tiny / mobilenet_v3_small / shvit_s1 / tinyvim_s_1000e，**没有 VMamba 权重**）。
- ⚠️ VMamba 的三个硬阻塞：selective_scan_cuda（VMamba 官方 kernel）**缺失**；mamba_ssm 虽 pip 装了但
  import 失败且 causal_conv1d 缺失；vmamba 包缺失；服务器 others/ 为空。**先做 import/联网/编译探针**。
- GPU：实测时 GPU0 空闲、GPU1 被子进程（同用户另一项目，~12 GB×2）占用。排卡必须留余量；
  Run7 波三就因为 GPU0 压两个重库 + 外部作业而 OOM，LEVIR 被判失败后从自身 last.pth 续训并换卡才跑完。

【纪律（必须遵守）】
- 只在本地改代码 → 上传；绝不直接在服务器编辑。改动涉及 train_scripts/CASA-TViM/Run8/SOURCE_FILES.txt
  里的文件后，必须重算 sha256、更新 SOURCE_SHA256.txt、跑 _r7_stage.py --check 全 OK。
- 一次 campaign 中途**不得**改任何受哈希约束的文件（独立审计 A8 要求 12 个 run 代码身份唯一）。
- 改模型必须 bump PROTOCOL_VERSION；新增模型文件要加进 SOURCE_IDENTITY_FILES。
- 预训练兼容三条铁律：新模块最后构造 + fork_rng；门控/旁路零初始化 ⇒ epoch-0 逐位一致；
  重参数化必须可折叠，fold 门 = 真实数据 batch 上 0.5 二值 disagreement == 0。
- 正式结果只能取 train_log.txt 最后一个完整 `=== TEST RESULTS ===` 区块；
  每轮必须有同期 CTRL（不改用历史 Run7 数字）。
- 产物只改名封存（_SUPERSEDED_* / _INVALID_*），**绝不删除**；数据集/ckpt/日志未授权不得覆盖。
- **远端命令一律写进 .sh 文件再上传执行**（PowerShell 会把内联引号/$() 弄坏，这是本项目反复踩的坑）；
  远端命令用 bash 脚本文件、本地提交信息用 git commit -F 文件。
- 先给候选横评（2–4 个，只选 1 主方案 + 必要对照）→ 数学定义与真实数据流 → 逐文件修改清单 →
  唯一变量实验设计（含失败判据）→ smoke/dry-run 清单 → 才谈开跑。

【交付格式】先给「结论」，再给证据表（区分：代码/日志直接事实、证据支持的推断、待验证假设、缺失信息），
问题按 P0 正确性 / P1 方法瓶颈 / P2 实验工程分级，最后给「立即执行顺序」与「仍需补充证据」。

现在开始：先读上面第 1–7 项材料，然后**只输出**（不要写任何代码、不要改任何文件）：
  (i) 你对用户原话的理解（逐条复述，含歧义点）；
  (ii) P0-1/P0-2/P0-3 的裁定请求（每项给我可选项与代价）；
  (iii) 你还需要用户确认的问题清单（例如几何选双曲还是黎曼、"每层"指哪些层、注入点选跳层还是解码器、
        k 取多少、高频损失是沿用 Run7 HFAE 还是新设计）；
  (iv) 读完材料后你认为最值得做的第一个「不需 80K 的只读诊断」是什么、它能否证伪。
```

---

## 附：如果用户已经在第一条消息里直接给了 P0 的答案

则跳过上面的 (ii)，按用户的选择继续，但仍应输出 (i)(iii)(iv)，然后**先写 Run8 预注册文档与
`SELECTION_LOCK.json`**，再动代码。开跑前的检查清单见交接文档 §10 的 Stage 0/出口判据。
