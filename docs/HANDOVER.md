# CASA-CD 交接文档

> 生成于 2026-11-25，对应停训时的状态：`HEAD = 039fb90`（`origin/main` 已同步），CASA-MinorGoal / Run10 停在 15/24。
> 更细的历史与结论在 `README.md`（对外叙述）和 `.codex/STATE.md`（逐轮流水账）里，本文只讲「东西在哪、是干什么的、怎么用」。

---

## 0. 一页速览

- **做什么**：普通 RGB、全监督的遥感二值变化检测，四个库（LEVIR / SYSU / WHU / CDD），主打「~2 M 参数、CPU 级延迟」的轻量主干：MobileNetV2 四尺度 → TAR 时相融合 → 跨尺度压缩检索（完整 Q / 224 个 KV）→ DCR 由粗到细解码 → 四级深监督 + 128² 主输出。
- **当前代码口径**：`models/` 是 **CASA-MinorGoal / Run10**（协议 `casa_minor_run10_lean_v1`）。Run8 的 cd2、Run9 的换骨干、Run10 早期的层次原型都已经移出这棵树。
- **当前训练状态**：Run10 已停训，**15/24 完成、0 失败**。SYSU 12 项全齐，LEVIR 只出 `D_CTRL`×2 与 `D_CG`×1。
- **当前结论边界**：SYSU 上 TAR 竞争门 **+0.99 pp**（过预登记门 0.59，两 seed 同向）；检索层 **+0.13 pp**（噪声内，σ=0.39），减法版不掉点；DCR 的 128² 主输出**尚未兑现**（SYSU −0.01、LEVIR +0.11）。
- **入库范围**：源码、说明、指标表、源码快照进 git；数据集、权重、日志、`outputs/` 不进。

---

## 1. 根目录逐项

| 路径 | 作用 |
|---|---|
| `README.md` | 对外主文档：任务定义、四库协议、各轮结论、`docs/experiment_metrics.xlsx` 的读法。**「当前主线」一节跟着最新轮次改。** |
| `AGENTS.md` | 工作约定（项目规则）。改模型前先确认当前轮次、不覆盖在跑的源码、数据/权重/`outputs/` 不提交等。新接手先读这份。 |
| `models/` | 当前代码树（见 §2）。训练、测试、推理的唯一来源。 |
| `models_v1/` | 早期（CASA / CASA-TViM 时代）的代码树，只作历史参考，不再训练。 |
| `train_scripts/` | 逐轮实验套件（见 §3）。**每一轮的 `pipeline.py` 就是那一轮的调度器**，不是全局服务。 |
| `tools/` | 通用分析工具（见 §4）：下载日志、写指标表、生成源码快照、算 FLOPs、对象级指标。 |
| `docs/` | 文档、指标表、PPT、快照、文献（见 §6）。 |
| `outputs/` | **不进 git**：本地日志镜像、分析产物、图片、PPT 渲染、远程命令脚本（见 §5）。 |
| `others/` | 第三方/参考实现（BoltzFormer、CoSA、ERMoE、HypMix 等），只读参考。 |
| `.codex/` | 项目工具与上下文：SSH 连接、状态查询、服务器说明、逐轮流水账（见 §7.1）。 |
| `.vscode/sftp.json` | **服务器凭据（host/port/user/password），已在 `.gitignore` 里，永不提交。** |
| `.tmp/`、`.deps/`、`.agent-teams/`、`.dsh-matlab-runner/` | 本地临时物，全部被忽略。 |
| `.gitattributes` / `.gitignore` | 行尾归一（LF）与忽略规则（`*.pth`、`outputs/`、`.vscode/`、`.tmp/` 等）。 |
| `README_MoE.md` | MoE 探索线的独立说明（`train_scripts/CASA-MoE/`），与当前主线无关。 |

---

## 2. `models/` —— 当前代码树

```
models/
  train_CASA.py            训练/验证/测试入口（唯一 main）
  requirements.txt         依赖（torch、openpyxl…；服务器用 conda env casacd）
  README.md                本树口径、开关表、参数速查表
  SOURCES.md               每个文件的来源与演进（哪一轮引入、为什么）
  HYPERBOLIC_DESIGN.md     双曲几何部分的设计笔记（打分几何、数值稳定）
  operating_point.py       阈值/工作点选择（val 校准）
  profile_CASA.py          FLOPs / 参数量 / 延迟剖析
  dataset/
    dataset.py             CD 数据集：A/B/label 读取、同步裁剪翻转旋转、时相交换
    casa_transforms.py     光度增强（Run10 之前引入，默认关闭）
  model/
    casa_net.py            CASANet：编码器 → TAR → 检索 → DCR → 头；FineRefine（128² 主输出）；switch_to_deploy
    mobilenet_v2.py        主干（ImageNet 预训练，四尺度 24/32/96/320）
    str_tar.py             TAR：RepLocalBlock、MultiScaleTAR、ChangeGateTARStage(cg)、SPP 变体、DifferenceTemporal1x1
    str_dcr.py             DCR 由粗到细解码（reparam / plain 两种融合），四级输出
    str_reparam.py         可折叠重参数块（1×1/3×3 分支折叠成单卷积）
    edge_supervision.py    Sobel 边缘图 + 边缘头 + 边缘损失（默认关，`--edge-weight` 打开）
    metric_tool.py         F1/IoU/Kappa/OA 等指标
    layers/
      __init__.py
      pyramid_context.py     跨尺度上下文基类（位置/尺度嵌入、channel_norm）
      hyperbolic_context.py  双曲检索：Lorentz 提升、测地距离打分、support 先验、去均值 + 门控读出、尺寸先验
      euclidean_context.py   欧氏孪生（同参数形状，只换距离；对照臂 `--module matched`）
      difference_context.py  KV 取自 |P−Q| 的差分原型（`--module difference`）
      manifold_context.py    在流形上池化 K/V + 独立 support 头 + level 通路（`--module manifold`）
      class_pooling.py       类条件池化：每格 2 类原型、support(mass/cell/pixel/confidence)
      lorentz_ops.py         测地距离/重心等数值算子（含小 z 的级数展开）
  test_*.py                模型侧单测（见 §8），每个都能独立 `python models/test_xxx.py`
  analyze_*.py / check_*.py 一次性诊断脚本（留档，不参与训练）
```

**当前主干开关**（`train_CASA.py --help` 为准）：
`--module none|hyperbolic|matched|difference|manifold`、`--tar-variant reparam|plain|spp|spp1|diff|cg`、
`--decoder-variant reparam|plain|none`、`--support mass|cell|pixel|confidence|none`、`--ds-weights 1,.5,.25,.125`、
`--retrieval-scales 0,1,2,3`、`--compact-prototypes`、`--prototype-pooling euclidean|lorentz`、`--aggregate lorentz|euclidean`。

**参数（Run10 口径，train/deploy）**：`D_CTRL` 2,042,324 / 1,934,019；`D_CG` 2,072,276 / 2,025,667；检索臂 2,090,099 / 2,043,490。

---

## 3. `train_scripts/` —— 逐轮实验套件

```
train_scripts/
  CASA-Pyramid/     Run1 … Run16   （早期「双曲 H / 差分检索 / DCR」路线）
  CASA-MinorGoal/   Run1 … Run10   （当前路线：SPP → 差分 KV → 边缘 → 流形 → cd2 → 换骨干 → 精简主干）
  CASA/             Run6, Run7     （更早的 CASA 主线）
  CASA-MoE/         Run1 … Run3    （MoE 探索，独立线）
```

一个轮次目录里固定是这些文件（以 `train_scripts/CASA-MinorGoal/Run10/` 为例）：

| 文件 | 职责 |
|---|---|
| `config.py` | **单一事实来源**：协议名、臂定义、seed、数据集顺序、步数、bs、显存预留、门限、`model_options(arm)` 拼的 CLI |
| `PROTOCOL.md` | 人读的协议：本轮的假设、臂、门限、预登记判据 |
| `validate.py` | 静态审计：矩阵形状、CLI 开关、模型构造、折叠一致性、梯度可达性 |
| `prepare.py` | 上机前检查：12 臂 smoke（前向/折叠/参数计数），写 `params.json`、`scheduling.json`、`readiness.json` |
| `bootstrap.py` | 把 `models/` + `train_scripts/.../RunN/` 打包上传到服务器 `runtime/`（**本机改动后必须重冻**） |
| `pipeline.py` | 调度器：读 `scheduling.json`，按显存预留把任务派到 GPU 槽，写 `pipeline_status.json`，支持 `--resume` |
| `run_one.py` | 单个 (arm, dataset) 的实际启动：拼 `train_CASA.py` 命令行 |
| `collect_results.py` | 轮次级汇总（拉 TEST 区块 → 表格） |
| `compare.py` | 臂间对比 + 预登记裁决（`comparison.md/json`） |
| `benchmark.py` | 延迟/显存基准（deploy 形态，bs1） |
| `environment.sh` / `environment_probe.py` | 服务器环境激活与自检 |

历史轮次里 `CASA-Pyramid/Run15|16` 还引用过 `models_Run1/4/10`——那三棵树已删除，需要时从 git 历史 `a3e839d` 取回。

---

## 4. `tools/` —— 通用工具

| 文件 | 用途 |
|---|---|
| `download_training_results.py` | `--tag --run` 递归下载服务器日志/JSON 到 `outputs/<tag>/<run>/` |
| `extract_metrics_to_excel.py` | 扫 `outputs/**/train_log.txt` 的**最后一个完整** TEST 区块 → 重建 `docs/experiment_metrics.xlsx`（缺行会拒绝覆盖） |
| `models_to_txt.py` | 生成「源码 + 指标」txt 快照；`--out` 指定路径，`--metrics placeholder` 出空表（开跑时用） |
| `param_breakdown.py` | 参数量/FLOPs 按模块分解 |
| `benchmark_casa.py` | 端到端延迟基准 |
| `tvim_*` / `vit_pretrain_audit.py` | CASA-TViM 时代的诊断与权重审计，现基本只读档 |
| `tests/` | 通用工具单测（`python -m unittest discover -s tools/tests`；注意 script 风格测试要直接 `python <file>.py`） |
| `archive/` | 按轮次命名的旧脚本（`run1_*`…`run11_*`），**依赖当时的布局，多半跑不通，只备查** |

---

## 5. `outputs/`（不进 git）

```
outputs/
  CASA-MinorGoal/Run1 … Run10/   本地日志镜像：<arm>/<dataset>/{train_log.txt, result.json, checkpoint…}
  CASA-Pyramid/, CASA/, CASA-MoE/ 同上，按 tag 分
  CASA-MinorGoal/Run7/analysis/  tta_and_threshold.md 等分析产物
  pptx/                          PPT 构建脚本 + 渲染 PNG（build_*_deck.py、export_png.ps1、render_*/）
  remote_cmds/                   一次性远程脚本：launch_remote.py、push_file.py、*_probe.py、precompute_teacher.py …
  tooling_archive/, tooling_migration/  旧工具归档（zip）
  tmp_*.py, *.log                历史临时脚本与日志（可忽略）
```

> `outputs/` 里的 `tmp_*`、`*_watch.log` 都是当轮次的一次性产物，接手时**不要**把它们当 API。

---

## 6. `docs/`

| 路径 | 作用 |
|---|---|
| `experiment_metrics.xlsx` | **正式指标总表**（sheet `metrics`，429 行 × 14 列）。一行 = 一个 (tag, run, arm, dataset, seed) 的正式 TEST。跨轮比较只看这里，不看日志。 |
| `CASA_总体模型结构.pptx` | 顶层结构图（编码器一栏仍写 MobileNetV2，历史遗留）。 |
| `temporary/` | 逐轮交付快照：`models_and_metrics_<tag>_<Run>/`（txt 源码+指标、`README.md`、`comparison.*`、用户自维护的 pptx）以及 2026-11-17 的一批设计/诊断 md。 |
| `temporary/过去的想法/` | 更早的设计草稿。 |
| `experiments/` | 按 tag 的实验记录（CASA-Pyramid / CASA-MoE），含曲线与阶段结论。 |
| `simulation/` | 双曲几何的数值仿真（`hyperbolic/`）。 |
| `参考文献/` | 文献 PDF 与索引（体积大，未全部入库）。 |
| `RSML-3_服务器环境与变化检测数据统一说明.md` | 服务器环境与四库数据格式的长文说明。 |

---

## 7. 服务器与 GitHub

### 7.1 服务器（RSML-3）

| 用途 | 路径 |
|---|---|
| 项目根 | `/home/yqwang/projects/CASA-CD` |
| 数据集 | `/share_datasets/CD/<dataset>/{A,B,label,list}/`（LEVIR/SYSU/WHU/CDD 的 `*-CD-256`） |
| 运行输出 | `/home/yqwang/outputs/CASA-CD/<tag>/<Run>/<arm>/<dataset>/` |
| checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/<tag>/<Run>/` |
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight/` |
| 冻结 runtime | `/home/yqwang/outputs/CASA-CD/<tag>/<Run>/runtime/`（`bootstrap.py` 上传，训练实际跑这份） |

**连接**：`python .codex/ssh.py "<命令>" [超时]`。凭据只从 `.vscode/sftp.json` 读（**不提交、不要贴进文档/日志**）。`paramiko` 已在依赖里。

**环境**：非登录 shell 里**没有** `python`，只有 `/usr/bin/python3`（无 torch）。任何远程 Python 都要先：

```bash
source /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime/train_scripts/CASA-MinorGoal/Run10/environment.sh
```

（内含 `conda activate casacd`、`PYTHONPATH=.../envs/mamba_base/...`、`LD_LIBRARY_PATH=...`。）

**GPU 约定**：两张 RTX 5090（各 32 GB）。GPU0 常有他人任务，本项目默认只用 **GPU1**、两槽（`D_*` 9,800 MiB / `R_*` 13,600 MiB）。

**已知损坏**：2026-11-24 起 **sftp 子系统坏**（任何路径都 `FileNotFoundError`）。传文件改用：

```bash
REMOTE_TARGET=/home/yqwang/... python outputs/remote_cmds/push_file.py <本地文件>
```

目标路径**必须**用环境变量传——MSYS 会把命令行里的 `/home/...` 改写成 `C:/Users/...`。

**其它工具**：`python .codex/status.py --tag CASA-MinorGoal --run Run10`（队列/GPU 只读状态）；后台启动一律 `python outputs/remote_cmds/launch_remote.py "<命令>" <相对日志名>`。

### 7.2 GitHub

| 项 | 值 |
|---|---|
| remote | `https://github.com/YuqiWang-code/CASA-CD.git`（`origin`） |
| 分支 | `main`（无 PR 流程，直接提交推送） |
| 当前 HEAD | `039fb90` |
| 凭据 | 必须显式指定 helper，否则弹 GUI 卡死：`GIT_TERMINAL_PROMPT=0 git -c credential.helper='!"/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"' push` |
| 入库 | 源码、`README`/`AGENTS`、`docs/experiment_metrics.xlsx`、`docs/temporary/**` 快照 |
| 不入库 | `*.pth`/`*.pt`、`outputs/`、`.vscode/`、`.tmp/`、`__pycache__/`、数据集 |
| 提交粒度 | 一轮一提交（源码 + 指标表 + 快照 + 文档）；指标与源码同笔提交，便于回溯 |

---

## 8. 常用操作清单

```bash
# 1) 改模型后：先跑模型单测（九个，全部能独立运行）
for t in test_CASA test_run7_modules test_run4_modules test_ablation_variants \
         test_pyramid_context test_hyperbolic_context test_minor_goal \
         test_edge_supervision test_context_geometry; do python models/$t.py; done

# 2) 重冻 runtime（改了 models/ 就必须做；先删远端 readiness.json）
python train_scripts/CASA-MinorGoal/Run10/bootstrap.py

# 3) 上机前体检（12 臂 smoke + 折叠 + 写 scheduling.json）
python outputs/remote_cmds/launch_remote.py "cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime/train_scripts/CASA-MinorGoal/Run10 && source environment.sh && export CUDA_VISIBLE_DEVICES=1 MINOR_GPUS=1 MINOR_SLOTS=2 && python prepare.py" run10_prepare.log

# 4) 起队列 / 续跑
python outputs/remote_cmds/launch_remote.py "cd .../Run10 && source environment.sh && export CUDA_VISIBLE_DEVICES=1 MINOR_GPUS=1 MINOR_SLOTS=2 && python pipeline.py --resume" run10_queue.log

# 5) 收结果 → 指标表 → 快照
python tools/download_training_results.py --tag CASA-MinorGoal --run Run10
python tools/extract_metrics_to_excel.py
python tools/models_to_txt.py --tag CASA-MinorGoal --run Run10 \
  --out docs/temporary/models_and_metrics_CASA-MinorGoal_Run10/models_and_metrics.txt
python train_scripts/CASA-MinorGoal/Run10/compare.py --root outputs/CASA-MinorGoal/Run10 \
  --markdown outputs/CASA-MinorGoal/Run10/comparison.md --json outputs/CASA-MinorGoal/Run10/comparison.json
```

---

## 9. 踩过的坑（按被坑次数排序）

1. **改了 `models/` 没重冻 runtime** → 远端跑的还是旧代码（`TypeError: unexpected keyword argument`）。改完必须 `bootstrap.py`；`readiness.json` 存在时 `bootstrap.py` 会拒跑，先删它。
2. **MSYS 改写以 `/` 开头的参数** → `/tmp/x` 变 `C:\Users\...`、远程路径变 `C:/...`。日志名给相对名，远程目标用环境变量 `REMOTE_TARGET`。
3. **`gamma_context` / `fine.out` 零初始化** → 第 0 步相关参数梯度精确为 0；测"新模块改变了输出"前必须先把门 `fill_(0.5)`，否则测到的是恒等。
4. **`pkill` 模式匹配到自己的 shell** → 用 `pgrep -f 'pipeline[.]py'`、`'runtime/models/train_CASA[.]py'` 这种带方括号的写法。
5. **`scheduling.json` 是 runtime 冻出来的** → 改了 reservation/槽数，必须删 `readiness.json` → 重冻 → 重新 `prepare.py`，否则排队沿用旧值。
6. **BatchNorm 折叠不是"随便折"** → 128² 上有死通道（`running_var` 到 1e-45），折 BN 会把 FP32 舍入放大成 4.8e-2 的概率差。`FineRefine` 因此保留两个 BN（+128 参数）。
7. **参数量表别用全局正则改** → 会把 `(2, 6, 256, 256)` 这类元组一起改掉，直接语法错误。整块替换字典。
8. **`python -m unittest` 对 script 风格测试会报 NO TESTS RAN** → 直接 `python models/test_xxx.py`。
9. **PowerShell `Set-Content -Encoding utf8` 写 BOM** → 远端 `source environment.sh` 报 `conda.sh\r: No such file`；打包前去掉 BOM、CRLF→LF。
10. **中文/绝对路径进 sftp** → 已改用 `push_file.py`（base64 走 exec）。

---

## 10. 未结事项（接手后的第一件事）

1. **Run10 剩 9 项**（LEVIR 的 `D_CG-2335` + 四个 `R_*` 臂 × 2 seed）未跑；停训点在 15/24。要续跑就按 §8 第 4 步 `--resume`。
2. **HRA 的判决缺两块证据**：①LEVIR 两 seed；②「HRA 挂在不带竞争门的 TAR 上」这一臂（否则无法区分"HRA 无效"与"HRA 与竞争门冗余"）。
3. **DCR 128² 主输出未兑现**：`D_CTRL` SYSU 81.38 ≈ Run7 的 81.39，LEVIR +0.11；需要单独看这一项（例如 128² 头是否收敛太慢）。
4. **指标表与快照未回填**：Run10 的 15 项还没进 `docs/experiment_metrics.xlsx`，`docs/temporary/models_and_metrics_CASA-MinorGoal_Run10/models_and_metrics.txt` 的指标列仍为空表。
