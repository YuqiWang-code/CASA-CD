# CASA-CD｜SYSU 微小变化瓶颈分阶段诊断（Run-Diag）——交给本地 DSH 的完整执行规范

> **性质：诊断任务，不是 Run4 新结构训练方案。** 目标是在保留 CASA-TViM-STRNet 主线和所有已完成实验的前提下，用可复现的证据回答：**微小真实变化的证据究竟是提取不足、在 TinyViM 中途被弱化、在 CAACP 聚合时被抑制，还是在 TAR/DCR/预测头里未被有效利用？**
>
> 日期：2026-10-10；基准仓库：[YuqiWang-code/CASA-CD](https://github.com/YuqiWang-code/CASA-CD)；撰写时核对的 `main` commit：`fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f`。DSH 执行时先确认 HEAD；如有更新，比较模型/诊断相关文件的增量，以实际代码为准并记录偏差，不得假装版本相同。
>
> **执行顺序：P0 → D0 → D1 → D2 → D3 → D4（条件触发）→ 诊断报告 → 决策。** 本轮不训练新的 CASA-CD 模型，不更改 loss/增强/阈值/训练 seed，不增加推理期模块，不覆盖旧 checkpoint/日志/已有 `docs/temporary/*.json`。
>
> **交付的不是“口头分析”，而是已实现的只读诊断工具、测试、真实 SYSU 诊断结果、可视化图、归因报告及后续唯一主方案的推荐。** 如果服务器或 checkpoint 无法访问，必须停在可执行代码与本地 smoke 阶段，明确未实测，禁止杜撰指标。

---

## 0. 给 DSH 的总指令（优先读）

你现在是 CASA-CD 项目的科研代码审计及诊断执行代理。本次唯一任务：**校准旧诊断的评价口径，定位 TinyViM-S-Slim + CAACP-SS2D + TAR/DCR 对 SYSU 小变化的首次信息衰减位置，并给出有证据的下一轮方向选择；不要先发明新模块。**

1. 完整阅读：`README.md`、`models/model/casa_tvim_str_net.py`、`models/model/tinyvim_s_slim.py`、`models/model/layers/ss2d.py`、`models/model/layers/caacp_ss2d.py`、`models/model/str_tar.py`、`models/model/str_dcr.py`、`models/train.py`、`models/eval.py`、`models/dataset/{dataset.py,Transforms.py}`、`models/model/metric_tool.py`、`analyse/run2_zero_cost_diag.py`、`analyse/run3_report.py`、`docs/temporary/{run2_zero_cost_diag.json,run3_z12_summary.md,run3_report.md}`、`docs/temporary/CASA-TViM_Run1/metrics_tables.md`、`train_scripts/CASA-TViM/Run{1,2,3}/README.md`。检查真实服务器目录是否与文档一致。
2. 新诊断尽量**仅增文件，不触碰主训练图**。如确需加 hook/forward 可观测接口，只能加默认关闭的诊断接口，并证明 `diag=off` 时与原模型输出完全一致。旧的 `run2_zero_cost_diag.py` 可以通过新增 `v2` 文件替代，**不得覆盖旧 JSON**。
3. 全部结果使用固定 `seed=16`、256×256 输入、原有 test split、固定二值判定 `p>0.5`（与 `models/train.py` 一致）、`gray>=128`，不按 test 集调阈值。诊断用轻量探针如需拟合，**只能用 train 数据拟合，test 只计算固定指标**；探针不属于新 CASA-CD 训练运行，也不作为正式模型指标。
4. **每一个归因必须区分**：`[F] 直接代码/日志/计算事实`、`[I] 证据支持的推断`、`[H] 待验证假设`、`[M] 关键证据缺失`。不得把相关性写成因果，不得把运行时 β 消融等同于“从头训练的消融”。
5. 输出 `PASS/WARN/FAIL/BLOCKED` 的逐阶段 Gate；任一 P0 Gate FAIL，禁止进入需要该 Gate 支持的归因和结构方案预注册；不得静默跳过失败项。
6. 所有诊断结果都必须能用固定 CLI 复现，包括代码 SHA、checkpoint SHA256、数据集 manifest/list SHA256、软硬件版本、诊断参数与用时。**执行过程中不修改历史 checkpoint、原始 train_log 或数据集文件。**
7. 所有分析必须以**当前主线 CASA-TViM** 为对象，不能混入旧的 ChangeViT/CASAA/UltraLight/STR-Fusion 结论。对照优先使用 CASA-TViM Run1 A2/M1 和 Run2/Run3 E1–E5。

---

## 1. 研究问题、当前事实与证据缺口

### 1.1 当前主线（依据 GitHub 2026-10-09 快照的代码事实）

```text
A/B 图像 [B,3,256,256]，拼 batch→ [2B,3,256,256]
  │ TinyViM-S-Slim，共享权重
  ├─ stem + Stage1 → F1：[2B,48,64,64]
  ├─ Stage2           → F2：[2B,64,32,32]
  ├─ Stage3 Prefix    → Xpre：[2B,168,16,16]
  │    └─ A/B 特征 cosine + rank→ 两时相共享 score（no_grad）
  ├─ Stage3 末 TViM 内 CAACP：16² 的低频 2×2 cell→8² scan
  │    ├─ c = c_avg + beta*(c_ca - c_avg)，beta 学习、初始化 0
  │    └─ 高频 dense/local 路径保留（不能写成整个 16² 被删除）
  ├─ Stage3 输出      → F3：[2B,168,16,16]
  └─ Stage4 slim      → F4：[2B,224,8,8]

每时相 F1/F2/F3/F4 → 四尺度 TAR（96C）→ DCR
  → 64×64 预测头（FRH=1 时另有 128×128 头）
  → bilinear 256×256 → sigmoid → P(change)
```

**重要精确性**：`TinyViMSlim.forward_stem_s2_prefix()` 的代码名称含 `s2`，但 `network[4]` 实际对应 index=2 的 Stage3；需以 Python 对象和 tensor shape 为准，避免“stage0/stage1/stage2”代码索引与论文 Stage1/2/3 的文字混淆。

`SS2D` 的低频池化参数是 `AvgPool2d(2 ** (3 - index))`：对于 1/4、1/8、1/16、1/32 四级，分别为 8×8、4×4、2×2、无池化；低频 scan 的栅格通常均为 8×8。**这只是低频通路，不等于特征整体变成 8×8**。

### 1.2 现有数值（报告/快照层事实，待 P0 用原始日志复验）

| SYSU，现有仓库汇总 | Recall | Precision | F1 | 备注 |
|---|---:|---:|---:|---|
| Run1 A2_STR | 81.24 | 85.29 | 83.21 | 无 CAACP、保留 STR |
| Run1 M1_FULL | **84.41** | 82.55 | **83.47** | 当前主对照，deploy 4,880,190 |
| Run2 E1_CP_CAACP | 80.94 | 85.43 | 83.13 | M1 + score 公式变更 |
| Run2 E2_FRH | 80.10 | 86.05 | 82.97 | M1 + 细粒度头 |
| Run2 E3_CP_FRH | 80.69 | 85.12 | 82.85 | 组合 |
| Run3 E4_RA_CAACP | 81.44 | 84.94 | 83.16 | residual anchor 改动 |
| Run3 E5_FS_TAR | 81.26 | 85.31 | 83.24 | stage1 3×3 temporal 分支 |

Run1 M1 `run2_zero_cost_diag.json` 报告 small(<256px) 0.1796 / medium(256–1023) 0.4620 / large(≥1024) 0.8273、boundary band±2 F1 0.6710。**这些 `comp_*` 数值有下面的 P0 统计语义问题，不可继续叫“严格对象级 F1”。**

**当前强证据支持**：M1 的高 Recall 工作点在 E1–E5 均显著后退（约 3–4.3pp），且 small 分组诊断异常低。

**尚未证明**：small 是否在 Stem/Stage2/Stage3 已经丢失、M1 的 CAACP 是否直接伤害 small、TAR 是否忽视 F1/F2、由更高分辨率 score 能否提升完整 F1。

### 1.3 本轮要分别检验的四个互斥程度有限的假设

| 假设 | 机制推测 | 证伪/支持证据 |
|---|---|---|
| H1 早期编码不足 | Stem/F1 不携带稳定变化证据 | F1 自身可分性低；独立探针也难恢复 small |
| H2 中间压缩削弱 | Stage2 下采样、Stage3 Prefix 或 CAACP 损失可分性 | 在同一 checkpoint 上某一阶段前好、后差，且反事实定位一致 |
| H3 多尺度融合抑制 | F1/F2 有证据但 TAR/DCR 未充分利用 | early probe 较好，TAR/DCR 特征或最终输出 small 显著较差；受控输入干预辅助佐证 |
| H4 输出判别不足 | DCR 特征可分但 head/final sigmoid >0.5 漏检 | 解码特征 probe 好，原头小目标召回低，probability calibration 诊断与之相符 |

不保证只有一个根因。若多个阶段同时衰减，报告**最早出现可重复明显恶化的节点**和**最大相对恶化的节点**，不要为了写论文强行只选一个。

---

## 2. 安全边界与执行环境

### 2.1 路径

```bash
PROJECT=/home/yqwang/projects/CASA-CD
DATA=/share_datasets/CD
CKPT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM
OUTPUT=/home/yqwang/outputs/CASA-CD
PRETRAIN=/home/yqwang/projects/CASA-CD/pretrained_weight/tinyvim_s_1000e.pth

# 本轮创建独立目录，绝不复用/覆盖 train_log.txt
DIAG_BASE=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1
```

环境：RSML-3、`casacd`、Python 3.10、Torch 2.14+cu132、2× RTX 5090；`selective_scan_cuda_oflex` 需能链接 torch/lib。**本轮默认仅用一张空闲 GPU，避免影响在训任务；不能假定 GPU 一定空闲。**

```bash
cd "$PROJECT"
git status --short
git rev-parse HEAD
nvidia-smi
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
python - <<'PY'
import sys, torch, scipy, cv2
print('python',sys.version)
print('torch',torch.__version__,'cuda',torch.version.cuda)
print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('scipy',scipy.__version__,'cv2',cv2.__version__)
PY
```

**本地 DSH**：若源码工作副本在 Windows，编写文件/跑 CPU-only 单元测试在本地；需要 `.pth`、CUDA 自定义 kernel 或真实数据时，通过已有的安全 SSH/SFTP 流程部署至 RSML-3，不能直接引用 Windows 路径假装可用。若当前代码与服务器仓库不同，记录二者 commit SHA 和未提交差异。

### 2.2 强制禁止

- 不执行 `git clean -fdx`、`git reset --hard`、覆盖式 `git clone`、`rm -rf` 既有资产；不删除 best/last 权重、训练日志、数据集、参考 PDF。
- 不在代码中出现为了诊断而临时 `model.train()`、改 BN running stats、改 optimizer/LR、test-label 参与推理路由；`GT` 只能用于**离线指标和诊断分桶**。
- 不直接调用 `analyse/extract_metrics_to_excel.py` 覆盖 `docs/experiment_metrics.xlsx`；不写回历史 `run2_zero_cost_diag.json`。
- 不把 `best_F1=x.pth` 文件名或训练过程中某个验证行当正式结果。只能用**同一个原始 train_log.txt 的最后一个完整 `=== TEST RESULTS ===` 至 `=== END TEST RESULTS ===` 区块**。
- 不以 `small F1` 单一数值或一张好看的热力图宣称因果结论；不能根据 test 表现调 probe 参数/阈值后再报告同一个 test 结果。

---

## 3. P0：首先修复“small component F1”的评价语义

### 3.1 已在仓库定位到的计算缺陷

`analyse/run2_zero_cost_diag.py` 中 `component_pr(pred_np, gt_np)` 先使用 `scipy.ndimage.label(gt_np > 0)`，构造指定 GT 正连通域掩码 `m`，再计算：

```python
TP = ((pred > 0) & (gt > 0) & m).sum()
FP = ((pred > 0) & (gt == 0) & m).sum()
FN = ((pred == 0) & (gt > 0) & m).sum()
```

由 `m⇒gt>0` 可知，**FP 恒等于 0**。于是单图 pseudo-Precision：有命中即 1，无命中为 0；`small P≈0.2` 并非“准确预测了 20% 的小目标区域”的正常 Precision，按图简单平均的 pseudo-F1 也不是对象级 F1。旧值 0.1796 / 0.8273 只能作为**历史受限正样本分组得分**，不得再用来设真实对象 F1 门槛。`run3_z12_summary.md` 中的 `small P/R/F1` 同理。

### 3.2 新增 `analyse/tvim_object_metrics.py`（只读、独立实现）

**A. GT 面积分组、全局累计像素召回（主诊断）**

对于 GT 的 4 邻接连通域（对应 SciPy 默认 `ndimage.label`），按面积 `a`：

- `tiny_1_15`：1≤a<16；`tiny_16_63`：16≤a<64；`small_64_255`：64≤a<256（细分诊断，避免全部 small 混在一起）。
- `small`：1≤a<256；`medium`：256≤a<1024；`large`：a≥1024（用于旧分组的口径对齐）。
- 所有面积均在**原生 256×256 GT** 上计算；不在 16² 的下采样标签上定义物体面积。

对每个 GT 对象 `G_j`：`tp_j = sum(pred[p]>0.5 for p in G_j)`、`r_j=tp_j/|G_j|`；按组给出：

1. **micro-pixel Recall**：`sum_j tp_j / sum_j |G_j|`（像素加权）。
2. **object-macro pixel Recall**：`mean_j r_j`（每个 GT 对象等权，能检验 1–15px 极小对象）。
3. **ObjectHit@1**：`mean_j [tp_j>=1]`（仅辅助，容易被偶然噪声命中）。
4. **ObjectHit@25%**：`mean_j [r_j>=0.25]`（本轮预注册主对象召回指标，不借助 IoU 调阈值）。
5. **ObjectHit@50%**：`mean_j [r_j>=0.50]`（辅助）；记录组内对象数、像素数、含该组图片数、未命中数量。

**注意**：这些是 `Recall/Hit`，不是 F1，也不是 Precision；永远不要用 `precision=1` 代替它们。没有 GT 对象的组：统计为 `null/NA`，不置 0，不参与 macro 分母。

**B. 真正匹配预测连通域的对象级指标（第二诊断）**

- 对预测二值图按**同样 4 连通**标记预测连通域，按每张图形成 GT×Pred IoU 矩阵。
- 固定一次性**最大权重二分匹配**（Hungarian 或具备确定性 tie-break 的同等实现）构造一对一匹配；仅保留 `IoU>=0.10` 为“宽松对象检出”，`IoU>=0.50` 为“严格对象检出”。不要用“任意重叠”作为唯一正式对象匹配口径。
- 全数据集 `ObjPrecision=matched_pred/N_pred`、`ObjRecall=matched_GT/N_GT`、`ObjF1=2PR/(P+R)`；对象面积组仅按 **GT 对象面积分层报告 ObjRecall**。由于预测端不能天然分配到 GT 面积分组，**不要在没有明确定义假阳性归属的情况下制造 `small ObjPrecision/F1`**。
- 记录 `N_gt,N_pred,N_matched,unmatched_pred,N_pred_area_bin`，报告 predicted-only 连通域的面积分布和 false-positive pixel area。小预测碎片很多时，严格 ObjectPrecision 应下降。
- 空图特殊情况：一张没有 GT 且没有 Pred 的图片，不增加 `N_gt/N_pred/N_matched`；有 Pred 但无 GT 时全部计 FP。全数据集分母为零时输出 `null` 并解释。
- 4 连通为**固定主口径**，8 连通只作为**一次性敏感性分析**，不可择优报告。

**C. 像素级全图与边界带**

- 全图按 **sum(TP/FP/FN/TN) → global P/R/OA/F1/IoU/Kappa**，与官方 `models/model/metric_tool.py`、`models/train.py` 的 `>0.5` 严格对拍。
- boundary band±2/±4 可沿用旧实现，但加 `GT boundary` 在图边缘时的操作定义、每组 TP/FP/FN、像素数、空 band 数。`±2px` 实际是以指定 structuring element 构造的形态学环带，不应称独立物体级指标。
- 分别报告 GT positive pixels、预测 positive pixels、TP、FP、FN；Precision 上升并不一定意味着 FP 绝对数量下降。

### 3.3 必做单元测试：`tests/test_tvim_object_metrics.py`

用全人造小阵列手算预期，禁止只靠真实数据抽样“看起来合理”。至少覆盖：

1. 一个面积 4 的 GT，预测完全命中：pixel Recall=1、ObjectHit@25%=1、IoU=1。
2. 4px GT 完全漏检：pixel Recall=0、ObjectHit=0；整个图片没有预测时对象 Precision 分母处理正确。
3. GT 命中，同时远处新增 3 个独立预测假阳性：**GT 组 Recall 不变，但全局 PixelPrecision 与 ObjectPrecision 必降**（直接回归旧 FP=0 bug）。
4. 2 个 GT 小对象被同一个预测连通域连接：一对一匹配至多命中一个 GT（避免多对一抬高 ObjectRecall）。
5. GT 大组件上只命中 1 像素：Hit@1=1，Hit@25%=0（证明两个指标不同）。
6. 各面积边界 15/16/63/64/255/256/1023/1024 恰好归入预定分组；全空 GT/Pred 的返回值符合规范。
7. 4 连通对角像素为两个对象；8 连通为一个对象（副分析只报敏感性）。
8. 背景假阳性只加 FP，不能被 GT 局部掩码过滤。
9. numpy 与 torch 掩码 `bool`、图像 batch size 1/16、CPU GPU 下同一二值图统计完全一致。
10. 对 256×256 原生 GT 的固定 image name 排序，改变 DataLoader batch size 不改变任何计数和面积组指标。

**Gate P0-METRIC PASS 条件**：上述测试全通过，输出 `metrics_protocol.json` 写明版本、4/8 邻接、IoU 阈值、>`0.5` 规则、面积组、空样本约定；对 M1 重算后不能覆盖旧的 0.1796，只能在报告中并列表为“旧伪 F1 / 新 small Recall / 新 ObjRecall”。

---

## 4. D0：环境、权重、数据与预测对拍——不通过就停

### 4.1 只核对并锁定 checkpoint，不按名字猜结果

**初始必要对照**：

```text
Run1/M1_FULL/SYSU-CD-256       ← 主模型
Run1/A2_STR/SYSU-CD-256        ← 无 CAACP、有 STR（用于机制相对参照）
Run2/E1_CP_CAACP/SYSU-CD-256   ← score 修改
Run2/E2_FRH/SYSU-CD-256        ← 末端 head 修改
Run2/E3_CP_FRH/SYSU-CD-256     ← 组合
Run3/E4_RA_CAACP/SYSU-CD-256  ← residual 修改
Run3/E5_FS_TAR/SYSU-CD-256    ← fine TAR 修改
```

**先只跑 M1**，M1 通过后再 A2；E1–E5 是后续扩展对照，不要求全部先跑。

`M1` 模型构造精确使用当前 `CASATViMSTRNet(tinyvim_pretrained_path=...,caacp=True,rep_mode='full',str_dim=96,caacp_score_mode='rank',frh=False,caacp_residual_mode='current',fs_tar=False)`；具体读取各 `train_scripts/.../train_SYSU...sh` 的实参，不能用默认 CLI 猜配置。先 build 模型、load state_dict、`eval()`，**严格核对 missing/unexpected keys**；探针的 `torch.load(...,weights_only=False)` 仅用于信任的本项目 checkpoint，不能加载未知来源文件。

`analyse/run2_zero_cost_diag.py` 通过 `sorted(glob(...best_F1=*.pth))[-1]` 选 checkpoint；**新代码禁止多 best 文件时按字典序猜哪一个是正式 checkpoint**：从原始 TEST 区块 `[BEST-F1]`、该实验 `run_manifest/sidecar`、实际文件内容作一致性校验；多候选但无法唯一确认则 `BLOCKED`，不擅自删除。

### 4.2 核对 TEST RESULTS 的正确方式

建议复用 `analyse/extract_metrics_to_excel.py` 的 `extract_test_block(text)` 与 `parse_block(block)`，**只导入函数、不要运行主程序覆盖 Excel**。为每个 Run 记录：原始日志路径、最后完整 TEST 区块起止、`Recall/Precision/OA/F1/IoU/Kappa`、训练图参数、部署图参数、可训练参数、deploy FLOPs、fold disagreement、训练预算/seed/batch、配置与 checkpoint SHA256。

缺日志不能用 README 代替正式结果；可先做技术性诊断，但 Gate 标为 `WARN/MISSING-LOG`，最终论文结论必须补原日志。

### 4.3 复算 M1 全图测试指标

复用 `models/eval.py` 同样的 test pipeline：`models/dataset/dataset.py` + `Transforms.Normalize(mean, std)` + `Scale(256,256)` + `ToTensor()`；从现有模型/训练参数读取 **BGR 通道顺序及 6 通道 mean/std**，不得凭 RGB 直觉更改。保证 `A/B/label` 使用 list 同名样本、`label=(gray>=128)`。

**注意官方 train.py 是 `p>0.5`，不是 `>=0.5`**。输出概率 Tensor shape `[B,1,256,256]`。无推理增强、无多尺度测试、无 TTA、不引入新阈值。

在 M1 全 4000 张 SYSU test 复算 `TP/FP/FN/TN` 和六指标，对照最后 TEST 区块：**以日志打印精度为准**，预期误差不超过 `1e-4`（更高精度统计值显示时可使用 5e-5 四舍五入容忍）；任何超差先调查模型配置、checkpoint、读图/归一化、BN eval、test list、CUDA 数值/折叠路径。

同一批图片还比较 train-graph eval 与 `deepcopy(model).switch_to_deploy()`：`max_abs_error`、`binary_disagreement`，**分母为全部 N×H×W**；hard gate：`binary_disagreement == 0`，不可用“很接近零”放行。可同时报告已有日志随机/真实 batch 的折叠差异。主审计不要在原 model 上永久 switch_to_deploy。

### 4.4 固定数据源 manifest 与人工样本索引

保存：`git_SHA`, `checkpoint_sha256`, `pretrain_sha256`, `test_list_sha256`, `dataset_count`, `index_to_name`（仅文件名）、`input/label shapes`、`empty_mask_count`、`total_gt_cc_by_size`。诊断目标对象的统计范围是**测试集所有图片**，不是“只挑有小目标的图片”。

**Gate D0-REPRO PASS**：TEST 块对齐、前向不崩、六指标复现、fold 二值 disagreement=0、P0 指标单元测试全绿。否则停止后续归因。

---

## 5. D1：修正后的小目标错误画像（Zero-training）

### 5.1 新增 `analyse/tvim_small_error_audit.py`

主输入：`--variant M1_FULL --run Run1 --dataset SYSU-CD-256 --ckpt ... --out-dir ...`；增量扩展 `A2_STR/E1/E2/E3/E4/E5`。每次只加载一个 variant、按文件名稳定顺序遍历全 test；输出 CSV/JSON，不保存全部 `[N,256,256]` logits（避免多 GB 文件）。

每张图和每个 GT CC 记录：

```text
sample_name / gt_component_id / area / bbox(x1,y1,x2,y2) / bbox_width,height
bbox_aspect_ratio / perimeter_approx / occupancy / touches_border
pixel_recall / hit1 / hit25 / hit50 / matched_iou_010 / matched_iou_050
model_p_mean_in_gt / model_p_max_in_gt / model_p_p90_in_gt
image_gt_area_ratio / image_fp_pixels / image_fn_pixels
```

GT object bbox 与面积是统计属性；**训练或正常 inference 不可以读取这些 GT 特征**。特征 `occupancy=area/(bbox_width*bbox_height)` 可区分线状细长目标与块状目标，不能仅凭面积假定目标映射到 `<1` 个特征 cell。

生成以下汇总：

- 各尺寸 `GT object count / pixel count / pixel Recall / object-macro Recall / Hit@1/25/50`；
- 按 `bbox_min_width <4 / 4~7 / >=8`、细长比、边界接触、变化密度分层的遗漏率；
- SYSU 所有 FP 像素、FP 连通域面积直方图、空图 FP 情况；
- boundary±2/±4 pixel F1 与各面积组中处于 band 内/外的 GT TP/FN；
- M1 vs A2 的同一样本/同一个 GT 对象 `Δrecall`（用 sample_name+GT component ID 稳定匹配）。

**主视角是 paired per-object delta**，不是仅比较两个模型各自平均。固定同一 test labels 避免样本组成偏差。

### 5.2 反驳“真的只是小目标”所需的控制

从诊断中固定分组对照：

1. 变化区域 small；2. medium；3. large；4. GT 无变化背景；5. 贴边/细长变化；6. 难负样本（GT=0、但图像 A/B 差异大）。

像素级混淆矩阵和分组 FP/FN 的绝对计数要同时给出。特别对 Run2/3 的“P↑R↓”回答：`TP` 降了多少、`FN` 增了多少、`FP` 绝对减少多少。不能只凭 Precision 比值作结论。

### 5.3 可视化要求（固定规则，不得手选好看的样本）

选择样本**仅用于说明，不进入指标计算**，选择逻辑预注册：按 `small FN 数量` 最大的前 6 张、按 `small FN rate` 最大且 small GT>=3 的前 6 张、small Hit@25 成功但边界 F1 低的前 4 张、FP 像素最多的前 4 张、随机固定 seed16 的 8 张，共去重最多 32 张。每张保存 `A/B/GT/M1_pred/A2_pred/error_overlay`；把 `TP` 绿、`FN` 红、`FP` 黄图例固定；未经许可不把图片拷贝出校内服务器/公开仓库。

**Gate D1-PHENOMENON**：证实“small 对象 Recall 低于 medium/large”需要同时提供分组样本数和 object-macro/pooled recall；若旧 `0.1796` 实际只是伪指标口径带来的夸张，立即更正研究结论，不能再沿用原叙事。

---

## 6. D2：按模型真实计算顺序定位首次证据衰减（本轮核心）

### 6.1 新增 `analyse/tvim_stage_recoverability.py`

以一个训练完成的**固定 M1 checkpoint** 为主，冻结全部参数、`model.eval()`、`torch.no_grad()`。只加临时 hook 或用不改输出的 `forward_with_diagnostics`，取以下节点：

| 编号 | 精确可观测节点 | 分辨率 | 备注 |
|---|---|---|---|
| L00 | `encoder.patch_embed` 后 | 64² | Stem 输出；非最终 F1 |
| L01 | `encoder.network[0]`/`norm0` 后 | 64² | F1，48C |
| L02 | `encoder.network[1]` 后 | 32² | Stage1→Stage2 的 stride-2 downsample 之后、stage2 blocks 之前 |
| L03 | `encoder.network[2]`/`norm2` 后 | 32² | F2，64C |
| L04 | `encoder.network[3]` 后 | 16² | Stage2→Stage3 stride-2 downsample 之后 |
| L05 | `encoder.network[4][:8]` 后 | 16² | Stage3 prefix，CAACP score 输入 `x`（命名以实际 depth 校验） |
| L06 | `encoder.network[4][8]` 后 / `norm4` 后 | 16² | Stage3 final TViM & F3；报告 norm 前后 |
| L07 | `encoder.network[5]` 后 | 8² | Stage3→Stage4 下采样后 |
| L08 | `encoder.network[6]`/`norm6` 后 | 8² | F4，224C |
| T01–T04 | `tar.stage{1..4}` 输出 | 64²/32²/16²/8² | 96C；这是**双时相融合后的单路变化特征**，不能再取 A/B cosine |
| D01–D04 | `decoder` 的 d3、d2、d1、refine 后 | 16²/32²/64²/64² | 需 hook 或只读诊断版明确逐层提取 |
| P00 | `head` 原始 logits 后 | 64²（FRH 为 128²） | 概率 sigmoid 前最好记录 logits |
| P01 | bilinear 后 sigmoid | 256² | 最终 0.5 阈值输出 |

**实现关键**：`F1/F2` 的 `norm0/norm2` 是分支读取，不改变送进下一层 `x`；pre/post 通过 `torch.cat([pre,post],dim=0)` 合批。不能用 `model.encoder.forward()` 代替主线的 `forward_stem_s2_prefix → set_pair_score → forward_caacp_block → forward_stage4`：普通 `encoder.forward()` 本身没有注入 paired score，可能得到不同结果。

最优实现顺序：优先采用 `register_forward_hook` 记录网络节点（浅拷贝只读、最后清除）；若要取 DCR 中间 `d3/d2/d1`，可新增只在诊断调用的逐层函数并与原 `DCRDecoder.forward` 的输出逐位对拍。**不要调用标准 `forward` 两次造成 score 状态或 BN 统计差异**。

### 6.2 不能把所有层一律做 A/B cosine

**编码器各节点 L00–L08**：均有 A/B 配对特征，定义余弦差分（仅观察性 proxy）：

\[
S_l(u,v)=1-\frac{\langle F_l^A(u,v),F_l^B(u,v)\rangle}{\max(\|F_l^A(u,v)\|_2,\varepsilon)\max(\|F_l^B(u,v)\|_2,\varepsilon)}.
\]

与 `change_score_cosine_2d` 一致时可 `F.normalize` 后求 `1 - sum`；对照保留 `L2` 差分、逐位置特征范数，以及 `(F^A-F^B)` 的 L2，不应仅靠一个 cosine 指标下结论。

**TAR/DCR 各节点 T/D**：已经是 A/B 融合后的单路特征，不可以对同一个 tensor 的两半执行 cosine。要测可恢复性，只能用固定容量 probe / 线性读出（见 D3），或查看同一模型特征对 GT 的统计信噪比和图像叠加。

### 6.3 分辨率对齐：必须双口径

低分辨率层之间比较很容易因 label 重采样造成假性“信息消失”。每一层的解释必须同时提供：

**主口径（原生 256² 共同栅格）**：将 score 以固定 `bilinear, align_corners=False` 上采样到 256²，**保留原 GT（绝不下采样 GT 来做主指标）**，计算：

- `Average Precision / PR-AUC`（推荐用 sklearn `average_precision_score`，记录公式与正类率 baseline）；
- 在**同一固定 FP budget**（例如固定背景预测像素比例 1%/5%，这是诊断排序曲线采样点而不是部署阈值）下的 TP Recall；
- GT 面积分组的 `score_mean / score_p90 / score_max` 与对应背景分布；
- 小目标的 per-object `score-positive-vs-local-background` margin（周围扩张环带做对照，排除目标本身）。

**副口径（native feature grid）**：用 `adaptive_avg_pool2d` / 精确整数分块平均得到每个 feature cell 的 `GT occupancy∈[0,1]`，并用 `maxpool` 得到“是否触及 GT”标签；输出 occupancy 分桶下的特征分数分布，不用 nearest label 导致小对象直接掉成 0。需区分：`cell touched` 不等于 `cell fully positive`，不能把它作为完整 mask 的 ground truth。

**比较时固定**测试样本、GT、up/downsample 几何和评估脚本，统一分组；AP 必须同时报告正类先验比例，否则跨数据集绝对 AP 不可直接类比。提供 pooled-micro 和 per-image-macro 两种聚合口径，空图单独统计；主判定采用 pooled AP + GT small object score margin。

### 6.4 “首次衰减”定义（必须预先固定）

不是“哪一层的原始 cosine AP 最小就说哪一层丢失”：通道维度不同、特征语义不同、Stage4 更粗，都影响 proxy。采用多证据级别：

- **S0：直接 proxy**。记录从 L00 到 L08 的 `AP / 小目标 margin / 大目标 margin / 硬背景 FP rate`。如果相邻两个有相同空间分辨率（L05 vs L06）仍出现下降，空间 alias 的混杂更少。
- **S1：受控弱探针**。D3 中对每个层冻结特征做同容量线性 probe，训练集训练、测试集评价；判断小目标可读出性是否真的下降。
- **S2：配对反事实**。D4 在**同一已训练权重**上受控关闭 CAACP 修正/某路径，观察相同 GT objects 的输出变化；这仅支持“该已训练模型当前依赖该路径”，不能证明从头训练因果。

只有 `S0+S1` 方向一致（最好附 S2）才允许写“某一阶段更可能是瓶颈”；如果 S0 与 S1 相反，优先认定“表示方式变化/原始 proxy 失效”，而不是硬解释为丢失。

### 6.5 Stage3 CAACP 局部内部证据（尤为重要）

因为当前创新机制恰好作用于 Stage3 最后一个 TViM，要在**同一个 op 内**记录：

```text
X_low @16² → C_avg @8²
           → C_ca  @8²（rank + shared 2×2 weight）
           → ΔC=C_ca-C_avg
           → C=C_avg+βΔC → selective scan → upsample + residual
X_high @16² → local_conv / high-frequency dense 分支
                           → concat → out_proj → TViM 残差 → F3
```

记录：`β`、`mean_abs(ΔC)`、`mean_abs(βΔC)`、`||βΔC||/||C_avg||`（分母加 eps）、`ΔC` 在 GT small/medium/large 对应 cell 的位置分布、cell 权重 entropy 与 top1 weight、low/dense feature 范数、SS2D 前/后变化分数差异、`residual=x0-Up(C)` 的 GT 分桶。注意 score 由 **no_grad** 的 A/B cosine 得到，score 权重不反传，但 β 与 X 路径仍可传梯度；不能把 stop-grad 当成“无法学习 CAACP”。

**已知记录命名潜在误导**：旧脚本将 batch 内 `mean(|ΔC|)` 的平方做汇总开方并命名 `context_delta_rms`，它不等于所有张量元素的严格 RMS。新诊断需要同时记录精确 `sqrt(mean(ΔC²))` 和 `mean(|ΔC|)`，不可混为一谈。

**诊断接口安全实现**：如果现有模块没有内部 hook，允许通过单独 diagnostic wrapper 或局部 `forward_core` 的只读采样在 `no_grad` 下记录，不能重新实现一套不同的 `cross_selective_scan` 算法来当原模型；所有辅助 tensor `.detach()`，按样本汇总，避免长时间保留 GPU 大 tensor。

### 6.6 预期输出

`stage_raw.csv`（每图×每层统计）、`stage_summary.json`（pooled AP/macro AP/small margin/大小对照/占比）、`stage_profile.png`（分阶段趋势；注明 score 方法）、`stage_examples/`（固定抽样）、`caacp_internal.json`；每个文件含 metadata 指向同一 manifest。

**Gate D2-VALID**：逐层 shape 与网络代码一致；开启/关闭 hook 的最终预测 `torch.equal` 或 `max_abs=0`（优先逐位；不支持逐位时列出原因且二值 disagreement=0）；同一个样本批次切片顺序正确；特征 score 中无 NaN/Inf；预注册的双栅格评价完整。

---

## 7. D3：冻结线性 Probe 识别“信息存在但被度量隐藏”

**这一步可选但优先级高。** 若 D2 只是 cosine 热力图，并不足以得出“早期信息丢失”。

### 7.1 新增 `analyse/tvim_linear_probe.py`

输入冻结模型 checkpoint 与确定的层名，可读出：

- 编码器双时相节点 L00–L08：统一 probe 输入 `concat(F_A,F_B, abs(F_A-F_B))` 或预注册更紧凑的 `abs(F_A-F_B)`，**只选一种主口径并在所有层保持一致**。
- TAR/DCR 节点：单路融合特征 `T/D`，输入通道维不同；使用同一结构范式 `Conv2d(C_in,1,kernel_size=1)`，仅最后读出容量随 C 线性变化，在报告中列出输入通道与探针参数，避免把不同参数量直接称为公平容量对照。
- `Conv1x1` probe **仅仅是离线可读性测量工具，不加入 CASA 网络，不以 probe 指标代替训练完成的正式变化检测成绩**。

### 7.2 公平训练约束

- **冻结整个 M1 编码器和 TAR/DCR**，包括全部 BatchNorm 统计；`torch.no_grad()` 提取特征，检测探针梯度只落在 probe。
- probe 的拟合只用 SYSU `train.txt`；为避免 4k test 被用于训练决策，建议在训练集内部用**固定文件名 hash 划分 90/10 的 train-probe/val-probe**，以内部验证选共同一次性规定的学习率/早停，随后锁定超参，test 一次性评估。禁止在 test 上挑 probe 版本、迭代步数、阈值。
- probe 使用 `BCEWithLogitsLoss` 作为**诊断分类器的常规拟合目标**，不是修改 CASA 主模型 BCE+Dice，也不是“新 loss 创新”。以每层相同固定学习率、最大训练步数和样本顺序为前提；示例预注册 `lr=1e-3, AdamW, steps=2000, batch_size=16, seed=16`，这只是可执行默认值，DSH 在开跑前写入 manifest，不得看 test 结果再调。
- 同一原生 256 GT 的 score 评估，同一上采样方式；train 时建议原生 feature grid 的 soft occupancy 监督/或统一 256 native mask 上采样 logits，在所有层使用同一策略，避免在不同分辨率把 weak target 训练信号裁没；具体选择必须先在 `probe_protocol.json` 锁定。**优先选“upsample logits 到 256²，再以原 mask 计算诊断 probe BCE”**。
- 训练集 probe 只做诊断，不进入主 model checkpoint；模型输出文件名带 `PROBE_ONLY`，不放在 `/share_datasets/.../checkpoints/CASA-CD` 的正式模型目录。

### 7.3 判据与报告

对每层报告：`train-probe/val-probe/test AP`、按 GT object size 的 pixel recall/score margin（probe 的固定阈值仅作为辅项）、GT small 与 hard-negative 的分数分布。**主对照指标 test AP 不依赖阈值**。

如果 `F1/F2 probe` 能较好分出小变化但 `L05/L06` 明显恶化，支持中间处理环节损失；如果多个 encoder probe 都较好但 `T/D probe` 恶化，支持后端融合不足；如果不同层 probe 的 AP 接近、仅 cosine 差异较大，则应写“余弦 proxy 表达不足”，而不是“信息丢失”。

**可选附加控制**：同一层随机置乱样本-标签的 train probe，test AP 应回落到接近正类基准；确保标签泄漏/索引错位不会虚构高 AP。

### 7.4 Gate D3-VALID

- 探针训练的 split 与 hash 可审计、没有 test 参与任何优化/早停/调参；
- 记录模型冻结参数 SHA256 前后相同；
- 每层 probe 只看本层特征，没有 GT oracle feature 与直接读取模型最终 logits；
- train/val/test 源文件列表互斥（文件名排他性），整轮评估只运行已锁定的 probe。

---

## 8. D4：受控反事实（零训练，不可冒充严格因果消融）

### 8.1 CAACP β 运行时关闭

在训练好的 Run1 M1 checkpoint 上：

- `Y_on`：原模型、原 β。
- `Y_off`：`deepcopy(M1)`，仅 `with torch.no_grad(): model.encoder.caacp_op.beta.zero_()`，其余**训练后的参数完全相同**。
- 输出对相同 A/B test 顺序的 `small GT recall`、`M1 all-pixel Recall/Precision`、`GT tiny object Hit@25`、`per-object paired Δ`。

解释边界：`Y_on - Y_off` 衡量**当前训练权重对 CAACP 修正项的局部依赖**；M1 的其余参数是在 β 可学习条件下训练的，不能当作 A2（重新训练无 CAACP）的受控对照，也不能直接证明“Stage3 是最早损失源”。

另有独立训练对照 `A2_STR`（同训练协议）可作为**相对结构作用**证据，但因训练轨迹和其他参数全都不同，不能用于定位精确的层内信息流失时刻。报告应将两个比较并排、分别标注。

### 8.2 禁止误用的假因果实验

- 把 TAR 某级输入直接置零，然后断言“该级不重要”：这会造成严重 out-of-distribution (OOD)，最多作为路径敏感性辅助诊断。
- 把 `F2` 从 M1 替换为 A2 的 F2：两模型 encoder 在训练后参数不同、BN 不同，跨模型拼接无自然数值等价；不能称严格反事实。
- 用 GT 变化掩码设置 CAACP score 得到改善然后宣称可部署：这是 oracle-only 上限诊断，不可作为实际方法或推理性能。

### 8.3 可选：诊断 TAR 对 F1/F2 的利用不足

如 D3 已显示 `F1/F2` 可读但 DCR 受损，允许在单模型固定 checkpoint 下添加只读 activation/gradient sensitivity 分析：`||∂logit/∂t_i||` 的空间归一化分布，或对各 `t_i` 固定小幅、同范数扰动的响应（必须报告 OOD 局限）。不同分辨率 gradient norm 不直接可比，需统一元素数/通道数归一化。**只用于决定研究方向，不能据此命名“已证实的抑制”。**

**Gate D4-COUNTERFACT**：原 ON 输出复现；OFF 只改变 β 一个参数；对照的其余参数 checksum 一致；无重训练；`ON/OFF` 的模型结构及预测可重复；报告明确其解释边界。

---

## 9. D5：四数据集一致性校验（只有 SYSU Gate 明确后执行）

本轮先在 SYSU 做全部 P0/D0/D1/D2/D3/D4。若定位到具体阶段，再对 `LEVIR-CD-256`、`WHU-CD-256`、`CDD-CD-256` 的 M1 **只重复 D0 + D1 + D2 中被证实关键的两三个节点**；不立即为每个数据集训练 probe。

目的：检验是否为 SYSU 特定现象还是跨数据集共性。至少报告四数据集 `small object count`、`small pixel recall`、`ObjectHit@25`、对象形状分布、两个关键阶段的 AP/score margin 变化、全图 F1/IoU、FP 数。避免把 SYSU 单数据集结果描述为“普适规律”。

特别注意：LEVIR/WHU 低变化率、空变化样本多；不能拿不同数据集的 pooled AP 数字横比就宣称阶段瓶颈一致，应报告正类比例与**阶段间同数据集相对变化**。

---

## 10. 统计可信度、对照与可证伪判决

### 10.1 单 seed、样本依赖与不确定性

沿用主模型 seed16，不选择第二 seed。对于同一 checkpoint 的 paired diagnostic，允许**不重新训练**、以**图像为抽样单元**（非以像素或连通域独立抽样）做固定 `bootstrap=1000, seed=16`，输出配对差异的 95% bootstrap 区间；大图/多对象相关性通过 image-level 聚类保留。这个区间只表征有限 test 集采样不确定性，不是跨训练 seed 的稳定性证据。

不要因 AP 数量差 0.001 就认定方法有效；若分层对象数过少、差异区间宽、数据定义不可靠，标 `INCONCLUSIVE`。

### 10.2 提前锁定的观察性机制阈值（不是新训练成功承诺）

建议由 DSH 在**开始运行前**将下列门槛固化在 `diagnosis_plan.json`，过程中不因结果不理想修改：

- `D1 size gap`：GT small 的 pooled pixel Recall 或 ObjectHit@25 **比 large 低至少 10pp**，且每组至少 100 GT 组件（否则只作描述性差异）；
- `D2 stage drop`：一个相邻编码器边界中，small GT 对比背景的 AP 或 margin **相对下降 ≥10%**，并且 paired image bootstrap 95% 区间差异方向同号（若 margin 可能过零，则主要用 AP 的绝对百分点差异）；
- `D3 recoverability`：相同 probe 协议下 small probe AP 在一个相邻关键阶段**绝对下降 ≥0.02**且配对区间指向同号；`D2` 和 `D3` 同向才称“高度怀疑该阶段损失”；
- `D4 on/off`：若 βOFF 改善 small 但 overall 降低，写“存在 tiny-vs-overall 权衡”，不得写“关闭 CAACP 能提升模型”；如差异微小/非稳健，判“不支持 β 本身是主因”。

上述数值是**诊断预注册建议值**，不是历史文献证明的普适标准，不应把它们转化为“每个阶段必须有提升”。尤其跨层粗细尺度不一致时，优先参考 D3 与对照合理性。

### 10.3 最终决策矩阵

| 最终证据组合 | 结论等级 | 下一轮主方案候选 |
|---|---|---|
| F1 已低且 probe 同样低 | `[I] H1`，需排除数据配准/标签误差 | 早期共享编码/temporal evidence；**不能直接增加大骨干** |
| F1/F2 好，F2→Stage3 Prefix 探针明显恶化 | `[I] H2` | Stage2 downsample / Stage3 prefix 的最小修改；先不假定 CAACP 是主因 |
| Stage3 Prefix 好，CAACP 后明显恶化且 βOFF 部分恢复 | `[I] H2-CAACP` | 优先研究 CAACP score 来源/low-vs-dense 选择；可评估 Stage2 relocation，但需修改原位置约束 |
| 编码器 F1/F2/F3 可读，TAR/DCR 明显恶化 | `[I] H3` | 高分辨率跨尺度证据保留；不再扩大 3×3 spatial kernel 盲试 |
| DCR 好，head 后 weak-positive 下降 | `[I] H4` | 主预测读出机制；已有 FRH 失败，不重复“再加边界头” |
| 多层均类似，尺度差异仅出现在 cosine proxy | `[I] proxy mismatch` | 不改架构，重新检验 representation/readout 解释 |
| D1 数据不足 / D2 与 D3 互相矛盾 / D0 不通过 | `INCONCLUSIVE/BLOCKED` | **停模型改进**，补证据，不跑 80K |

### 10.4 候选改进的排序纪律

最终仅提出 **2–4 个候选、择一主方案+必要对照**。不能因为 Run3 报告提过 S2-HCAACP 或 SP-DCR，就未经诊断直接宣布下一轮 Run4 必须做它们。每个候选必须写：**针对哪一节点的直接证据、机制数学定义、与 CAACP/STR 的本质区别、精确参数/FLOPs、预训练兼容与零初始化、train/deploy 图、最小消融、成功和失败判据**。建议优先比较：

- 候选 A：`Stage3 CAACP` 位置不变，仅使用 F2 引导的变化 score（须证明信息在 F2 较好；参数可近 0，但额外 score 计算量/时延需实测）。
- 候选 B：`Stage2-only hierarchical CAACP` 替代 Stage3（若证据明确指向更早空间聚合；这是对“仅 Stage3 改动”约束的例外申请，**必须征求用户批准**，不能擅自实现）。
- 候选 C：`TAR/DCR` 中已有 F1/F2 的变化证据保留（仅在 H3 成立时考虑，必须可折叠且≤5M）。

**所有这些现在只是候选，不是本轮实现内容。**

---

## 11. DSH 具体实现文件与 CLI 约定

### 11.1 建议新增（非覆盖）文件清单

```text
analyse/tvim_diag_common.py                # 固定路径/模型构造/严格日志解析/manifest/metrics helper
analyse/tvim_object_metrics.py             # 正确 GT 面积与预测 CC 的像素/对象指标
analyse/tvim_small_error_audit.py          # D1 全数据集 GT 组件错误画像
analyse/tvim_stage_recoverability.py       # D2 feature hook、A/B score、stage summary
analyse/tvim_linear_probe.py               # D3 可选冻结 probe，训练/验证严格分离
analyse/tvim_caacp_counterfactual.py       # D4 βON/OFF 固定 checkpoint
analyse/tvim_diag_report.py                # 汇总所有 evidence、Gate 与决策
analyse/tests/test_tvim_object_metrics.py  # 最少第3.3节 10个用例
analyse/tests/test_tvim_feature_capture.py # shape、pair、hook 等价与清理
train_scripts/CASA-TViM/Diag1/run_diag.sh  # 一键分阶段 Gate，不覆盖旧脚本
train_scripts/CASA-TViM/Diag1/README.md
```

文件名可做极小调整，但须在交接说明映射，不能把新分析代码插入正常生产 `train.py` 的默认路径。`pytest` 若未安装，可用标准 `unittest` 实现，不因此修改主环境包版本。

### 11.2 通用 CLI（DSH 实现时必须真正支持而不是伪示例）

```bash
# 这些是要求 DSH 实现的命令接口，当前 GitHub 不存在这些新脚本！
cd /home/yqwang/projects/CASA-CD
export CUDA_VISIBLE_DEVICES=1  # 仅示例；先由 nvidia-smi 检查空闲卡
DIAG=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1
mkdir -p "$DIAG"

# 1. P0 metrics unit tests（无需数据）
python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v

# 2. D0 严格对拍 M1。GPU_VISIBLE后 torch 设备为 cuda:0
python analyse/tvim_diag_common.py audit \
  --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D0/M1_FULL"

# 3. D1 小对象诊断
python analyse/tvim_small_error_audit.py \
  --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D1/M1_FULL"

# 4. D2 full test 阶段信息流失诊断
python analyse/tvim_stage_recoverability.py \
  --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D2/M1_FULL"

# 5. D4 β 反事实（不是从头重训的消融）
python analyse/tvim_caacp_counterfactual.py \
  --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D4/M1_FULL"

# 6. 先 D0-D2 输出诊断报告，再判断是否启动 D3 probe
python analyse/tvim_diag_report.py \
  --root "$DIAG" --out "$DIAG/DIAGNOSIS_REPORT.md"
```

**Probe D3 示例：**

```bash
# 仅当 D0/D1/D2 PASS 后执行；独立列出探针运行预算与结果
python analyse/tvim_linear_probe.py \
  --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --seed 16 --steps 2000 \
  --out-dir "$DIAG/D3/M1_FULL"
```

`run_diag.sh` 需先运行 `P0` 与 `D0`，读取每阶段机器可解析的 `gate.json`；后续阶段必须根据 Gate 逐项显式启动，不可误写“失败后照跑”。加入 `--limit N` 的 smoke（固定前 N 张，仅标 `SMOKE_NOT_FULL`），正式结论必须 `--limit 0` 全测试集。

### 11.3 产物目录

```text
/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1/
  RUN_MANIFEST.json               # 统一代码/数据/模型 SHA 和协议
  DIAGNOSIS_REPORT.md              # 给导师/研究助手的摘要、证据、Gate 与方向判断
  REPRODUCE.md                     # 实际执行命令、耗时、设备、异常与恢复
  gate_summary.json
  D0/M1_FULL/{test_result.json,fold_equivalence.json,gate.json}
  D0/A2_STR/{...}                  # M1 通过之后
  D1/M1_FULL/{object_level.csv,image_level.csv,summary.json,examples/,gate.json}
  D1/A2_STR/{...}
  D2/M1_FULL/{stage_raw.csv,stage_summary.json,stage_profile.png,caacp_internal.json,gate.json}
  D3/M1_FULL/{probe_protocol.json,probe_results.csv,probe_weights_PROBE_ONLY/,gate.json}
  D4/M1_FULL/{paired_counterfactual.csv,summary.json,gate.json}
  D5/{...}                        # 只有 SYSU Gate 支持后跨数据集扩展
```

写文件采用**新建独立目录**，若路径已存在，改用 `Diag1_<timestamp>` 或按用户确认恢复原诊断 manifest，绝不静默覆盖；长时间程序 checkpoint 只用新的 `PROBE_ONLY` 输出目录，断点恢复必须验证 protocol/checksum 一致。

---

## 12. Smoke、Dry Run、正式运行的顺序

### 12.1 Smoke（CPU 单元 + 最小 GPU，无真实数据训练）

**Smoke S0**：上节 10 项 component metric 合成测试全部 PASS；FP 加入必须降低 Precision。

**Smoke S1**：随机 A/B，`model.eval()`；hook 关闭 vs hook 开启的同一 forward 最终输出一致（目标 bitwise `torch.equal`）；原本的 weights/buffers checksum 不变；每次 hook 用 `try/finally` 移除，无重复注册。

**Smoke S2**：真实/合成 tensor shape 检查：L00–L08、T01–T04、D01–D04，按 2B 合批和 A/B 分半；0.5 二值预测与独立 `eval.py` 输出一致。

**Smoke S3**：D2 聚合：相同结果以 batch1、batch16 重排数据，理论汇总相同（浮点容差例如 1e-6）；`np.isfinite` 全 PASS；空 GT、全正 GT、含 tiny GT 都可处理。

**Smoke S4**：D4 复制模型，ON 与当前模型 bitwise 一致；OFF 的 β=0 且其他参数、buffers 不变；`model.eval()` 后 BN running stats checksum 不变。

**Smoke S5（如果运行 D3）**：全模型前后 checksum 相同，probe 有梯度、主模型无梯度，train/val/test 文件集合互斥，训练集 shuffle 受 seed 控制。

### 12.2 Real Data Dry Run（只读前 16/64 张）

- `--limit 16` 验证完整 I/O、BGR/label、单图组件统计、json/csv 输出。
- `--limit 64` 观察推理耗时、GPU peak VRAM、hook buffer 是否泄漏、预测对拍误差。
- dry run 必须标 `dry_run=true`、`not_for_conclusion=true`；这些结果不能作为“SYSU 诊断结论”。
- 若测得样本 CPU 连通域匹配/概率存储的额外开销较大，优化分析代码不改变数学定义/抽样方法（例如逐图增量汇总）。

### 12.3 正式步骤

1. P0 环境/代码审查 → S0–S4 smoke → `gate-P0 PASS`。
2. D0 M1 全 4000 张正式 test 复算、fold 验证；失败停。
3. D1 M1 全量错误画像 + A2 全量同图对象配对对照。
4. D2 M1 全量分阶段（先 16/64 dry run 验证无显著显存问题）。
5. 评估 D2 与 D1 是否同向；必要时 D3 probe，同一冻结 M1、train-only 拟合。
6. D4 βON/OFF（M1）+ 可选 E1–E5 回顾，控制总体计算预算。
7. 仅在 SYSU 结论稳健时对另外三数据集做收窄 D5。
8. 更新完整诊断报告；**此时停止，等待用户选择唯一创新方案**。本轮不自动训练新的 80K，不对 old logs/checkpoints 做清理。

---

## 13. 精确恢复与异常处理

- **GPU OOM**：优先将 batch 16→4→1、减少同时缓存的 feature、图像逐张写 CSV；不得偷偷缩小输入、减少 test 集、换模型。重启后从新的诊断 manifest + 按 name 记录的已完成样本索引精确恢复，跳过需对 SHA 匹配。数据流分母完整才算 `FULL`。
- **CUDA kernel import 失败**：先核 `LD_LIBRARY_PATH` 和 `selective_scan_cuda_oflex` 的可用性。不要修改 SS2D 算子逻辑去临时造新前向，若兜底 Triton/其他 kernel，必须先与原 kernel 对拍，且输出误差/阈值 disagreement 记录在 D0。
- **原日志无完整 TEST 块**：标 `BLOCKED-RESULT`；技术性测试可做但不得标“复现正式结果”。
- **best 文件不唯一/缺失**：标 `BLOCKED-CKPT`，请用户确认可用 checkpoint；禁止按文件名最大就强行跑。
- **数据读取错误/label 变成灰度或彩色**：先按 256×256、gray>=128 与 list 文件核对，再跑模型。
- **指标不复现**：优先核对模型状态字典/BN/配置/归一化顺序/部署图/测试列表，然后才考虑数值 kernel 差异。
- **Stage score 很弱但 probe 很好**：该位置不允许断言“信息丢失”，先认为简单差异度量不足。
- **对象群体很少/Bootstrap 区间跨 0**：结论标 `INCONCLUSIVE`，不借小样本微弱差异选择模型方向。

---

## 14. 最终报告强制模板

DSH 最终完成后创建 `DIAGNOSIS_REPORT.md`，按以下模板填写真实结果（占位符未运行前必须保留空，不可臆造）。

### 14.1 实验身份与完整性

```text
repo sha: ...
server source sha / dirty status: ...
M1 checkpoint path + sha256: ...
A2 checkpoint path + sha256: ...
TinyViM pretrained sha256: ...
SYSU test_list sha256 / n images: ...
model arch/caacp/score_mode/residual/rep/frh/fs_tar: ...
torch/cuda/kernel/device: ...
original train_log path / final TEST block location: ...
D0 reproduced full R/P/OA/F1/IoU/Kappa: ...
train/deploy params/FLOPs + unsupported_ops: ...
fold max_abs + binary_disagreement: ...
P0/D0/D1/D2/D3/D4 gate: ...
```

### 14.2 核心诊断表

| 节点 | 原生尺度 | small GT count | small pixel recall 或 margin | pooled AP | probe AP（如执行） | 分辨率控制 | 证据等级 |
|---|---|---:|---:|---:|---:|---|---|
| F1 64² | 1/4 | 待跑 | 待跑 | 待跑 | 待跑 | 上采样/occupancy | [M] |
| F2 32² | 1/8 | 待跑 | 待跑 | 待跑 | 待跑 | 同上 | [M] |
| Stage3 Prefix 16² | 1/16 | 待跑 | 待跑 | 待跑 | 待跑 | 同上 | [M] |
| Stage3 Post/ F3 16² | 1/16 | 待跑 | 待跑 | 待跑 | 待跑 | 同上 | [M] |
| F4 8² | 1/32 | 待跑 | 待跑 | 待跑 | 待跑 | 同上 | [M] |
| TAR/DCR（单路） | 多尺度 | 待跑 | 禁用 A/B cosine | 不适用（原生 score） | 待跑 | probe 同协议 | [M] |

完整报告另需：`旧伪 small-F1 vs 新真实对象召回`、`M1 vs A2 same-objects paired`、`P↑R↓ 的 TP/FP/FN 绝对计数`、`CAACP βON/OFF`、`图像级 bootstrap CI`、`跨数据集 D5（如果做）`、有固定规则的 10–20 个失误样例图。

### 14.3 分阶段结论（每条含可核验证据）

- `[F]` 哪个结构节点使用了哪种池化/SS2D，哪里还保留高频信息。
- `[F]` 修复了什么指标口径 bug，旧 0.1796 实际代表什么；新 small 正确召回/检测率是多少。
- `[I]` 哪一个相邻处理阶段发生第一次稳健衰减，证据来自哪些表和哪些统计口径。
- `[H]` 如果该阶段更改结构，预计改善哪类对象、可能伤害哪个负样本群体；明确机制失败的替代解释。
- `[M]` 仍缺哪一条足以推翻当前判断的关键证据。

### 14.4 对下一轮 Run4 的建议（只推荐，不执行）

写**2–4 个候选，排序并且只选 1 个主方向**，包括核心机制、pretrained zero-init、增量 deploy Params/FLOPs、单变量消融、测试设计、失败阈值。明确是否需要用户批准放宽“CAACP 仅 Stage3”约束。**不要在本诊断任务末尾直接开始新的 full training。**

---

## 15. 验收清单（DSH 提交前逐项勾选）

- [ ] **代码身份**：记录当次 Git HEAD、working tree、文件差异与服务器 HEAD，无未解释偏差。
- [ ] **P0 指标正确性**：旧 pseudo-F1 bug 的反例测试全部 PASS；分面积 pixel recall / object hit / global ObjPrecision、Recall、F1 语义分离。
- [ ] **数据正确性**：A/B 同名对应、label gray>=128、test 集 4000 张（以实时 list 数为准）、无 test 漏样、没有改阈值/增强。
- [ ] **正式结果复现**：原始 `train_log` 最后一完整 TEST RESULTS + 复算六指标对齐；部署图二值 disagreement 0。
- [ ] **D1 错误画像**：每个 GT object、每图、整体三层记录；M1/A2 同一对象 paired delta；错检/漏检绝对数。
- [ ] **D2 分阶段 hook**：shape/时相分半/score 数值正确；hook off/on 输出一致；raw score + native occupancy 双口径；DCR 单路不误做 cosine。
- [ ] **D3 若执行**：probe 只拟合 train，内部 val 固定，test 不调参；主模型冻结和 BN 不变。
- [ ] **D4 若执行**：ON/OFF 除 β 外完全等参，报告不得称从头训练消融。
- [ ] **统计与复现**：manifest/checksums、image-level bootstrap、全部 Gate、失败记录、实际命令与完整产物索引。
- [ ] **提交安全**：`git diff --cached --stat` / `git diff --cached --name-only` 查暂存区；不提交数据/权重/服务器输出/临时缓存/密钥；没有未经授权的覆盖删除。
- [ ] **最终交付**：实现代码 + 单测 + 运行命令 + `DIAGNOSIS_REPORT.md` + 统计 CSV/JSON + 图 + 一份非常明确的“下一轮做/不做/为什么”的裁决。

---

## 16. 参考的仓库材料（确保可溯源）

以下文档/文件属于本项目已有材料，执行时以源代码和真实日志为先：

- 主架构：[models/model/casa_tvim_str_net.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/models/model/casa_tvim_str_net.py)
- TinyViM-S-Slim：[models/model/tinyvim_s_slim.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/models/model/tinyvim_s_slim.py)
- SS2D：[models/model/layers/ss2d.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/models/model/layers/ss2d.py)
- CAACP：[models/model/layers/caacp_ss2d.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/models/model/layers/caacp_ss2d.py)
- 旧诊断 bug：[analyse/run2_zero_cost_diag.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/analyse/run2_zero_cost_diag.py)
- Run2 诊断：[docs/temporary/run2_zero_cost_diag.json](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/docs/temporary/run2_zero_cost_diag.json)
- Run3 报告：[docs/temporary/run3_report.md](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/docs/temporary/run3_report.md)
- Run1 汇总：[docs/temporary/CASA-TViM_Run1/metrics_tables.md](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/docs/temporary/CASA-TViM_Run1/metrics_tables.md)
- 训练协议、原始结果规范：[models/train.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/models/train.py) 与 [analyse/extract_metrics_to_excel.py](https://github.com/YuqiWang-code/CASA-CD/blob/fd8e0e3dee49ceffc4fbf9e2a3d6ada99be51f6f/analyse/extract_metrics_to_excel.py)
- 服务器数据契约：`docs/RSML-3_服务器环境与变化检测数据统一说明.md`。

**最后约束：用户给 DSH 的当前任务是“把问题诊断清楚”，不是“必须找到显著改进”。允许诊断后给出 `INCONCLUSIVE` 或认为研究动机需要修正。机制判断不能先于证据。**
