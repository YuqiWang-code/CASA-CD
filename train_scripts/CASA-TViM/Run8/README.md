# Run8 — CASA-VMamba-STRNet（VMamba-Tiny + CAACP-HQKV + HGSL）执行索引

> 生成 2026-10-15。**用户裁定**：① deploy ≤5M 约束**解除**，改为「除 VMamba-Tiny 本身以外，
> 模型总体参数量尽量小」；② 接受分阶段、每阶段唯一变量、每阶段配同期 CTRL。
> 上游 Run1–Run7 全部完成（Run7 `MECHANISM-PASS`，winner `E9_HFAE_A`）。

---

## 0. 冻结输入（**开跑前不得更改**）

| 项 | 值 |
|---|---|
| 骨干 | **VMamba-Tiny**（`vssm_tiny_0230` 口径，**不是** `VSSM` 类默认值）：`depths=[2,2,5,2]`、`dims=[96,192,384,768]`、`ssm_ratio=2.0`、`ssm_d_state=1`、**无 z 门控（noz）**、`ssm_conv_bias=False`、`bias=False`、patch_embed v2（3→48→96，两个 3×3/s2 conv + LN2d）、downsample v3（3×3/s2 conv + LN2d）、`norm_layer=ln2d`、`mlp_ratio=4.0` |
| 骨干参数量 | **29,938,176（29.94 M）**，官方 ckpt 实测 `sum(numel)=30,705,832`（含 1000 类头 770,536） |
| 骨干权重 | `pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth`（sha256 `09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201`；2026-10-15 由用户授权从 STR-RepNet 复制） |
| scan 内核 | `selective_scan_cuda_oflex`（Run1–Run7 生产同款） |
| 新模块 | `CAACP-HQKV`：4 个 stage × 双分支（detail / global）× SAT 式 QKV |
| 新损失 | `HGSL`（λ=**0.2 冻结**，与 Run7 HFAE 同值） |
| 几何 | Lorentz 双曲，`κ=1`，`r0=0.5`、`a=0.5`（**冻结，禁止搜索**） |
| 判据 | `SELECTION_LOCK.json` 的 `acceptance_criteria_frozen_numeric` 是**唯一规范形式** |
| 协议版本 | `run8_vmamba_hqkv_exact80k_v1` |

> ⚠️ **参数口径警告**：项目旧文档写「官方 VMamba-T ≈30M」，但 `others/VMamba/vmamba.py::VSSM.__init__`
> 的**类默认值**是 `depths=[2,2,9,2]`/`d_state=16`/`patch v1`/`downsample v2`，按它搭出来是
> **42.95 M** 且**加载不了这枚权重**。本篇的 29.94 M 才是这枚 ckpt 的真实口径。
> 论文引用必须**连同上面那行配置一起标注**。

## 1. 机制一句话

**把 VMamba 四个 stage 的双时相输出映射到 Lorentz 双曲空间，用「局部离散度半径」与
「全局偏离半径」两条非恒定半径分别算出测地距离，据此在规则网格上把 K/V 压缩到 `k·N`
（Q 保持 `N`），两个分支各做特化 QKV，再把 4 个 stage 的增量加到 DCR 的 1/4 输入；
同时用测地分数本身做一条几何监督损失（HGSL）。**

### 1.1 为什么这**不是**第三次重做 CASAA（P0-3 陷阱①）

| 维度 | CASAA（Run1/Run3，终止） | CAACP-HQKV |
|---|---|---|
| 空间结构 | ViT 变长 token + 密度峰值聚类（含 `randperm` 随机采样） | **规则二维网格**，无随机采样 |
| 打分来源 | GT 占用率 / 单时相 rank | **双时相 Lorentz 测地距离 + 非恒定半径** |
| 分支数 | 1 | **2（各自独立 QKV）** |
| 注入位置 | 编码器第 8–11 层 | 编码器→解码器**跳层连接**（1/4，可切 1/16） |
| 权重公式 | 逐图 `rank` | **不用 rank**（见 §1.3） |

### 1.2 为什么这**不是**重复 Run7 E10（P0-3 陷阱②）

Run7 的 E10 是「把**边缘头**的特征零初始化注入 DCR 1/4 输入」，不优于 E9（只加损失）。
Run8 的注入对象是**变化感知聚合后的跨尺度增量**（不是边缘证据），且注入量由
`γ_detail/γ_global` 两门控制、门**零初始化**。**但这不是有效性证据**，只是机制不同；
`H1_HYP` vs `CTRL_R8` 就是为它准备的单变量检验。

### 1.3 为什么绕开了几何退化（P0-3 陷阱③）——**本轮最关键的一条**

固定半径下 `cosh d = 1 + sinh²r(1−cos)`，`acosh` 严格递增 ⇒ **与 `1−cos` 逐像素同序**
（`models/_legacy/test_run8_cpu_precheck.py` 的 C2 在真实张量上复现：`frac_rank_diff=0.0`）。

**更强的结论**：退化的根源其实是 `rank_normalize_2d` —— `rank(·)` 只保留序、丢弃一切幅度，
所以**任何保序变换**（不止固定半径双曲）走 rank 之后都得到逐位相同的权重。

⇒ Run8 的两条设计决策：
1. **取消逐图 rank**，让 Q/K/V 直接消费**数值分数**（softmax 对幅度敏感）；
2. **半径非恒定**：`r = r0 + a·q`，`q` 来自局部离散度（detail）或全图偏离（global）。

CPU 预检实测（随机张量）：`detail(变半径) vs 余弦` 的 `frac_rank_diff = 0.9785`；
`detail vs global` 两分支 `frac_rank_diff = 0.9746` ⇒ **非退化**。
⚠️ 同时实测 `Kendall τ(双曲, 同式欧氏) = 0.9983` —— **曲率本身带来的排序增量很小**，
`H2_EUC` 这条对照必须如实报告，不得因为 τ 高就淡化。

## 2. 四个变体与唯一变量

| 波 | 变体 | hqkv | geometry | hgsl | inject | 唯一变量 | 作用 |
|---|---|---|---|---|---|---|---|
| 1 | **`CTRL_R8`** | 0 | — | 0 | none | — | **同期对照**（VMamba + TAR/DCR） |
| 2 | **`H1_HYP`** | 1 | hyperbolic | 0 | dcr1 | 整个 HQKV 模块 | **主实验** |
| 3 | **`H2_EUC`** | 1 | euclidean | 0 | dcr1 | 仅几何 | **曲率必要性对照**（同式欧氏） |
| 4 | **`H3_HYP_GS`** | 1 | hyperbolic | 1 | dcr1 | 仅 HGSL | 新损失的增量 |

> `H1` vs `H2` 是**调研（2026-10-13 §6.3 / 2026-10-14 §11.2 G4）强制要求**的对照：
> 两者用**完全相同的 `r_A, r_B, u_A, u_B`**，只把测地距离换成切向欧氏距离。
> 若 `H1 ≈ H2`，结论必须写成「局部径向结构信息有用」，**不许**写成「负曲率有用」。

## 3. 参数预算（用户要求的"非骨干尽量小"）

实测（服务器 GPU 图，`models/_legacy/run8_dry_run.py` 会打印）：

| 组件 | 参数量 |
|---|---:|
| VMamba-Tiny 骨干 | **29,938,176**（29.94 M；官方 ckpt 口径） |
| CAACP-HQKV | **304,131** |
| TAR | 1,351,688 |
| DCR | 686,987 |
| head | 193 |
| **非骨干合计** | **≈ 2.34 M**（CTRL 为 2.04 M） |

QKV 之所以便宜：`V` 用 **per-channel depthwise 1×1**（`groups=rank`），
单分支参数从 `2·dim²` 降到 `3·dim·rank`；`dim=768, rank=32` 时 1.18M → 74K。

> ⚠️ **HQKV 保留在部署图里**（它是推理期真实算子）。`switch_to_deploy()` 只折叠 TAR/DCR。
> 因此**不得**声称 HQKV 零推理代价——这不是 Run7 的 `aux_only` 形态。

## 4. 执行顺序

```bash
cd /home/yqwang/projects/CASA-CD

# 0) 身份对拍（必须全 OK）
sha256sum -c train_scripts/CASA-TViM/Run8/SOURCE_SHA256.txt | grep -c ': OK'

# 1) 本地已跑：python models/_legacy/test_run8_cpu_precheck.py   （CPU，24/26 起）

# 2) 服务器 GPU 预检（smoke + dry-run，四库各一次 CTRL_R8 与 H1_HYP）
bash train_scripts/CASA-TViM/Run8/_r8_smoke.sh
bash train_scripts/CASA-TViM/Run8/_r8_dryrun.sh

# 3) 波一（对照，必须先跑完）
bash train_scripts/CASA-TViM/Run8/run_all.sh --wave 1
# 4) 波二 / 三 / 四
bash train_scripts/CASA-TViM/Run8/run_all.sh --wave 2
bash train_scripts/CASA-TViM/Run8/run_all.sh --wave 3
bash train_scripts/CASA-TViM/Run8/run_all.sh --wave 4
```

**排卡**：GPU1 被同用户另一项目占用（~24.5 GB / 100%），**只有 GPU0 可用** ⇒
`run_all.sh` 强制四库**串行**（Run7 波三曾因同卡 2 并发 OOM）。

## 5. 禁止项

1. **禁止搜索** `λ_HGSL`（0.2）、`r0`（0.5）、`a`（0.5）、学习率、增广、阈值、seed；
2. **禁止**用 Run7 的数字当同期对照（骨干不同，不可比，见 `SELECTION_LOCK.json:backbone_change.why_not_run7_comparable`）；
3. **禁止**修改 Run1–Run7 的文件（`ss2d.py` / `caacp_ss2d.py` / `tinyvim_s_slim.py` /
   `casa_tvim_str_net.py` / `str_hf_edge.py`）；
4. **禁止**在一次 campaign 中途改任何受哈希约束的文件（独立审计 A8 要求 16 个 run 代码身份唯一）；
5. **禁止**声称 HQKV「零推理代价」；
6. **禁止**在 `H1 ≈ H2` 时仍主张「负曲率必要」；
7. 产物只改名封存（`_SUPERSEDED_*` / `_INVALID_*`），**绝不删除**。

## 6. 结果归档

| 用途 | 路径 |
|---|---|
| 日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/` |
| dry-run | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8_DRY/<dataset>/<variant>/` |
| 正式指标 | 每个 `train_log.txt` 的**最后一个完整** `=== TEST RESULTS === … === END TEST RESULTS ===` |

## 7. 已知风险（开跑前冻结，事后不得淡化）

| # | 风险 | 现状 |
|---|---|---|
| R1 | Diag1 实测 CAACP β 修正只有 0.19%（M1），任何"只改 CAACP 内部打分"的方案增益上界≈0 | Run8 不用 rank、不用单 scalar β；但 **E9 的 β 比 M1 大 4 倍（SYSU 0.0224 vs 0.0056）**，不可外推 |
| R2 | **Kendall τ(双曲, 欧氏) = 0.9983** ⇒ 曲率增量可能极小 | `H2_EUC` 就是为抓这个而设 |
| R3 | 注入先例失败（Run7 E10 < E9，SYSU 小目标召回 −0.0351） | 机制不同≠有效；`H1 vs CTRL` 检验 |
| R4 | CASAA 终止结论「routing ranking ↑ ≠ F1 ↑」 | 上文 §1.1 列了 5 条差异，但**未被证伪** |
| R5 | 文档写「官方 VMamba-T ≈30M」，实测 42.95M | **已解决**：真实口径 = 29.94 M（见 §0 配置行）；旧值来自 `VSSM` 类默认值，已作废 |
| R6 | GPU1 被占，只剩 GPU0 | **实测 batch32 peak 26.9–27.7 GB**（余量仅 ~5 GB）⇒ 四库**必须串行**，不可同卡 2 run |
| R7 | 折叠门非零 disagreement | 实测 1–3 个像素、margin ≤ 3e-7、max_abs_diff ~1e-6 ⇒ 全部为 **`KNIFE_EDGE_ARTIFACT`**（Run7 预注册 §7.6 已披露、分析工具判据同定义）。**报告里必须写明走的是这条判据，不是裸硬门 0** |

## 8. 收口 runbook（四波跑完后）

1. 逐 run 硬门 + 对象级复算 + 裁决 + 独立审计（照 Run7 `finish_run7.sh` 的六步链结构移植）；
2. **必须有同期 CTRL**，不得用 Run7 数字；
3. 五种 verdict 写法照 Run7 §8.4；**负结果照写**；
4. 本地归档：下载 16 份日志 → 重建 `docs/experiment_metrics.xlsx`（先备份 + 断言无丢行）→
   生成 `docs/temporary/models_and_metrics_CASA-VMamba_Run8.txt` 快照；
5. 更新根 `README.md` 的 Run8 结果节 → `git commit -F <文件>` → `git push origin main`。
