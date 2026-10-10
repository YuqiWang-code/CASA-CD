# Run5 — FHR-TAR（原生高频证据复用）执行索引

> **当前状态：S1 只读前测已完成，判定 `NO_80K`（`status=FAIL`，`val_screen_status=BOTH_SCREENED_OUT`）。**
> **Run5 正式训练次数 = 0**；`models/` 未被修改（与 Run4 后状态逐字节相同）；
> 本目录不含也不会补写训练脚本。判定依据见 §4，全部产物见 §5。
> 这与 Run5 设计文档 §11.1 步骤 3 的硬要求一致：`gate.json` 非 PASS ⇒ 不写训练脚本、不改 `models/`、不开 80K。

## 0. 冻结输入

| 项 | 值 |
|---|---|
| 设计文档 | `docs/temporary/CASA-CD_Run5_原生高频证据复用与结构改进实验设计_2026-10-10.md` |
| 设计文档 SHA256 | `8d721d77cf3a362c24687db54f20ecd1fad7bab05fd5b46f910f1501351239bc` |
| 冻结仓库版本 | `YuqiWang-code/CASA-CD@a6d49bc151674a93a8274aca52e2ca707f8d070a` |
| 前测脚本 | `analyse/tvim_run5_hf_pair_preflight.py`（`run5_hf_pair_preflight_v1`） |
| 前测输出根 | `/home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run5-HF-PREFLIGHT-20261010` |
| 代码/资产身份 | `train_scripts/CASA-TViM/Run5/SOURCE_IDENTITY.json`（服务器 `--require-all` 生成） |

## 1. S1 前测：唯一问题与唯一入口

**问题**：TinyViM 内部、`cat(x_low,x_high)` 与 `out_proj` **混合之前**的
`local_conv(x_high)`，对 M1 已漏检的 SYSU small 对象（1–255px）是否提供
**同尺度旧 tap（`L03b_norm2`/`L06b_norm4` 冻结 probe）之外**的增量证据，
且该证据对未匹配小预测连通域（FP-like）的响应显著更低。

两个候选插点**互斥，最终只选一个**：

| 候选 | 原生 HF 源 | shape 契约 | 目标 | 同尺度 tap | deploy Δparams | deploy ΔMAC |
|---|---|---|---|---|---|---|
| `Pair-2` | `encoder.network[2][-1].op.local_conv` | `2B×32×32²` | `TAR.stage2.temporal` | `L03b_norm2` | 6,144 | 6,291,456 |
| `Pair-3` | `encoder.network[4][-1].op.local_conv`（CAACP block） | `2B×84×16²` | `TAR.stage3.temporal` | `L06b_norm4` | 16,128 | 4,128,768 |

选择**只用 `train.txt` 的 D3 SHA1-hash 10% probe-val**；选择规则按 `val_novel_rescue` 降序 →
相差 ≤0.01 用 `val_gap` 降序 → 仍平局取 deploy MAC 更低者（Pair-3）→ 字典序。
锁定后 `selection_val.json[locked]=true`，**test 只评价一次、不参与选层**。

### 1.1 前测硬门（预注册，禁止事后回调）

| ID | 硬门 | 数值 |
|---|---|---|
| G0 | P0 身份 / 禁重拟合 / 已锁定 selected_pair / 输出目录全新 | SHA256 严格匹配 + `frozen` + `selection_from_train_val_only` |
| G1 | 全量 test、时间、bootstrap | `n_test=4000` 且 `elapsed<=600s` 且 `valid_bootstraps>=950/1000` |
| G2 | small 漏检对象有效数 | `>=100` |
| G3 | 未匹配小预测有效数 | `>=100` |
| G4 | 实际救回潜力 | `rescue_rate>=0.25`（Run4 前测为 0.5085，但已不足以开门） |
| G5 | 比假阳性更相关 | `gap>=0.15` **且** 图像级 bootstrap 95%CI 下界 `>0.03` |
| G6 | 对旧 tap 的独有信息 | `novel_rescue>=0.10` **且** CI 下界 `>0.02` |
| G7 | 同尺度 rank margin 增量 | `mean(margin_HF)−mean(margin_TAP)>=0.02` **且** 配对 CI 下界 `>0` |
| G8 | 两对只能选一对且不侵入 1/4 | `selected_pair∈{Pair-2,Pair-3}`，test 不切换，`no_new_train=true` |

任一 FAIL → `NO_80K`：**归档证据、不另选次优 Pair、不改阈值重跑**
（§11.2 唯一止损动作 = 只读 SYSU FP 空间画像）。

### 1.2 输出结构（`--out-dir` 必须为空）

```text
TViM-Run5-HF-PREFLIGHT-20261010/
  inputs.json  selection_val.json  gate.json  bootstrap.json  layer_diagnostics.json
  val_per_object.csv  test_per_object.csv  test_per_image.csv
  figures/  SHA256SUMS.txt
```

`--limit>0` 只用于链路自检，状态恒为 `DRY_ONLY`，**永不 PASS**。

## 2. 本地可跑的正确性护栏

```bash
python -m compileall -q analyse models train_scripts
python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v
python analyse/tvim_run5_hf_pair_preflight.py --help
```

`analyse/tests/test_tvim_run5_preflight.py` 覆盖纯逻辑：平均秩/逐图 rank 单调性、
环带 margin（含“其它 GT 对象不进环带”）、对象集合口径（missed_small / FP-like）、
逐图计数、三类图像级 bootstrap、§5.3.3 选层规则、gate 常数与预算冻结值。

## 3. 尚未执行（门 PASS 后才允许）

S2 按 §6 实装 `TemporalRep1x1` 零权重原生 HF 分支 → S3 T0–T11
（`models/test_run5_native_hf_tar.py`）→ S4 上传+SHA 对拍 → S5 `smoke_test.py` +
T0–T11 → S6 `models/run5_dry_run.py` → S7 显存探针 →
S8 两波 8×80K（波一 `M1_R5CTRL`，波二 `E7_FHR_TAR`）→ §9.4 收口裁决。

**本目录在门 PASS 之前不会出现 `_gen_scripts.py` / `run_all.sh` / `post_train.sh`。**

### 3.1 为什么没有执行（2026-10-11）

前测在 **val 初筛** 阶段即终止，未进入 test pass，因此 S2–S8 **全部未启动**：
`gate.json.selected_pair = null`、`test_pass_executed = false`。

## 4. Run5 前测结果（已完成，判定 `NO_80K`；训练次数 = 0）

### 4.1 执行记录

| 阶段 | 结果 |
|---|---|
| S0.1 环境 | torch 2.14.0+cu132 / CUDA 13.2 / 2×RTX 5090；`selective_scan_cuda_oflex` import PASS |
| S0.2 代码身份 | `make_source_identity.py --require-all` PASS（12 核心文件 + Run4 冻结文件 + Run5 新增 + tinyvim pth + 四库 list 全部有 SHA） |
| S1.1 单测 | `analyse/tests/test_tvim_*.py` 46 项 rc=0 |
| S1.2 P0 素材 | M1 best `best_F1=0.8347.pth` SHA256 == Diag1 锁；SYSU test list `sha256_text_lines` == Diag1 锁；两个 tap probe `encoder_input=concat`、`C_in=192/504` PASS |
| S1.3 DRY 排练（`--limit 600`） | `DRY_ONLY`，链路健康；排练先行捕获 1 个真实缺陷（见 §4.4） |
| **S1.4 正式前测** | **val 1212 张 + 统计，用时 26.8 s（预算 600 s）→ `status=FAIL` / `val_screen_status=BOTH_SCREENED_OUT` / `decision=NO_80K`；test pass 依 §5.3.3 未运行** |
| S1.4b 独立审计 | `INDEPENDENT_AUDIT.json`：**25/25 checks OK** |

### 4.2 判定证据：两个候选在 val 初筛即被排除

val = SYSU `train.txt` 的 D3 SHA1-hash 10% 固定子集（1212 张）；`n_missed_small=101`、
`n_fp_like=387`，**有效样本数门（≥30 / ≥50）通过**，故失败不是样本不足。
初筛门（§5.3.2，预注册）：`gap>=0.05` 且 `novel_rescue>=0.05`。

| 候选 | rescue(HF) | fp_like(HF) | **gap** | novel_rescue | tap_rescue（同尺度旧 probe） | 初筛 |
|---|---:|---:|---:|---:|---:|---|
| `Pair-2`（1/8 → `TAR.stage2.temporal`） | 0.2970 | 0.2636 | **0.0335** | 0.1683 | 0.2772 | ❌ `gap<0.05` |
| `Pair-3`（1/16 CAACP → `TAR.stage3.temporal`） | 0.0891 | 0.1137 | **−0.0246** | 0.0693 | 0.1089 | ❌ `gap<0.05` |

**`eligible = []` ⇒ 无合格候选 ⇒ §5.3.3「两个都 FAIL 则退出 `NO_80K`」，且不得对 test 做任何挑选。**

### 4.3 机制读数（val，同尺度 rank 口径）

| 量 | Pair-2 HF | Pair-2 TAP | Pair-3 HF | Pair-3 TAP |
|---|---:|---:|---:|---:|
| small margin | 0.00697 | 0.00685 | 0.00120 | 0.00086 |
| medium margin | 0.01343 | **0.07808** | 0.01737 | 0.02521 |
| large margin | **−0.01465** | **0.14478** | 0.02313 | 0.13069 |
| small-vs-bg AP | 0.01418 | 0.02056 | 0.00257 | **0.02132** |

- 在 **small** 尺度上，原生 `local_conv(x_high)` 与同尺度冻结 tap 几乎无差别（Pair-2 Δ=+0.0001，
  Pair-3 Δ=+0.0003），即 **native HF 对 small 对象并不比已有 tap 更可读**；
- 在 **medium/large** 尺度上 native HF 明显更差，尤其 1/8 的 large margin 为负；
- native HF 对 **未匹配小预测（FP-like）** 的响应与其对真实漏检对象的响应几乎相同
  （Pair-2：0.2636 vs 0.2970；Pair-3：0.1137 vs 0.0891）⇒ 它**不选择性编码漏检 small 变化**。
- 这与设计文档 §1.3 预注册的失败模式 **R1（`x_high` 主要是背景纹理/配准伪差）与 R2（与
  `norm2/norm4` 旧 tap 冗余）** 一致；`novel_rescue` 未为零（Pair-2 17/101）说明并非完全冗余，
  但**不足以**通过判别性（gap）门。

### 4.4 缺陷与隔离（不删除任何产物）

前测脚本首版存在一处 P1 缺陷：`missed_small` 只用 `r<0.25` 判定，漏掉 Run4 原有的
`bin_index(area) ∈ small` 组过滤，导致 medium/large 漏检对象被计入（1250 vs 正确 295）。
该缺陷由**独立最小复算**发现（复算精确重现 Diag1/Run4 冻结值 `n_small=364`、
pooled `0.19742166`、Hit@25 `0.18956044`、missed `295`）。修复后：

- 新增回归单测（medium/large 即使 `r<0.25` 也不得进入 `missed_small`）；
- `gate.json` 增加 `small_reference_reconciliation` 区块，把 `n_small_objects` /
  pooled recall / Hit@25 与 Diag1 冻结值直接对账，此类缺陷不可能再静默通过；
- 缺陷期与标签取代期产物以 `_INVALID_*` / `_SUPERSEDED_*` 后缀**改名保留**，未删除；
- 未采用失败口径下的任何数字作为结论。

### 4.5 未做的（明确记录）

- 未运行 G1–G8 的 test 评价（§5.3.3 禁止）；G5/G6/G7 阈值保持预注册值，未被修改；
- 未改 `models/`；未写训练脚本；未用 Run4 E6/CTRL 权重做任何 warm start；
- 未转试 `res`、未加 edge head、未改 3×3、未做多 seed/阈值/loss 挤分。

### 4.6 下一步（按设计文档 §8.4 D1 / §10，唯一立即动作）

只读 **SYSU FP 空间画像**：对预测 <256px 未匹配连通域统计
`(i)` 与 GT 边界距离 ≤4px 的比例、`(ii)` 变化图与 A/B 单时相边缘的重叠、
`(iii)` 组件面积/纹理能量、`(iv)` 与错位邻域关系，按**图像级 bootstrap** 报 CI；
**不增模型、不跑 80K**。若结论指向 1/16→1/32 表征尺度，则组织导师会审骨干级改动并**重新立项**
（需写明不再保证 ImageNet epoch0 对应的代价）。

## 5. 产物索引

```text
/home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run5-HF-PREFLIGHT-20261010/
  inputs.json                 sha256 608bce0a9306de0b94acd0752a5185270da834fd14488b2bc2176a8ceff9fbf0
  selection_val.json          sha256 f363cd3cd9abf24d9ce7a3a64517a88e78bb199b8e60591e26c98b41b5749d91
  gate.json                   sha256 a1cf4b2cc7c856a64e58d33300655b3ddee7fd7cc98d6a23e69a6ff424b9cd91
  bootstrap.json              sha256 d28e3ed7c73ed78e70673186d6baa3c6a8bc889ad964fb3dd1f24d8a1ddb21dc
  layer_diagnostics.json      sha256 31bd28f962ae1963e094745848f7a61c226578563cc300204e878545b3975ee8
  val_per_object.csv          sha256 38b66aa08ca12adfec73107604381f3ea1bed0baf84be3bb35c4944a72906479
  test_per_object.csv         sha256 ec6138c03bda9456e8050c5db3c833367e85b61c95d8551321a7a7214fe47215  (表头 only：test 未运行)
  test_per_image.csv          sha256 7ea26d4b340bd5f00afe9863b639e94300d6ca2f314f3043512173ab3998ef59  (表头 only：test 未运行)
  INDEPENDENT_AUDIT.json      sha256 efbbd211eff5905354340cd027b982ae46ce484f3b8fcb4038f51f4a1fdccb79  (25/25 OK)
  SHA256SUMS.txt
  _run.log（同级）             /home/yqwang/outputs/CASA-CD/diagnostics/TViM-Run5-HF-PREFLIGHT-20261010_run.log
```

决策锁：`train_scripts/CASA-TViM/Run5/SELECTION_LOCK.json`（`decision=NO_80K`、
`training_runs_executed=0`）。

`SOURCE_IDENTITY.json` 的**权威副本在服务器**（由 `--require-all` 生成，含预训练权重与四库 list 的
SHA256，`missing=[]`）；仓库内一份是本地生成物，因此 `missing` 只列出本地不存在的服务器资产
（`pretrained_weight/tinyvim_s_1000e.pth` 与 `/share_datasets/CD/**/list/*.txt`），
其 `files_sha256` 中的源码/新增文件条目与服务器逐条相同。
