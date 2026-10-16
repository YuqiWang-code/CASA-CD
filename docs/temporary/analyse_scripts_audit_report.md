# CASA-CD `analyse/` 全量脚本审计报告

**范围**：`F:\Code_Repositories_2\CursorCode\CASA-CD\analyse\**\*.py` 共 **41 个文件**（36 个顶层脚本 + 5 个 `analyse/tests/` 测试），已逐个完整读取（跳过 `__pycache__`/`.pyc`）。
**现场事实核对**：`outputs/` 下存在 **92 个** `train_log.txt`；`docs/experiment_metrics.xlsx` 存在；`docs/temporary/` 存在 11 个 `models_and_metrics_*.txt` 快照（baseline_Run1、CASA-STR_Run1、CASA-TViM_Run2/Run3/Run7、CASAA_Run1/Run2、STR-Fusion_Run2_TASS、UltraLight_Run4/Run5/Run6）。

**三条必须先纠正的任务预设**（读码后确认，非猜测）：
1. `tvim_object_metrics.py` **不使用 8 连通**：主口径 = **4 连通**（`connectivity=4`），8 连通只作为一次性敏感性（`connectivity_sensitivity: 8`）。
2. `tvim_object_metrics.py` **不含 KNIFE_EDGE_ARTIFACT 逻辑**（文件内 grep 无此项）。该裁决定义在 `run4_fold_knife_edge.py` 与 `run7_fold_knife_edge.py` 中。
3. `extract_metrics_to_excel.py` **没有 append 语义、没有 backup、没有 "assert no rows lost"**：每次 `Workbook()` 新建工作簿覆盖 `docs/experiment_metrics.xlsx`（详见 §1.3）。

---

## 1. 重点文件深读

### 1.1 `analyse/tvim_object_metrics.py` —— 权威对象级指标定义（28,834 B / 573 行，纯 numpy+scipy，不 import torch）

**定位**：修正 `run2_zero_cost_diag.py::component_pr` 的评价语义缺陷。旧实现的 `FP = ((pred>0) & (gt==0) & m).sum()` 中 `m ⊆ (gt>0)`，故 **FP 恒为 0**，得到的是「受限正样本组内的 pseudo-Precision/pseudo-F1」，不是对象级 P/R/F1（模块开头 docstring 与 `METRICS_PROTOCOL["note"]` 明确记录）。

#### (a) 连通域 / 对象如何形成
- **结构元**（verbatim）：
  - `CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)`
  - `CONN8 = np.ones((3, 3), dtype=bool)`
- `label_components(mask, connectivity=4)`：`ndimage.label(mask.astype(bool), structure=CONN4 or CONN8)`；返回 `(labels, n)`。`METRICS_PROTOCOL["connectivity_primary"] = 4`，`["connectivity_sensitivity"] = 8`。
- **没有最小面积过滤**：`area = int(counts[j])`，1 像素对象照常计入（测试 `test_05` 用 `pred[20,20]=True` 单像素命中 large GT；`test_04` 用 area=16 的块）。
- **不忽略边界**：贴边对象照常计入；`touches_border` 只作为元数据记录：
  `bool(y0 == 0 or x0 == 0 or y1 == gt_b.shape[0] or x1 == gt_b.shape[1])`。
- 面积一律在**原生 GT 分辨率**（Diag1 场景为 256×256）上按上述连通性计算（模块 docstring §A）。

#### (b) 面积分组（GT 对象分层）
```python
AREA_EDGES = [1, 16, 64, 256, 1024]          # 左闭右开
AREA_GROUPS = ["tiny_1_15", "tiny_16_63", "small_64_255", "medium_256_1023", "large_1024_inf"]
GROUP_ALIAS = {"small":  ["tiny_1_15", "tiny_16_63", "small_64_255"],   # 即 1 <= area < 256
               "medium": ["medium_256_1023"],                            # 256 <= area < 1024
               "large":  ["large_1024_inf"]}                             # area >= 1024
```
- `bin_index(area)`：`a<16→tiny_1_15`、`a<64→tiny_16_63`、`a<256→small_64_255`、`a<1024→medium_256_1023`、否则 `large_1024_inf`；**`a < 1` 抛 `ValueError(f"invalid area {area}")`**。
- 旧分组口径对齐注释：`1<=a<256 / 256<=a<1024 / a>=1024`。

#### (c) 预测对象 ↔ GT 对象如何匹配
- `object_iou_matrix(pred, gt, connectivity=4)`：逐图构造 GT×Pred 的 `iou (n_gt × n_pred)`；用 `ndimage.find_objects` 的 bbox 交集做剪枝；对每个有交集的 (i,j)：`inter = |{gt_lab==i} ∩ {pr_lab==j}|`，`union = gt_areas[i-1] + pr_areas[j-1] - inter`，`iou[i-1,j-1] = inter/union`。
- `match_objects(...)`：**一次性最大权重二分匹配**，不是贪心、不是质心匹配：
  ```python
  rows, cols = linear_sum_assignment(-iou)
  ```
  匹配在**全 IoU 矩阵**上完成（与阈值无关），随后逐对按阈值判定：
  - `IOU_LOOSE = 0.10` → `n_matched_loose`（并把该 pred 索引加入 `matched_pred`）
  - `IOU_STRICT = 0.50` → `n_matched_strict`
  即：一个低 IoU 的对也会占用一行一列，之后才被阈值过滤——这是"先分配后阈值"的语义，报告口径必须按此描述。
- 协议字符串（`METRICS_PROTOCOL`）：
  - `"matching": "scipy.optimize.linear_sum_assignment on -IoU (one-shot max-weight bipartite)"`
  - `"iou_loose": 0.10`、`"iou_strict": 0.50`
  - `"threshold_rule": "pred_bool = (prob > 0.5)  # 与 models/train.py 一致（严格大于）"`
- `match_objects` 还返回：`n_gt`、`n_pred`、`iou_pairs_loose`、`pred_areas`、`gt_areas`、`unmatched_pred_areas`、`matched_gt_idx`、`matched_pred_idx`、`fp_pixels = |pred & ~gt|`、`pred_pixels`、`gt_pixels`、`iou_max`。
- 边界情形：`n_gt==0 or n_pred==0` → 不做分配，`unmatched_pred_areas = 全部 pred_areas` 直接返回。

#### (d) 精确公式
**像素级全图六指标** `_counts_to_scores(tp, fp, fn, tn)`，`eps = float(np.finfo(np.float32).eps)`（= 1.1920929e-07）：
```
OA        = (tp+tn) / (tp+fn+fp+tn + eps)
recall    = tp / (tp+fn + eps)
precision = tp / (tp+fp + eps)
F1        = 2*recall*precision / (recall+precision + eps)
IoU       = tp / (tp+fp+fn + eps)
pre       = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / (tp+fp+tn+fn)**2
Kappa     = nan  if (1.0 - pre) == 0.0  else (OA - pre)/(1 - pre)
```
- `total == 0`（如空 boundary band）→ 六指标全部 `float("nan")`（注释：「指标无定义 → nan，不臆造 0」）。
- `Kappa` 显式走「与 metric_tool 同式（无 eps）：单类退化时 0/0 → nan（numpy 语义）」。

**GT 面积分层对象级指标**（`groups[g]`，`n = n_objects`）：
```
pixel_recall_micro        = tp_pixels / n_pixels              (n==0 → None)
object_macro_pixel_recall = macro_recall_sum / n                (n==0 → None)
hit1                      = 命中数(tp >= 1) / n                  (n==0 → None)
hit25                     = 命中数(r >= 0.25) / n                (n==0 → None)
hit50                     = 命中数(r >= 0.50) / n                (n==0 → None)
n_missed                  = 对象数(tp == 0)      # 即 hit1 的补集
n_pixels / tp_pixels / n_objects / n_images_with_group
```
其中单对象 `r = tp / area`。分层只报对象级 Recall（`hit*` 与 macro pixel recall），**ObjPrecision 只报数据集级**（模块 docstring §B）。

**对象级 P/R/F1**（`summary()["object"]`，`op`=ObjPrecision，`or`=ObjRecall）：
```
ObjPrecision_loose = n_matched_loose / n_pred     (n_pred==0 → None)
ObjRecall_loose    = n_matched_loose / n_gt       (n_gt==0   → None)
ObjF1_loose        = 2*op*or/(op+or)  if (op and or) else None
ObjPrecision_strict / ObjRecall_strict / ObjF1_strict  同式，用 n_matched_strict
unmatched_pred_loose = n_pred - n_matched_loose
fp_pixels
pred_cc_count_by_area / unmatched_pred_cc_count_by_area / gt_cc_count_by_area  (按 AREA_GROUPS 直方图)
```

**边界带** `boundary_band(gt, r, structure=CONN4)`（verbatim 协议）：
```
band_r = binary_dilation(gt, st5, iterations=r, border_value=0)
       & ~binary_erosion(gt, st5, iterations=r, border_value=0)
```
- `st5 = 4-邻接 structuring element [[0,1,0],[1,1,1],[0,1,0]]`；`border_value=0` ⇒ **带不能超出图像**。
- 同时记录 `gt_border_pixels`（首末行 + 首末列上的 GT 像素数：`gt[0,:] + gt[-1,:] + gt[:,0] + gt[:,1]`）与 `band_pixels`、`radius`，用于说明贴边截断。
- `band_metrics`：band 内 `tp/fp/fn`，`tn = band.size - band.sum()`，`empty_band = (band_pixels == 0)`；空 band 的六指标为 `nan`。
- `ObjectMetricAccumulator` 默认 `bands=(2, 4)`，即 band2 与 band4 两个半径，并给出 `group_tp_fn`（各面积组 band 内/外的 TP/FN）。
- 数据集汇总里 band 用 `_counts_to_scores(tp, fp, fn, band_pixels - tp - fp - fn)`。

#### (e) 其它数值常量与口径（verbatim）
- `"empty_group_policy": "None (NA), macro 分母不含空组"`
- `"empty_image_policy"`：
  - 无 GT 且无 Pred → 不计入 `N_gt/N_pred/N_matched`
  - 有 Pred 无 GT → 计入 `N_pred` 且全部 `unmatched`（FP）
- `"version": "tvim_object_metrics/v1 (Diag1)"`
- 形状/形态分层（仅 `_enrich_rows`，用于 D1 明细）：
  - `bbox_aspect_ratio = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))`
  - `bbox_min_width = min(bbox_w, bbox_h)`；`width_stratum = "<4" if min<4 else ("4-7" if min<8 else ">=8")`
  - `elongated = int(bbox_aspect_ratio >= 3.0)`
  - `density_stratum = "0-1%" if ratio<=0.01 else ("1-5%" if ratio<=0.05 else ">5%")`，其中 `ratio = gt_positive_pixels / n_pixels`（图像级 GT 面积占比）
  - `occupancy = area / max(1, bbox_area)`
  - `perimeter_approx` = 4 邻接边界边数（对象上下左右与背景相邻的边计数）
  - `matched_iou_010 = int(j in matched_gt)`；`matched_iou_050 = int(iou[j-1].max() >= 0.50)`；`iou_max_any_pred`
  - `model_p_mean_in_gt / model_p_max_in_gt / model_p_p90_in_gt`（`np.quantile(vals, 0.90)`）
- `aggregate_summaries(items, label="")` **未实现**：`raise NotImplementedError("use ObjectMetricAccumulator directly for aggregation")`。
- 累加器保证与图像顺序/batch size 无关（全部 macro 用 `sum + count`），并提供 `per_image_metric(key)` 供 bootstrap 使用。
- **KNIFE_EDGE_ARTIFACT 不在本文件**：定义见 §1.6 与表格中 `run4_fold_knife_edge.py` / `run7_fold_knife_edge.py`。

---

### 1.2 `analyse/tvim_diag_common.py` —— Diag1 共享基础设施（30,991 B / 692 行）

**路径与数据集注册表**（全部可由环境变量覆盖，verbatim 默认值）：
```python
PROJECT      = os.environ.get("CASA_PROJECT", "/home/yqwang/projects/CASA-CD")
DATA_ROOT    = os.environ.get("CASA_DATA",    "/share_datasets/CD")
CKPT_ROOT    = os.environ.get("CASA_CKPT",    "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
OUTPUT_ROOT  = os.environ.get("CASA_OUTPUT",  "/home/yqwang/outputs/CASA-CD")
PRETRAIN_TINYVIM = <PROJECT>/pretrained_weight/tinyvim_s_1000e.pth
DIAG_BASE        = <OUTPUT_ROOT>/diagnostics/TViM-TinyLoss-Diag1
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
log_path_for(run, variant, dataset) = <OUTPUT_ROOT>/CASA-TViM/<run>/<variant>/<dataset>/train_log.txt
```
`VARIANT_CFG`（9 个变体 → `CASATViMSTRNet` 构造实参，`str_dim` 固定 96）：
| 变体 | caacp | rep_mode | caacp_score_mode | frh | caacp_residual_mode | fs_tar |
|---|---|---|---|---|---|---|
| M1_FULL | True | full | rank | False | current | False |
| A1_CAACP | True | plain | rank | False | current | False |
| A2_STR | False | full | rank | False | current | False |
| A0_TVIM_PLAIN | False | plain | rank | False | current | False |
| E1_CP_CAACP | True | full | cp | False | current | False |
| E2_FRH | True | full | rank | True | current | False |
| E3_CP_FRH | True | full | cp | True | current | False |
| E4_RA_CAACP | True | full | rank | False | avg_anchor | False |
| E5_FS_TAR | True | full | rank | False | current | True |

eval 变换 6 通道统计（`train.py` 默认，ImageNet×2，BGR 顺序）：
`MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]`，`STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]`。

**日志解析（哪些行 / 哪些正则）**：
- `parse_test_block(log_path) -> (block_text, info_dict, lineno_range)`，只读**最后一个完整 TEST 区块**。
- `_extract_test_block`（与 `extract_metrics_to_excel.py` **逐字一致**）：
  `re.findall(r"=== TEST RESULTS ===\n(.*?)\n=== END TEST RESULTS ===", text, re.S)`，取 `blocks[-1]`。
- `_parse_block`（逐字一致）：六指标行
  `Recall=([0-9.]+)\s*\|\s*Precision=([0-9.]+)\s*\|\s*OA=([0-9.]+)\s*\|\s*F1=([0-9.]+)\s*\|\s*IoU=([0-9.]+)\s*\|\s*Kappa=([0-9.]+)`
- 参数/FLOPs 解析优先级（与 xlsx 脚本同）：
  - `Params(M)`：`[EFFECTIVE-PARAMS]` → `[DEPLOY-PARAMS] .*effective=` → `[TOTAL-PARAMS|PARAMS|DEPLOY-PARAMS|TOTAL-TRAIN-GRAPH-PARAMS]`
  - `Trainable(M)`：`[TRAINABLE-PARAMS]` → `[DEPLOY-PARAMS] .*trainable=`
  - `FLOPs(G)`：`[FLOPS|DEPLOY-FLOPS]`
  - `Mode`：`[MODE]` → `[ARCH]`
- 额外字段正则（仅 `parse_test_block` 有）：
  `[BEST-F1]`、`[PRETRAIN-SHA256]`、`[CAACP-BETA]`、`[REPARAM-MAX-ABS-ERROR]`、`[REPARAM-ARGMAX-DISAGREE]`、`[REPARAM-REAL-MAX-ABS-ERROR]`、`[REPARAM-REAL-ARGMAX-DISAGREE]`
- `span = (starts[-1] + 1, ends[-1] + 1)`，行号 1-based，用于报告溯源。
- 为什么重新实现：服务器 `casacd` 未装 `openpyxl`；一致性由 `tests/test_tvim_log_parser.py` 对拍官方实现保证。

**checkpoint 选择（唯一 best，禁止按字典序猜）**：
- `pick_best_ckpt(run, variant, dataset, ckpt_root=None)`：`sorted(glob(d + "/best_F1=*.pth"))`
  - 0 个 → `FileNotFoundError("[BLOCKED-CKPT] no best_F1=*.pth in ...")`
  - \>1 个 → `RuntimeError("[BLOCKED-CKPT] ambiguous best checkpoints ...")`
  - 返回 `(path, meta)`，`meta` 含 `ckpt_dir / candidates / path / sha256 / arch / run_manifest`。
- `verify_arch_sidecar(variant, meta, dataset)`：老 `arch.json` 只有 5 字段（`arch, backbone, caacp, rep_mode, str_dim`），Run2/Run3 起才有 `caacp_score_mode / caacp_residual_mode / frh / fs_tar`；缺失字段按当时默认值 `{"caacp_score_mode":"rank","caacp_residual_mode":"current","frh":0,"fs_tar":0}` 补齐后比较；有 mismatch → `FAIL`，无 `arch.json` → `WARN`。

**模型构造 / 数值纪律**：
- `build_model(variant, device="cuda:0", ckpt_path=None, pretrain=None, strict=True)`：`CASATViMSTRNet(pretrain, caacp=, rep_mode=, str_dim=96, caacp_score_mode=, frh=, caacp_residual_mode=, fs_tar=)`；`load_state_dict(..., strict=False)` 后若 strict 且 missing/unexpected 非空 → `RuntimeError("[D0] state_dict mismatch: ...")`；随后 `model.eval()` + 全部 `requires_grad_(False)`。
- `set_eval_numerics()`：`cudnn.allow_tf32=False`、`cuda.matmul.allow_tf32=False`、`cudnn.deterministic=True`、`cudnn.benchmark=False`。
- `make_loader(dataset, list_name="test", batch_size=16, num_workers=4, size=256, mean=None, std=None, shuffle=False)`：变换链 `Normalize(MEAN,STD) → Scale(256,256) → ToTensor`；list 路径 `<DATA_ROOT>/<dataset>/list/<list_name>.txt`；`pin_memory=True, drop_last=False`。

**复用的指标定义（与 `models/model/metric_tool.py` 完全同口径）**：
- `EPS32 = float(np.finfo(np.float32).eps)`
- `cm_to_counts(confusion)`：`confusion` 2×2（rows=gt, cols=pred）→ `tp=c[1,1], fn=c[1,0], fp=c[0,1], tn=c[0,0]`
- `counts_to_scores(tp,fp,fn,tn)`：与 §1.1 同式，**额外返回 `Pre`**，且 `Kappa = (oa-pre)/(1-pre)` **无 nan 保护**（严格复刻 metric_tool.cm2score）。
- `confusion_from_bool(pred_bool, gt_bool)` → `np.array([[tn, fp], [fn, tp]])` int64。
- `bootstrap_ci(values, n_boot=1000, seed=16, alpha=0.05)`：先剔除非有限值；`rng.integers(0, v.size, v.size)`；`np.quantile(means, [alpha/2, 1-alpha/2])`；返回 `{mean, lo, hi, n}`。
- `paired_bootstrap_ci(a, b, ...)`：断言同形状，对 `a-b` 做图像级 bootstrap。
- `code_identity(project=PROJECT)`：对 `DIAG_CODE_FILES`（12 个文件：`models/model/casa_tvim_str_net.py`、`tinyvim_s_slim.py`、`layers/caacp_ss2d.py`、`layers/ss2d.py`、`str_tar.py`、`str_dcr.py`、`str_reparam.py`、`metric_tool.py`、`dataset/dataset.py`、`dataset/Transforms.py`、`train.py`、`eval.py`）逐个 `sha256_file`；有 `.git/HEAD` 则记录其内容（服务器工作副本非 git 仓库，故以文件哈希为准）。
- `sha256_file`（1 MiB 分块）；`sha256_text_lines`（逐行 `rstrip("\r\n")` + 补 `\n`，避免 CRLF 差异）——**list 锁用的是后者**。
- `env_info()`：python/torch/cuda_build/cudnn/gpu_count/gpu_names/platform/time + 可选 scipy/sklearn/cv2 版本。
- `write_json`（`indent=2, ensure_ascii=False, default=float`）；`write_gate(out_dir, stage, status, checks, extra=None)` → `gate.json`，`status ∈ {PASS, WARN, FAIL, BLOCKED, NOT_RUN}`（assert 校验）。

**`run_audit(args)`（D0-REPRO 子命令）**：
- 复算全测试集六指标并与日志对拍：`deltas[lk] = abs(logged[lk] - repro[mk])`，`max_delta`，`repro_ok = max_delta <= args.tol`（默认 `--tol 1e-4`）。
- 折叠等价性：`torch.manual_seed(16)`，随机 batch `--fold-batch 16`（3×256×256）+ 真实 batch；`_fold_stats` 给 `max_abs_error / binary_disagreement / n_elements / n_flipped`；注释 `"disagreement 分母 = N×H×W（全部元素）；hard gate = 0"`。
- 写 `manifest.json`、`test_result.json`、`fold_equivalence.json`。
- 9 项 checks：`checkpoint_unique / state_dict_strict / arch_sidecar / log_test_block_found / reproduce_within_tol / reproduce_max_abs_delta / fold_random_disagreement_zero / fold_real_disagreement_zero / no_nonfinite / n_images_seen_full`（9 项 + 记录值）。
- `_deploy_flops`：fvcore `flop_count(model, (pre, post, label))`，2×3×256×256 + 1×1×256×256 label。
- `protocol` 子命令：把 `tvim_object_metrics.METRICS_PROTOCOL` 落盘为 `<out-dir>/metrics_protocol.json`。

---

### 1.3 `analyse/extract_metrics_to_excel.py` —— TEST 指标 → Excel（6,067 B / 166 行）

**输入**：
- 无 argparse、无环境变量。常量：
  `ROOT = dirname(dirname(abspath(__file__)))`、`OUTPUTS = <ROOT>/outputs`、`DOCS = <ROOT>/docs`、`OUT_XLSX = <ROOT>/docs/experiment_metrics.xlsx`。
- 扫描：`sorted(glob(os.path.join(OUTPUTS, "**", "train_log.txt"), recursive=True))`。
- 只读每个 log 的**最后一个完整** `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块（正则同上）；**明确不用 val-best 行、也不用 checkpoint 文件名**（docstring）。
- 跳过条件并打印：无 TEST 区块 → `[skip] {rel}: no TEST RESULTS block`；无六指标行（`"F1" not in info`）→ `[skip] {rel}: no metric line`；成功 → `[ok] {rel}: F1={F1*100:.2f}`。

**输出与 schema**（`docs/experiment_metrics.xlsx`，工作表名 `"metrics"`）：
```
表头：Tag | Run | Experiment | Dataset | Recall | Precision | OA | F1 | IoU | Kappa | Params(M) | Trainable(M) | FLOPs(G) | Mode
```
- `METRIC_KEYS = ["Recall", "Precision", "OA", "F1", "IoU", "Kappa"]`；输出 **×100 并 `round(..., 2)`**（注释：`0.9877 -> 98.77`）。
- `Params(M) / Trainable(M) / FLOPs(G)` 原样写入（不 ×100），单元格 `number_format = "0.00"`（列 K/L/M）；六指标列 E–J 也是 `"0.00"`。
- 表头 `Font(bold=True)`；`ws.auto_filter.ref = ws.dimensions`；`ws.freeze_panes = "A2"`；列宽 `A10 B8 C14 D16 E10 F10 G8 H8 I8 J8 K10 L12 M10 N10`。
- `parse_path(rel)`：`dataset = parts[-1]`；`rest = parts[:-1]`；`tag = rest[0]`；`rest[1:]` 中匹配 `^Run\d+(_\w+)?$` 者为 `run`，其余以 `"/"` 连接为 `experiment`。例：`baseline/Run1/WHU-CD-256 → (baseline, Run1, "", WHU-CD-256)`；`CASA-CD/Run1/A0_Plain/CDD-CD-256 → (CASA-CD, Run1, A0_Plain, CDD-CD-256)`。
- 若一行都没解析到：仍写出**仅表头**的空表，并打印「没有找到任何 TEST RESULTS 区块；已写入空表头」。

**写入语义（关键）**：
- `wb = Workbook(); ws = wb.active; ws.title = "metrics"` —— **每次运行新建工作簿**，`wb.save(OUT_XLSX)` 直接覆盖。
- **没有 append 到已存在工作簿的逻辑；没有备份（无 .bak/时间戳副本）；没有 “assert no rows lost” / 行数守恒断言**；也不按 (Tag,Run,Experiment,Dataset) 去重或对比历史行。
- 后果：若某次运行 run 目录暂时不可见（挂载失效、log 被截断），该行会**静默消失**，且不会报错、不会保留旧值。这是当前实现与「append + 防丢行」预期的真实差距。
- 依赖：模块顶层 `from openpyxl import Workbook / from openpyxl.styles import Font` ⇒ 未装 openpyxl 的环境**无法 import 本模块**（这正是 `tvim_diag_common` 复制正则的原因）。

---

### 1.4 `analyse/models_to_txt.py` —— models 源码 + 指标快照（3,355 B / 89 行）

**CLI**：`--tag`（default `"baseline"`）、`--run`（default `"Run1"`）；无其它参数。
**输出**：`docs/temporary/models_and_metrics_<tag>_<run>.txt`（`DOCS_TMP = <ROOT>/docs/temporary`，`os.makedirs(..., exist_ok=True)`）。
- 注意 docstring 称「每次运行新建，不覆盖旧文件」，但代码是 `open(out_path, "w")` ⇒ **同一 (tag, run) 重跑会覆盖同名文件**；"不覆盖"只在 tag/run 不同（文件名不同）时成立。已存在的 11 个快照文件印证了 `<tag>_<run>` 命名约定。

**抓取来源（实测，与任务预设不同）**：
1. **model 源码**：`os.walk(<ROOT>/models)`，`SKIP_DIRS = {"__pycache__", "build", "csrc", ".git"}`，`SKIP_EXTS = {".pth", ".pt", ".pyc"}`，只收扩展名 `.py/.yaml/.yml`，按 `sorted(names)`；相对路径以 `models/` 为基准。
2. **指标**：`sorted(glob(<ROOT>/outputs/**/train_log.txt, recursive=True))`，复用 `from extract_metrics_to_excel import extract_test_block, parse_block, METRIC_KEYS`（因此同样需要 openpyxl）；只保留含 `F1` 的条目。
3. **不抓取 README，也不读 `docs/experiment_metrics.xlsx`**。

**输出结构**：
```
# CASA-CD models snapshot + metrics
# tag=<tag>  run=<run>

================================================================================
## MODELS CODE (<N> files)
================================================================================
================================================================================
# FILE: models/<rel>
================================================================================
<文件原文>
... (每个文件一段)
================================================================================
## METRICS (<M> results)
================================================================================
Run | Recall | Precision | OA | F1 | IoU | Kappa | Params(M) | Trainable(M) | FLOPs(G) | Mode
<rel> | <k:.4f> ... | <Params:.3f> | <Trainable:.3f> | <FLOPs:.4f> | <Mode>
```
缺失值统一写 `-`（六指标 `.4f`，Params/Trainable `.3f`，FLOPs `.4f`）。
末尾 `print`：`已写入 <path>`，并 `代码文件 <N> 个，指标 <M> 行`。

---

### 1.5 `analyse/tests/` —— 逐测试函数与它守卫的不变量

运行方式（各文件 docstring）：`python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v`。

#### 1.5.1 `tests/test_tvim_object_metrics.py`（17,610 B / 376 行，13 个 test 方法）
`class TestObjectMetrics`（全部人造小阵列 + 手算预期）：
| 测试 | 守卫的不变量（引关键断言） |
|---|---|
| `test_01_area4_full_hit` | 2×2（area 4）完整命中：`n_objects==1`、`n_pixels==4`、`pixel_recall_micro==1.0`、`object_macro_pixel_recall==1.0`、`hit25==1.0`、`hit1==1.0`、`n_matched_loose==1`、`n_matched_strict==1`（"IoU=1 >= 0.5"）、`ObjF1_strict≈1.0`；像素 precision/recall `assertAlmostEqual(..., 1.0, delta=1e-6)`（因 metric_tool 口径含 float32 eps） |
| `test_02_area4_total_miss` | 漏检：`pixel_recall_micro==0.0`、`object_macro_pixel_recall==0.0`、`hit1==0.0`、`hit25==0.0`、`n_missed==1`；**无预测时 ObjPrecision 为 `None` 而不是 0**（`assertIsNone(ObjPrecision_loose/_strict)`），`ObjRecall_loose==0.0` |
| `test_03_fp_reduces_precision_regression_old_bug` | **旧 `component_pr` FP 恒为 0 的直接回归测试**：加 3 个 2×2 背景 FP 后，GT 组 recall/hit25 不变，但全图像素 `precision` 必须下降、`fp` 必须上升，且对象级 `ObjPrecision_loose` 必须下降并 `> 0.0`（"仍检出 1/4"） |
| `test_04_two_gt_one_pred_cc` | 一个预测连通域跨接两个 GT（area 各 16）：`n_gt==2`、`n_pred==1`、`n_matched_loose==1`、`ObjRecall_loose≈0.5` ⇒ 一对一匹配至多命中一个 |
| `test_05_large_gt_one_pixel_hit` | large（40×40=1600）：单像素命中 ⇒ `hit1==1.0` 但 `hit25==0.0`（证明 Hit@1 与 Hit@25 语义不同），`pixel_recall_micro≈1/1600` |
| `test_06_area_bins_and_empty_spec` | 分箱边界（`bin_index(15/16/63/64/255/256/1023/1024)`）；面积 256 → medium；**空组为 None 且不参与 macro**；全空图：`n_images==2`、`empty_both_images==2`、`ObjPrecision/ObjRecall/ObjF1` 全 `None`、`OA≈1.0`（含 eps 口径）、`Kappa` 为 `nan`；有 Pred 无 GT：`n_gt==0`、`n_pred==1`、`n_matched_loose==0`、`unmatched_pred_loose==1`、`pixel.fp==9` |
| `test_07_connectivity_diagonal` | 对角两像素：`n4==2`、`n8==1`；累加器主口径 4 连通时 `n_gt==2`，8 连通时 `n_gt==1` |
| `test_08_background_fp_not_filtered_by_gt_mask` | 背景 FP 不得被 GT mask 过滤：累计 6 张后 `fp` 上升，但 GT 组 `pixel_recall_micro` 保持 `1.0`；`band2.fp >= 0` |
| `test_09_torch_numpy_cpu_gpu_consistency`（`@skipUnless(HAS_TORCH, "torch unavailable (local Windows env)")`） | torch bool CPU/CUDA 与 numpy uint8 输入下，`pixel.tp/fp`、`object`、`groups` 与基线**逐字段相等** |
| `test_10_batch_and_order_invariance` | batch size 1/4/16 与逆序 7 下 `pixel/groups/object` 全部在 `1e-9` 内一致（`_close` 递归比较，`nan==nan`）；整数计数严格相等（`tp/fp`、`object`）；`n_images==16` |

`class TestBoundaryBand`：
- `test_band_definition_and_empty`：`band_pixels == int(band.sum())`、`gt_border_pixels==0`；`band[10,8]` 为真（距 GT 2px）、`band[10,7]` 为假（3px）、`band[17,17]` 为假（内部深处）；小目标（4×4）4 邻接腐蚀 2 次后为空 ⇒ 整块都在带内（形态学定义的自然结果）；空 GT 的 `band_pixels==0` 且 `band_metrics(...)["empty_band"] is True`
- `test_border_truncation_recorded`：贴角 GT（0,0,4,4）在 r=4 下 `gt_border_pixels > 0` 且 `band_pixels > 0`

`class TestPixelMetricsMatchMetricTool`：
- `test_counts_to_scores_matches_metric_tool`：用 `model.metric_tool.get_confuse_matrix` + `cm2score` 官方实现，对本模块同式的 `_scores` 逐键对拍，容差 `1e-12`（recall/precision/OA/F1/IoU/Kappa）

辅助函数：`canvas`、`rect`、`_with_extra_fp`、`_close(a, b, tol=1e-9, path="")`（递归 dict/list/数值，`nan==nan`，`None is None`）、`_scores`（本地复刻 eps 口径）。

#### 1.5.2 `tests/test_tvim_log_parser.py`（2,891 B / 74 行，2 个 test 方法）
`class TestLogParserEquivalence`：
- `test_matches_official_on_real_logs`（`@skipUnless(HAS_OFFICIAL, "official parser needs openpyxl")`）：在真实 train_log 上断言
  `self.assertEqual(b_mine, b_off, f"block differs for {log}")` 与
  `self.assertEqual(_parse_block(b_mine), off_parse(b_off), f"parsed dict differs for {log}")`，
  并要求 `n_checked > 0`（"no log contained a TEST RESULTS block"）。
  默认日志列表：`outputs/CASA-TViM/Run1/M1_FULL/SYSU-CD-256/train_log.txt`、`Run1/A2_STR/SYSU-CD-256`、`Run3/E5_FS_TAR/SYSU-CD-256`；全部缺失则回退到 `outputs/**/train_log.txt` 的**前 3 个**。
- `test_parse_test_block_extra_fields`：`block is not None`、`"F1" in info`、`span is not None` 且 `lo < hi`。
- 守卫目标：诊断侧复制的正则与官方实现**逐字一致**（含最后一个区块语义与行号 span），无日志时 `skipTest`。

#### 1.5.3 `tests/test_tvim_run5_preflight.py`（16,370 B / 357 行，30 个 test 方法）
`class TestRank`（6）：
- `test_avg_rank_range_and_ties`：`_avg_rank01` 输出形状同输入、`min==0.0`、`max==1.0`；并列值取平均秩（`r[1]==r[2]`）；单调性
- `test_avg_rank_label_independent_and_deterministic`：同输入两次结果 `assert_array_equal`；打乱顺序后 `c == a[perm]`（值→秩对应不变）
- `test_avg_rank_constant_is_half`：完全并列 → `0.5`，`atol=1e-12`（注释：直接测 rank 本体，避开 `rank_tensor` 插值抖动）
- `test_rank_tensor_is_monotone_in_upsampled_values`：先 bilinear 上采样再排秩，沿值排序后秩单调不减（`diff >= -1e-12`），范围 ⊂ [0,1]
- `test_rank_tensor_constant_stays_in_range`：常量场经双线性上采样有 ULP 抖动（注释：秩跨度可达 0.5，是预注册口径固有性质），只要求形状 `(1,256,256)`、有限、⊂[0,1]
- `test_rank_tensor_upsample_then_rank`：上半为 1 的场 → `out[:128].mean() > out[128:].mean()`

`class TestRingMargin`（3）：
- `test_margin_positive_for_inside_high`：object 内 score=1、环带 0 ⇒ `ok is True` 且 `m ≈ 1.0`
- `test_empty_ring_is_skipped`：全 GT 图 → `ok is False`、`m is None`
- `test_ring_excludes_other_gt`：紧邻的另一 GT 对象被 `∩(~GT)` 排除 ⇒ `ok is True` 且 `m > 0.5`（注释：obj 必须是**全尺寸 mask**，与生产调用一致）

`class TestImageRecords`（5，含合成 64² 场景 `_mk_case`：1 漏检 small GT(16px)、1 命中 large GT(400px)、1 未匹配 small 预测(9px)）：
- `test_counts_and_sets`：`n_missed==1`、`n_fp==1`、`n_hit_HF==1`、`n_hit_TAP==0`、`n_fp_hit_HF==0`、`n_hf_only==1`、`n_margin_pairs==1`；`kinds == ["fp_like","missed_small"]`；missed 对象 `area==16`、`pixel_recall==0.0`、`iou_max_any_pred==0.0`；fp_like `area==9`、`fp_like_id is not None`；`small_mask.sum()==16`（只含 1–255px GT）；`len(gt_rows)==2`
- `test_tap_hit_kills_novel`：令 TAP=HF ⇒ `n_hit_TAP==1`、`n_hf_only==0`、`n_tap_only==0`
- `test_medium_large_not_counted_as_missed_small`：**回归项（§5.2.1）**——128² 图含 small16/medium400/large1600 全漏检时，`n_missed==1`、`n_missed_pre_ring==1`、`r["area"]==16`、`small_n==1`、`small_px==16`、`small_tp==0`、`small_hit25==0`、`all(r["galias"]=="small")`
- `test_small_hit25_excluded_from_missed`：16px 对象命中 4/16（r=0.25）⇒ `small_hit25==1` 且 `n_missed==0`（**边界含等号**）
- `test_matched_pred_is_not_fp_like`：最大面积 pred 若被匹配则不得出现在 `fp_like_id` 中

`class TestBootstrap`（4）：
- `test_boot_ratio_brackets_obs`：`obs == num.sum()/den.sum()`、`valid==200`、`lo<=obs<=hi`
- `test_boot_gap_matches_obs`：`obs == hit.sum()/dm.sum() - fph.sum()/df.sum()`
- `test_boot_gap_zero_denominator_is_invalid`：分母为 0 ⇒ `obs is None`、`valid==0`
- `test_boot_mean_pair`：`obs == s.sum()/k.sum()`

`class TestSelectionRule`（8，§5.3.2/§5.3.3 纯函数）：
- `test_higher_novel_wins` → Pair-2
- `test_novel_tie_broken_by_gap`、`why == "novel_rescue>gap"`
- `test_gap_tie_broken_by_lower_mac`、`"deploy_mac" in why`
- `test_full_tie_lexicographic`、`"lexicographic" in why`
- `test_single_candidate`、`("Pair-3","novel_rescue")`
- `test_val_screen_classification`：`classify_val_screen` 四分支 `BOTH_SCREENED_OUT / INSUFFICIENT_COUNTS / VAL_TIMEOUT / PASSED`（决定都是 NO_80K，但记录不可混同）
- `test_val_screen_thresholds_frozen`：`VAL_MIN_MISSED==30`、`VAL_MIN_FP==50`、`VAL_SCREEN_GAP_MIN==0.05`、`VAL_SCREEN_NOVEL_MIN==0.05`、`TIE_EPS==0.01`
- `test_empty_eligible`：`select_pair({}, []) == (None, None)`

`class TestConstants`（5）：
- `test_gate_values_frozen`：`G4_rescue_rate_min==0.25`、`G5_gap_min==0.15`、`G5_gap_ci_low_min==0.03`、`G6_novel_rescue_min==0.10`、`G6_novel_ci_low_min==0.02`、`G7_margin_delta_min==0.02`、`MARGIN_THRESHOLD==0.03`、`BUDGET_S==600.0`、`EXPECT_N_TEST==4000`
- `test_list_lock_uses_text_lines_hash`：`SYSU_TEST_LIST_SHA_KIND=="sha256_text_lines"`、两个 SHA 长度 64（注释：与 raw sha256 不同，写死 kind 防被改）
- `test_pair_specs_budget`：`hf_channels == contract_shape[0]`、`hf_spatial == contract_shape[1]`、`deploy_delta_params == 96*2*hf_channels`、`4880190 + deploy_delta_params < 5000000`；`Pair-2 == 6144`、`Pair-3 == 16128`
- `test_small_alias_is_1_to_255`：1/15/16/63/64/255 ∈ SMALL_ALIAS；256/1023/1024 ∉
- `test_small_ap_hist_excludes_other_gt`：`ha.sum() == (small | ~gt).sum()`、`hp.sum() == 4`、`ha.sum() <= score.size - 64`（其它 GT 对象像素不得进入样本）

#### 1.5.4 `tests/test_tvim_run5_fp_profile.py`（10,421 B / 243 行，19 个 test 方法）
`class TestEdgeField`（4）：
- `test_edge_sparse_field_degeneracy_guard`：只有两列有梯度 ⇒ 90 分位为 0 时必须退化为 `mag > 0`，**不得把整幅图判成边缘**：`assertFalse(edge[:, :14].any())`、`assertTrue(edge[:, 15:17].any())`、`edge.mean() < 0.2`
- `test_edge_on_step_boundary`：竖直阶跃处有边缘，平坦区无；`mag` 阶跃处均值更大
- `test_edge_is_deterministic_and_scale_equivariant`：同输入两次 `assert_array_equal`；`img*4.0` 后边缘集**完全相同**（正比缩放不改变分位阈值下的边缘集）
- `test_uniform_image_yields_no_edges`：常量场 → 空边缘集（不是"全图都是边缘"）

`class TestDistToGt`（2）：`test_distance_zero_on_gt`（GT 上为 0，邻 2px 处 2.0，10px 处 10.0，`places=6`）；`test_no_gt_is_infinite`（全 `inf`）

`class TestBestShiftIou`（3）：`test_zero_shift_when_aligned`（IoU 1.0、`sh==(0,0)`、`k==1`）；`test_recovers_two_pixel_shift`（`sh==(0,-2)` 且 `MAX_SHIFT >= 2`）；`test_no_gt_returns_zero`（IoU 0.0、`sh==(0,0)`、`k is None`）

`class TestObjectMetrics`（4）：
- `test_fp_component_next_to_gt`：`area==16`、`dist_to_gt_min_px≈2.0`、`frac_px_within_4px_of_gt≈0.75`（列 35..38 距离 2,3,4,5 ⇒ 3/4 在 4px 内）、`frac_px_within_1px_of_gt==0.0`、`frac_px_on_any_edge_1px==0.0`、`gradA_mean==0.0`、`change_abs_mean==0.0`、`best_shift_iou is None`（`want_shift=False`）
- `test_fp_component_touching_gt_band`：`frac_px_within_1px_of_gt≈0.5`、`frac_px_within_2px_of_gt≈0.75`、`dist_to_gt_min_px==0.0`
- `test_shift_search_flags_misalignment`：差 2px 平移 ⇒ `best_shift_iou≈1.0`、`shifted_matched==1`、`matched_at_zero_shift==0`（**不能把错位误判为命中**）
- `test_no_gt_gives_none_distance`：`dist_to_gt_min_px is None`、`frac_px_within_4px_of_gt==0.0`、`best_shift_iou==0.0`

`class TestBackgroundControl`（2）：`test_control_avoids_gt_and_pred`（窗口与 GT、膨胀预测均无交集；`win.sum() == round(sqrt(area))**2`）；`test_returns_none_when_image_full`（全 GT ⇒ `None`）

`class TestBootstrap`（4）：`test_ratio_brackets_obs`；`test_ratio_zero_denominator`（`obs is None`、`valid==0`）；`test_paired_diff`（`obs == (a-b).mean()`、`n==4`）；`test_paired_ignores_nonfinite`（`obs≈1.0`、`n==1`）

`class TestConstants`（2）：`test_frozen_locks`（两个 SHA 长 64、`MAX_SHIFT==3`、`SHIFT_MATCH_IOU==0.50`、`EDGE_Q==0.90`、`BOOTSTRAP==1000`）；`test_small_alias_covers_1_to_255`

#### 1.5.5 `tests/test_tvim_run6_discriminability.py`（30,484 B / 662 行，55 个 test 方法）
`class TestPoolMap`（4）：`test_weighted_mean_exact`（加权均值 2.5）、`test_unequal_weights`、`test_zero_coverage_returns_none`（覆盖率 0 ⇒ `None`）、`test_matches_plain_mean_on_binary_mask`

`class TestResizeFrac`（5）：`test_full_mask_stays_full`（面积守恒 ±1%）、`test_empty_mask_is_empty`、`test_half_plane_is_half`（`places=3`）、`test_tiny_object_survives_downsampling_with_small_weight`（1 像素对象在 1/8 尺度权重 `>0` 且 `<0.01`，`pool_map != None` 且 `≈1.0`）、`test_object_entirely_outside_coarse_grid_can_vanish`（覆盖率全 0 ⇒ `None`，由 `_impute` 补中位数）

`class TestAucAp`（10）：`test_auc_perfect_separation`（1.0 与反向 0.0）、`test_auc_all_ties_is_half`、`test_auc_hand_computed_with_ties`（0.75）、`test_auc_partial`（2/3）、`test_auc_nan_dropped`、`test_auc_single_class_is_none`、`test_ap_hand_computed`（`0.5+1/3`）、`test_ap_perfect`、`test_ap_reversed_ranking_hand_computed`（`5/12`，且 `ap < 0.5`）、`test_ap_no_positive_is_none`

`class TestCrossScaleFeatures`（4）：`test_equal_scales`（四尺度 `Y_ratio_s*==0.25`、`Y_range==0.0`、`Y_cv==0.0`、`Y_entropy==ln4`）、`test_ratios_sum_to_one_and_dominant_scale_wins`（比值和 1.0、`Y_ratio_s1>0.99`、`Y_range>1e5`）、`test_entropy_minimal_when_one_scale_dominates`（`<1e-6`）、`test_negative_values_clipped_and_zero_safe`（负值被 clip 到 ≥0、`Y_cv` 有限）

`class TestFitLogistic`（4）：`test_separable_data_auc_above_0_9`、`test_deterministic`（两次权重 `assert_array_equal`）、`test_l2_shrinks_weights`（`lam=1.0` 的 `||w[1:]||` 小于 `lam=1e-3`）、`test_standardization_uses_fit_statistics_only`（`mu==1.5`、`sd==X.std()+1e-9`）

`class TestImpute`（4）：`test_median_from_fit_only`（中位数**只从拟合集取**：`med[0]==2.0`、`med[1]==20.0`，且填补值应用到另一集）、`test_all_nan_column_falls_back_to_zero`、`test_does_not_change_finite_values`、`test_matrix_keeps_none_as_nan`（`None → nan` 而非丢行）

`class TestMasks`（4）：
- `test_interior_and_ring`：`interior_mask(5×5).sum()==9`；`ring == dilate(obj, CONN4, 4) & ~obj`、与 obj/interior 交集为 0、`ring.sum()==104`；平移不变
- `test_ring_does_not_depend_on_gt`：**Run6 刻意去掉 Run5 的 `∩(~GT)`**（否则是 GT 泄漏），并断言函数签名 `ring_mask.__code__.co_varnames[:2] == ("obj", "iterations")`
- `test_interior_empty_for_single_pixel`：单像素内部为空、`ring(iterations=1).sum() > 0`
- `test_ring_uses_4_connectivity_structure`：`R.CONN4` 与 `M.CONN4` `assert_array_equal`；单像素 `ring1.sum()==4`（上下左右），`ring1[7,7] is False`（对角不属于 4 连通结构元）

`class TestHandPixelSpace`（4，手工证据通道必须在**归一化张量**口径）：
- `test_channel_mean3`：三通道均值 3.0；6 通道输入必须 `assertRaises(ValueError)`
- `test_frozen_change_abs_scale_is_normalized_not_raw`：用真实 `Transforms.Normalize/Scale/ToTensor` 链证明 `norm_gap < 10.0`、`norm_gap > 0`、`raw_gap > norm_gap`、`raw_gap / norm_gap > 5.0`（冻结读数 0.9730 只可能在归一化口径出现）
- `test_image_context_does_not_rescale_input`：`ctx["chg"]` 必须等于传入张量的 `|mean3(A)−mean3(B)|`（`atol=1e-6`），并显式断言**不等于** `×255` 的反归一化结果；形状/dtype 断言 `edgeA_d.shape==(256,256)`、`dtype==np.bool_`、`maps["Y_s1"].shape==(64,64)`、`shape["refine"]==64`
- `test_check_finite_raises_on_nan`：NaN 必须抛 `ValueError`

`class TestFeatureColumnCompleteness`（4，2026-10-11 远程首轮真实缺陷的回归）：
- `test_every_feature_column_is_assigned`：`compute_row` 必须为 `ALL_FEATURES` **每一列**赋值（`absent == []`）
- `test_no_feature_column_is_none_for_a_normal_object`（`none_cols == []`）
- `test_missing_report_detects_all_nan_column`（整列缺失计数 3）
- `test_no_all_missing_column_across_multiple_objects`（含 1 像素小对象，5 个对象拼接后无整列缺失）
- 注释记录了原始缺陷：`refine_mag / logit_mean / logit_std / logit_ring` 从未赋值 ⇒ 整列 NaN ⇒ 被 `_impute` 静默填成中位数 0 ⇒ `n_missing_imputed_*` = amp 4362 / head 13086 / full 17448（= 列数 × 4362 行）

`class TestProbeReportRowConservation`（4，**行数守恒**）：
- `test_missing_column_does_not_drop_rows`：`res["n_train"] == len(tr)`、`res["n_test"] == len(te)`、`n_missing_imputed_train > 0`、`n_missing_imputed_test > 0`、`not res["insufficient"]` ⇒ **缺失列不得使整行被丢弃**
- `test_confounded_feature_gives_auc_near_half`（常数列 → `test_auc==0.5`；`pure` 列 → `1.0`）
- `test_insufficient_guard`（样本过少 ⇒ `insufficient is True`）
- `test_single_feature_aucs`（`pure` 列 AUC 1.0）

`class TestK7Selection`（7）：`test_clear_winner_by_train_cv`（`why=="train_cv_auc"`）、`test_tie_within_eps_prefers_fewer_params_C1`（`"tie" in why`）、`test_exactly_eps_boundary_is_tie`（相差恰 0.005 → C1）、`test_just_over_eps_is_not_tie`（0.0051 → C2）、`test_c2_can_win_alone`（`why=="only_usable"`）、`test_tie_prefers_c1_even_when_c1_is_weaker`、`test_no_usable_candidate`（`(None, None)`）

`class TestFrozenConstants`（7）：
- `test_hard_gates_match_preregistration`：`K1_n_pos_min==300`、`K1_n_neg_min==2000`、`K2_test_auc_min==0.80`、`K2_test_auc_ci_low_min==0.75`、`K3_test_ap_min==0.35`、`K3_test_ap_ci_low_min==0.25`、`K4_auc_over_handcrafted_min==0.15`、`K4_handcrafted_baseline_auc==0.5992`、`K4b_auc_over_full_baseline_min==0.05`、`K5_auc_over_amplitude_min==0.10`
- `test_lock_hashes_and_budget`：`M1_SYSU_SHA=="45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457"`、`SYSU_TEST_LIST_SHA=="5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797"`、`BOOTSTRAP==1000`、`BUDGET_S==900.0`、`VAL_FRAC==0.10`、`PROTOCOL_TAG=="run6_scale_gate_exact80k_v1"`、`PROFILE_VERSION=="run6_discriminability_v1"`
- `test_frozen_reference_object_counts`：`n_pos_expected==327`、`n_neg_expected==2430`、`handcrafted_best_single_auc==0.5992`
- `test_baseline_groups_are_disjoint_and_cover_all_features`：跨组无重名、`sorted(flat)==sorted(ALL_FEATURES)`、候选项与三基线组两两不交、`BASELINE_FULL == AMP+HAND+HEAD`
- `test_candidate_sets_are_baseline_plus_exactly_one_group`：`candidate_C1 == BASELINE_FULL + C1`、`candidate_C2 == BASELINE_FULL + C2`、8 个 PROBE_SETS 名称固定
- `test_amplitude_baseline_contains_four_scale_mean`：`"mag_s_mean" in BASELINE_AMP`（§4.1 要求）
- `test_object_rule_constants`：`SMALL_ALIAS == GROUP_ALIAS["small"]`、不含 medium/large、`EDGE_Q==0.90`

`class TestPositiveRule`（4，对象口径必须与 Run5/F14 逐字一致）：`test_matched_gt_is_positive`、`test_missed_low_recall_is_positive`（r=0.10 → 1）、`test_unmatched_high_recall_is_not_included`（r=0.25 与 0.60 → `None`）、`test_boundary_recall_024_is_positive`（`0.2499 → 1`）

`class TestSmallGroupFilter`（2）：`test_only_small_alias_passes`（1/15/16/63/64/255 通过；256/1023/1024/5000 不通过）、`test_area_zero_raises`（`bin_index(0)` 抛 `ValueError`）

`class TestBudgetSplit`（2）：`test_split_is_proportional_and_positive`（`fit+ev==900.0`、`fit>ev`、`fit==900*10800/14800`）、`test_minimum_floor`（两段各 `>=60.0`）

`class TestGateHelpers`（3）：`test_k_structure`（`_k` 的 pass/threshold/note 字段；非 bool truthy 必须转 False）、`test_boot_ci_shape_and_reproducibility`（同 seed 两次结果相等、`a[2]==50`、`a[0]<a[1]`、`a[0]>0.8`）、`test_boot_ci_returns_none_when_no_valid_resample`（无正样本 ⇒ `lo is None`、`n==0`）

`class TestNumericHelpers`（3）：`_weighted_mean_std` 的零权重 → `(None, None)`；常量图 → `(3.0, 0.0)`；与均匀 mask 的普通均值/标准差一致

---

### 1.6 KNIFE_EDGE_ARTIFACT 裁决逻辑（不在 `tvim_object_metrics.py`）

两处实现，**判据定义完全相同**：
- `analyse/run4_fold_knife_edge.py`（随机合成 batch，`torch.randn(8,3,256,256)`，`--seed 16`）
- `analyse/run7_fold_knife_edge.py`（**真实数据 batch**，与 `train.py:884-911` 逐行对齐；`dataset root /share_datasets/CD/<dataset>`，test.txt，batch 16，shuffle=False）

核心代码（两文件一致）：
```python
a = y_train.cpu().numpy().astype(np.float64)
b = y_deploy.cpu().numpy().astype(np.float64)
flip      = (a > 0.5) != (b > 0.5)
d         = np.abs(a - b)
margin_tr = np.abs(a - 0.5)
max_abs   = float(d.max())
flipped_margins = margin_tr[flip] if flip.any() else np.array([])
knife_edge = bool(flip.any() and flipped_margins.max() < max_abs)
verdict = ("NO_FLIP" if not flip.any() else
           ("KNIFE_EDGE_ARTIFACT" if knife_edge else "NEEDS_REVIEW"))
```
- **无容差常数**：比较是严格 `<`（不是 `<=`），阈值就是本次折叠的实测 `max_abs`。
- 输出 JSON 字段：`n_pixels`、`max_abs_diff`、`n_flipped`、`flipped_fraction`、`flipped_per_image`、`flipped_pixels`（最多前 50 个翻转像素的 `idx / p_train / p_deploy / margin_train_abs_p_minus_half / abs_delta`）、`flipped_margin_max`、`n_pixels_with_margin_below_max_abs_diff`、`margin_percentiles_abs_p_minus_half`（`min / p0.01 / p0.1 / p1 / median`）、`verdict`、`note`。
- Run7 版本额外支持 `--log-fold-err-real` / `--log-disagree-real` 同源性对拍，判据为**相对容差 `< 1e-3`**（注释：日志只印 4 位有效数字）。
- Run4 版本 `note` 原文还指出：「§9-R6 要求定位数值与 FP64 融合，本文件即该定位证据」。
- 相关下游使用：`run4_fet_report.py` 的层 A 把「合成随机 batch max_abs < 2e-4」作为数值等价性硬门、「真实 batch disagreement == 0」作为操作性硬门，并在随机分歧非 0 时要求 `$RUN4_KNIFE_EDGE_DIR/KNIFE_EDGE_<DS>_<variant>.json` 在场（默认目录 `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run4_preflight_checks`）。

---

## 2. 全部脚本密集表（41 个文件）

> 「输入」= 从 argparse / 模块常量读出的确切参数；「输出」= 确切路径或 stdout。

### A. 基础设施与报告类（5）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/tvim_object_metrics.py` | 权威对象级/像素级/边界带指标定义（纯 numpy+scipy） | 无 CLI；库函数 `(pred, gt)` bool 数组 | 无文件（供 import） | 4 连通主口径 + 8 连通敏感性；`AREA_EDGES=[1,16,64,256,1024]`；IoU loose 0.10 / strict 0.50；一次性最大权重二分匹配；`pred=(prob>0.5)` 严格大于；band=`dilate_r \ erode_r`（`border_value=0`）；空组/空 band → `None`/`nan`；`eps=float32 eps` |
| `analyse/tvim_diag_common.py` | Diag1 只读基础设施 + D0 审计 / 协议落盘 CLI | `audit`: `--dataset SYSU-CD-256 --run Run1 --variant {9 选 1} --device cuda:0 --out-dir <必填> --batch-size 16 --fold-batch 16 --num-workers 4 --tol 1e-4 --write-per-image`；`protocol`: `--out-dir`；env `CASA_PROJECT / CASA_DATA / CASA_CKPT / CASA_OUTPUT` | `<out>/manifest.json`、`test_result.json`、`fold_equivalence.json`、`gate.json`；`protocol` → `<out>/metrics_protocol.json` | 复现容差 `1e-4`；折叠二值 disagreement 硬门 = 0；TEST 区块取最后一个；checkpoint 必须唯一 `best_F1=*.pth`；TF32 off + cudnn deterministic |
| `analyse/extract_metrics_to_excel.py` | 汇总所有 train_log 的 TEST 指标到一个 Excel（整表重建） | 无参数；扫描 `<ROOT>/outputs/**/train_log.txt` | `<ROOT>/docs/experiment_metrics.xlsx`（sheet `metrics`） | 六指标 ×100 保留 2 位；仅最后一个完整 TEST 区块；**覆盖式重建，无 append/backup/丢行断言** |
| `analyse/models_to_txt.py` | models/ 源码 + 指标汇总为 txt 快照 | `--tag baseline --run Run1` | `docs/temporary/models_and_metrics_<tag>_<run>.txt` | 代码只收 `.py/.yaml/.yml`（跳过 `__pycache__/build/csrc/.git`、`.pth/.pt/.pyc`）；指标 `.4f`、Params/Trainable `.3f`、FLOPs `.4f` |
| `analyse/tvim_diag_report.py` | 汇总 Diag1 全部证据 → markdown 报告（AUTO 结论 + [M] 占位） | `--root <必填> --out <必填> --write-manifest --interpretation` | `<out>` 的 md；`--write-manifest` → `<root>/RUN_MANIFEST.json`；可选追加 `<root>/INTERPRETATION.md` | 阈值：D1 size gap ≥**10.0pp**、D2 stage drop 相对 ≥**0.10** 且 bootstrap 同号、D3 probe AP 绝对下降 ≥**0.02**、每组最少 **100** 个 GT 对象；跨数据集绝对 AP 不可横比 |

### B. Run1 / Run2 / Run3 与历史诊断（7）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/run1_strfusion_budget.py` | STRFusion G4 参数量/FLOPs 预算门 | `--pretrained_weight_path <必填> --gpu_id 0` | stdout；`exit 0/2` | `BUDGET = 3.0e6`；deploy EFFECTIVE ≤ 3.0M；C0(plain)/M1(full) deploy 参数**逐位相等**；deploy `state_dict` 中 `bn` 键数必须 0；FLOPs 输入 2×3×256×256（fvcore，上报 unsupported）；dead = `encoder.resnet.layer4/fc/avgpool` |
| `analyse/run1_strfusion_interface_audit.py` | SF-D0 深度即尺度接口审计（4 数据集） | `--data_root <必填> --datasets <必填> --pretrained_weight_path <必填> --ckpt_r41_dir "" --gpu_id 0 --batch_size 16 --num_workers 4` | stdout；`exit` PASS=0 / FAIL=2 / AUDIT_INVALID=3 | B4 边界 PR-AUC 参考 **CDD 0.5216 / LEVIR 0.5369 / SYSU 0.6113 / WHU 0.5718（±0.01）**；SYSU R4-1 ResNet 1/8 对照 **0.6535 / 0.5948（±0.005）**；G1 `fuse4 ≥ B4+0.02` 且 ≥2/4 + SYSU 必过；G2 `fuse2 ≥ B4−0.02` 且 ≥3/4；`fuse4 = mean(map1..4)`、`fuse2 = (map1+map2)/2`、`b4 = map4` |
| `analyse/run2_report.py` | Run1(M1_FULL) vs Run2(E1/E2/E3) markdown 报告 | 无参数（读 `outputs/CASA-TViM/...`） | `docs/temporary/run2_report.md` | §15 目标 F1(%)：CDD ≥97.18、WHU ≥95.07；LEVIR 期望 E1 91.4 / E2 91.5 / E3 92.0；SYSU ≥ E1 83.47 / E2 83.7 / E3 84.0；抽取 β、cell 熵、`REPARAM-REAL-ARGMAX-DISAGREE`、deploy total |
| `analyse/run3_report.py` | Run1 vs Run3(E4/E5) markdown 报告 | 无参数 | `docs/temporary/run3_report.md` | 硬条件 CDD 97.18 / WHU 95.07；机制条件 E4 `SYSU F1>83.47、Recall≥84.41、small F1≥0.195`，E5 `SYSU F1≥83.80、Recall>84.41、small F1≥0.205`；LEVIR 下限 E4 91.07 / E5 91.20 |
| `analyse/run2_zero_cost_diag.py` | 历史 D1/D2 零成本机制诊断（含已记录的 FP 缺陷） | `--gpu_id 0 --batch_size 16 --limit 0 --ckpt_run Run1 --variants M1_FULL,A2_STR --datasets ""` | stdout；`docs/temporary/run2_zero_cost_diag.json` | 常量 `CKPT_ROOT=/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run1`、`DATA_ROOT=/share_datasets/CD`、`PRETRAIN=.../tinyvim_s_1000e.pth`、`LN4=1.3862943611198906`；变化率分箱 `zero / 0-1% / 1-5% / >5%`；环带 ±2 / ±4px；组件分组 `small<256 / medium<1024 / large`；**`FP` 恒为 0（伪指标）** |
| `analyse/casaa_router_diagnostic.py` | CASAA 路由–GT 对齐（V/D/F 三路 score） | `--dataset <必填> --dataset_root <必填> --pretrained_weight_path <必填> --ckpt <必填> --ckpt_tag <必填> --gpu_id 0 --batch_size 16 --num_workers 4 --limit 0 --out ""` | stdout + 可选 `--out` 文本 | `CASAA_LAYERS=[8,9,10,11]`；`casaa_keep_ratio=0.25`、`casaa_change_share=0.5`、`casaa_router="change"`；GT = `AvgPool16(label)` 的 occupancy>0；Top32 coverage / precision / K32−K33 gap / tie 率；D = `1−cos(avgpool2(detail_c[2]))` 再 percent-rank；F = `0.5*rank(V)+0.5*D`；分桶 `0 / 1-32 / 33-128 / 129+` |
| `analyse/param_breakdown.py` | 按组件拆分参数预算 | `--pretrained_weight_path <必填> --vit_depth 12 --gpu_id 0` | stdout | `DEAD_RESNET = ("encoder.resnet.layer4","encoder.resnet.fc","encoder.resnet.avgpool")`；`EFFECTIVE = TOTAL − dead`；`TRAINABLE` 单独统计 |
| `analyse/vit_pretrain_audit.py` | DeiT 预训练原位继承核对（CPU 即可） | `--pretrained_weight_path <必填> --vit_depth 4` | stdout；`assert` 中断即失败 | 逐张量 `assert (src-dst).abs().max() == 0.0`；`pos_embed` 源 `(1, 1+196, C)` → 去 cls/dist → 14×14 → bicubic → `assert model pos_embed.shape == (1, 256, 192)`；depth 之外 block 残留数必须 0 |

### C. Run4（5）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/run4_detail_interface_audit.py` | R4-D0 Detail 接口审计（SYSU 全 test） | `--dataset_root <必填> --pretrained_weight_path <必填> --ckpt_r41_dir <必填> --ckpt_r42_dir <必填> --ckpt_r42b_dir <必填> --gpu_id 0 --batch_size 16 --num_workers 4 --out ""` | stdout + 可选 `--out` 文本 | 每尺度 `AdaptiveAvgPool2d(16,16)` 后 `s=1−cos(f1,f2)`；Primary PR-AUC/Spearman/Top32 prec/Top32 cov；决策 A：`≥2/3` 尺度满足 `PR_raw ≥ PR_res−0.02` 且 `PR_ad ≤ PR_raw−0.03` 且 Spearman 同向下降 → ADAPTER_ALIGN；决策 B：`≥2/3` 尺度满足 `PR_raw ≤ PR_res−0.04` 且 `PR48−PR32 < 0.02` → PSD_DETAIL；否则 C（仍优先 PSD_DETAIL） |
| `analyse/run4_progress_eta.py` | Run4 进度/ETA 采样 + 可恢复状态快照 | `--log-root <必填> --variant M1_R4CTRL --interval 180 --write-status 0 --out` | stdout；`--out` JSON；`--write-status` → `<log-root>/STATUS.md` | `MAX_STEPS = 80000`；从 `[ACTUAL-OPT-STEPS] ... cumulative=<n>/` 取步数；两次采样求 rate/min 与 ETA clock |
| `analyse/run4_fold_knife_edge.py` | Run4 刀锋像素定位（随机合成 batch） | `--ckpt <必填> --variant M1_R4CTRL\|E6_FET1 --device cuda:0 --seed 16 --out` | stdout JSON + 可选 `--out` JSON | `KNIFE_EDGE_ARTIFACT ⇔ flip 存在 且 max(|p_tr−0.5| over flipped) < max|p_tr−p_dp|`；否则 `NEEDS_REVIEW`；无 flip → `NO_FLIP`；`strict=True` 加载并断言无 missing/unexpected |
| `analyse/run4_fet_report.py` | Run4 对象指标（3 模式）与三级裁决 | `--mode object`（+`--variant --dataset --run Run4 --ckpt-root --out-dir`）；`--mode verdict`（+`--log-root --ckpt-root --object-dir --out`）；`--mode markdown`（+`--verdict --out`）；公共 `--device cuda:0 --batch-size 16 --num-workers 4` | `<object-dir>/<variant>__<dataset>.json`；`--out` verdict.json；markdown 文本/文件 | 层 B 绝对门槛 `small_pooled_recall≥0.2474`、`small_hit25≥0.2396`；增量门槛 `+0.0500 / +0.0500 / ObjRecall +0.0100`；ObjPrecision 不低于 CTRL；`unmatched_pred_lt256` 不高于 CTRL；层 C 守门 `CDD 97.18 / LEVIR 91.07 / SYSU 83.47 / WHU 95.07` 且不低于同期 CTRL；论文目标 `98.00/92.50/85.00/95.00`；`BUDGET=5_000_000`；层 A：`steps==80000`、deploy ≤5M、合成随机 `max_abs<2e-4`、真实 `disagree==0`、六指标齐全；`IoU == F1/(2−F1)` 容差 `1e-3`、并列判据 `|ΔF1|≤5e-5 且 |ΔIoU|≤5e-4`；历史 M1 SYSU 参考 `0.1974/0.1896/0.7674/0.5320/2430/83.47/71.63` |
| `analyse/run5_postmortem_and_mobile_audit.py` | Run5-D0 postmortem（H1/H2/H3）+ MobileDetail-P3 raw gate | `--dataset_root <必填> --pretrained_weight_path <必填> --mobile_pretrained_weight_path <必填> --ckpt_r41_dir <必填> --ckpt_r42_dir <必填> --ckpt_r42d_dir <必填> --gpu_id 0 --batch_size 16 --num-workers 4 --skip_postmortem 0 --part all\|h2\|h1\|h3\|mobile` | stdout；`exit 0/2` | H2 判据：`conv_relL2≥0.15 且 ≥1.5×R4-1` 或 `act_psd.cos<0.90 且 act_r41.cos≥0.95` → 强支持；`<0.10 且 比值≤1.2 且 cos≥0.95` → 基本排除。H1：`ΔF1≤−0.30 且 ΔIoU≤−0.45` → 强支持；`|ΔF1|<0.15 且 |ΔIoU|<0.25` → 弱。H3：`r_kv/r_a ∈[0.70,1.30]` 判稳；`r∈[0.80,1.25]` 且 `|ΔH_n|<0.05` 判弱；`|ΔH_n|≥0.08` 越界。Mobile gate：`D4 PR≥0.50、D8 PR≥0.52、D4 prec≥0.46、D8 prec≥0.48`；参考 R4-2 F1 82.30 / IoU 69.92；(i)–(iv) 项见 §2 表格下方说明 |

### D. Run5 专项（2）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/tvim_run5_hf_pair_preflight.py` | Run5 §5 HF→TAR 配对前测（G0–G8 唯一开门依据） | `--dataset SYSU-CD-256 --run Run1 --variant M1_FULL --ckpt-root --dataset-root --train-list --test-list --probe-dir <必填> --device cuda:0 --batch-size 16 --num-workers 4 --split-hash sha1_10pct --seed 16 --bootstrap 1000 --limit 0 --budget-sec 600 --deploy-crosscheck-n 64 --save-figures 1 --out-dir <必填>` | `<out>/inputs.json`、`selection_val.json`、`val_per_object.csv`、`gate.json`、`bootstrap.json`、`layer_diagnostics.json`、`test_per_object.csv`、`test_per_image.csv`、`figures/`、`SHA256SUMS.txt`（val 未过时仍写 gate/bootstrap/layer_diagnostics 与**仅表头**的 test CSV） | 读出阈值 `MARGIN_THRESHOLD=0.03`；G2 `n_missed≥100`、G3 `n_fp_like≥100`、G4 `rescue≥0.25`、G5 `gap≥0.15 且 CI 下界>0.03`、G6 `novel_rescue≥0.10 且 CI 下界>0.02`、G7 `mean(margin_HF−margin_TAP)≥0.02 且配对 CI 下界>0`、G1 `n_test==4000 且 elapsed≤600s 且 valid_bootstraps≥950`；val 初筛 `n_missed≥30、n_fp_like≥50、gap≥0.05、novel≥0.05`；`TIE_EPS=0.01`；Pair-2 `Δparams 6144 / MAC 32·32·96·64=6291456`、Pair-3 `Δparams 16128 / MAC 16·16·96·168=4128768`，`4880190+Δ < 5,000,000`；probe 必须 `encoder_input=='concat'` 且 `C_in==3×tap_channels`；锁定 SHA 与 §3 表一致 |
| `analyse/tvim_run5_fp_profile.py` | Run5 §10 止损只读 FP 空间画像 | `--dataset SYSU-CD-256 --ckpt-root --dataset-root --test-list --device cuda:0 --batch-size 16 --num-workers 4 --bootstrap 1000 --seed 16 --limit 0 --out-dir <必填>` | `<out>/fp_profile.json`、`bootstrap.json`、`per_object.csv`、`per_image.csv`、`gate.json`（status 恒 `COMPLETED`）、`SHA256SUMS.txt` | 四组对象 A `fp_like_unmatched_small` / B `gt_small_matched` / C `gt_small_missed`(r<0.25) / D `background_control`；`EDGE_Q=0.90`（Sobel 幅值三通道均值，`mode="reflect"`，退化保护 `mag>0`）；`MAX_SHIFT=3`、`SHIFT_MATCH_IOU=0.50`；距离带 1/2/4px；`--out-dir` 必须空；`gate.json` 明确 `not_a_gate` |

### E. Run6 / Run7（3）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/run6_semantic_token_audit.py` | Run6-D0 Semantic/Patch-Token 零训练审计 | `--dataset_root <必填> --pretrained_weight_path <必填> --ckpt_r41_dir <必填> --ckpt_r40_dir <必填> --gpu_id 0 --batch_size 16 --num_workers 4` | stdout；`exit` PASS=0 / FAIL=2 / AUDIT_INVALID=3 | G1 P0 `PR≥0.44 & prec≥0.40`；G2 B4 `PR≥0.40 & prec≥0.36 & Spearman≥0.18`；G3 Fuse `PR≥0.50 & prec≥0.46 & Spearman≥0.25`；自检 CTRL R4-1 ResNet 1/8 `0.6535±0.005 / 0.5948±0.005`；`checksum_assert` 逐位比对 patch_embed/blocks0-3/norm/pos_embed |
| `analyse/run7_depth_source_audit.py` | Run7-D0 变化敏感深度金字塔四数据集审计 | `--data_root <必填> --datasets <必填> --pretrained_weight_path <必填> --ckpt_r41_dir "" --gpu_id 0 --batch_size 16 --num_workers 4` | stdout；`exit` 0/2/3 | C1 `PR(B2) ≥ PR(B12)+0.02`；C2 `PR(B2) ≥ PR(B4)−0.03`；C3 `prec(B2) ≥ prec(B4)−0.05`；**三项各需 ≥3/4 数据集**；CTRL 0.6535/0.5948 ±0.005；sources = `P0,B1,B2,B3,B4,B12` |
| `analyse/run7_object_metrics.py` | Run7 收口对象级只读复算（**deploy 图**） | `--run Run7 --variant <V>` 或 `--all-variants --variants M1_R7CTRL,E9_HFAE_A,E10_HFAE_B --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256`；`--dataset SYSU-CD-256 --ckpt-root /share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM --device cuda:0 --batch-size 16 --num-workers 4 --out <必填> --compare` | `<out>/object_metrics_<variant>_<dataset>.json`；`--compare` 打印对照表 | 必须 CUDA（selective_scan 仅 CUDA）；`ObjectMetricAccumulator(connectivity=4, bands=(2,4))`、`p>0.5`、`switch_to_deploy()`；冻结参考（仅对照）：`small_pooled_recall 0.1974`、`small_hit25 0.1896`、`n_gt_small_objects 364`、`unmatched_small_pred 2430`、`ObjP 0.5320`、`ObjR 0.7674`；`unmatched_small_pred_area_1_255 = tiny_1_15+tiny_16_63+small_64_255` 之和；RUN7_VARIANTS = `{"M1_R7CTRL":(0,"none"), "E9_HFAE_A":(1,"aux_only"), "E10_HFAE_B":(1,"aux_inject")}` |

### F. Run8 / Run9 / Run11（3）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `analyse/run8_b4_dense_recoverability_audit.py` | Run8-D0 B4 稠密可恢复性审计（4 数据集） | `--data_root <必填> --datasets <必填> --pretrained_weight_path <必填> --ckpt_r41_dir "" --gpu_id 0 --batch_size 16 --num_workers 4` | stdout；`exit` 0/2/3 | M1 全图像素 PR-AUC；M2 边界带 = `edge = mask XOR erode3×3`、`band = maxpool9×9(edge, pad=4)`；M4 = `conv_transpose2d(|B4_A−B4_B|, patch_embed.proj.weight, stride=16)` 的通道 L2（仅诊断）；G1 `PRpix(B4) ≥ PRpix(B12)+0.03`；G2 `PRbnd(B4) ≥ PRbnd(B12)+0.02`；C1/C2 各需 ≥3/4；C3 任何数据集不得下降 >0.01（4/4 必过）；CTRL 0.6535/0.5948 ±0.005 |
| `analyse/run9_overlap_reembedding_audit.py` | Run9-D0 O-PRE 互补性审计（4 数据集） | `--data_root <必填> --datasets <必填> --pretrained_weight_path <必填> --ckpt_r41_dir "" --gpu_id 0 --batch_size 16 --num_workers 4` | stdout；`exit` 0/2/3 | `s_O = 1−cos(O_A,O_B)`，O-PRE = 共享 `patch_embed.proj` 权重、`stride=8`、`reflect pad4` → 32×32；`proxy = U_B4 · (1 + 0.5·U_O)`（**固定 λ=0.5，不 sweep**）；G0 `PRbnd(OPRE) ≥ PRbnd(P0)+0.03` ≥3/4；G1 `PRbnd(proxy) ≥ PRbnd(B4)+0.02` ≥3/4；G2 `PRpix(proxy) ≥ PRpix(B4)−0.005` 于 4/4 且 `≥+0.01` 于 ≥2/4；G3 SYSU `Δbnd≥+0.03 且 Δpix≥+0.015` |
| `analyse/run11_tass_source_gate.py` | Run11 TASS-D0 raw 空间代理可行性门 | `--data_root <必填> --datasets <必填> --pretrained_weight_path <必填> --out_dir "" --gpu_id 0 --batch_size 16 --num_workers 4` | `<out_dir>/tass_d0_results.json`；stdout；`exit` 0/2/3 | 代理 `R64/R32/R16 = mean_c|AvgPool{k}(A)−AvgPool{k}(B)|`（k=4/8/16）→ bilinear 256² → 逐图像素 rank；`Rspatial = mean(rank(R64),rank(R32),rank(R16))`；`Rfuse = 0.5*rank_pixel(B4_up) + 0.5*Rspatial`（0.5/0.5 运行前固定）；G0 样本数三方一致 + `label gray>=128` + B4 边界 PR-AUC 复现 `0.5216/0.5369/0.6113/0.5718 ±0.01`；G1 `PRbnd(Rfuse)−PRbnd(B4) ≥ +0.020`，SYSU 必过且 ≥2/4；G2 `PRpix(Rfuse) ≥ PRpix(B4)−0.010`，SYSU 必过且 ≥3/4 |
| `analyse/run11_tass_budget.py` | Run11 TASS 预算门 | `--pretrained_weight_path <必填> --gpu_id 0` | stdout；`exit 0/2` | `BUDGET = 5.0e6`；deploy EFFECTIVE ≤5M；**非 TASS 的 `bn` 键必须 0**（TASS 单路径静态 stem 的 BN 合法保留）；输出 `headroom_to_5M`；dead 与 Run1 同定义 |

### G. 测试（5）
| 路径 | 作用 | 输入 | 输出 | 关键口径 |
|---|---|---|---|---|
| `tests/test_tvim_object_metrics.py` | P0 对象指标单元测试（13 方法，人造阵列手算预期） | `python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v` | unittest stdout | 覆盖面积分箱边界、空组 None 语义、连通性 4 vs 8、FP 回归、Hit@1 vs Hit@25、batch/顺序不变性、与 metric_tool 逐键对拍（`1e-12`）；torch 用例在无 torch 时 skip |
| `tests/test_tvim_log_parser.py` | 诊断侧正则与官方 `extract_metrics_to_excel` 逐字一致 | 同上 | unittest stdout | 真实 log 上 block 文本与解析 dict 全等；`n_checked>0`；span 存在且 `lo<hi`；openpyxl 缺失时 skip |
| `tests/test_tvim_run5_preflight.py` | Run5 §5 纯逻辑单测（30 方法） | 同上 | unittest stdout | avg-rank/rank-tensor、环带 margin 排除其它 GT、missed_small 面积过滤回归、hit25 边界 0.25、bootstrap 四点、选层规则与 val 初筛四分支、G4–G7/预算/锁定 SHA/PAIR_SPECS 常数冻结、small AP 排除其它 GT |
| `tests/test_tvim_run5_fp_profile.py` | Run5 §10 FP 画像纯逻辑单测（19 方法） | 同上 | unittest stdout | 边缘场退化保护（90 分位=0 → `mag>0`）、尺度等变、EDT 距离、±3px 平移搜索（含 2px 错位恢复）、背景对照避开 GT 与预测、bootstrap 四点、常数冻结 |
| `tests/test_tvim_run6_discriminability.py` | Run6 §4 可分辨性纯逻辑单测（55 方法） | 同上 | unittest stdout | pool_map/resize_frac 覆盖率口径、AUC/AP 手算（含并列）、C1 统计量、logistic 确定性与 L2 收缩、中位数只从拟合集取、ring 不用 GT（签名断言）、归一化像素空间（防反归一化）、**特征列完整性 P0 回归**、**probe 行数守恒**、K7 选一与全部 K 门 / 锁定 SHA / 组不交性冻结 |

---

## 3. 横切常量速查（跨脚本复用，改一个必须同步改这里）

| 常量 | 值 | 出现于 |
|---|---|---|
| `M1_SYSU_SHA` / `M1_SYSU_SHA256` | `45d688a01af22fe22521f4dfb4579e80827ddcaa886e07bafc778fc288731457` | run5_hf_pair_preflight、run5_fp_profile、run6_discriminability、run4_tap_preflight、tests/*run6* |
| `SYSU_TEST_LIST_SHA`（kind = `sha256_text_lines`） | `5f4122c9d32cb6db475f4f361618b2ac8bb10938df93d9550e0bb9a422a45797` | 同上 |
| 4 数据集顺序 | `CDD-CD-256, LEVIR-CD-256, SYSU-CD-256, WHU-CD-256` | tvim_diag_common、run2/3_report、run4_fet_report、run5_postmortem |
| 6 通道 MEAN / STD | `[0.406,0.456,0.485]×2` / `[0.225,0.224,0.229]×2` | tvim_diag_common、casaa_router_diagnostic、run2_zero_cost_diag、run4/5/6 系 |
| 二值阈值 | `pred = (prob > 0.5)`（严格大于，与 train.py 一致） | tvim_object_metrics、所有复算脚本 |
| GT small 组 | `1 <= area < 256`（4 连通，原生 256²） | tvim_object_metrics.GROUP_ALIAS、run4_fet_report、run5/run6 系 |
| 环带定义 | Run5 = `(dilate_4(O) \ O) ∩ (~GT)`；Run6 = `dilate_4px(O) \ O`（**刻意去 GT**） | run5_hf_pair_preflight、run6_discriminability（tests 有签名断言） |
| bootstrap 单位 | image-level，n=1000，seed=16，95% 分位 | tvim_diag_common、run5/run6 系 |
| run6 K 门 | K1 `300/2000`；K2 `0.80 / CI_low>0.75`；K3 `0.35 / CI_low>0.25`；K4 `+0.15` vs `0.5992`；K4b `+0.05`；K5 `+0.10`；K7 `tie≤0.005 → C1`（8 参数 vs C2 961） | tvim_run6_discriminability |
| run6 冻结读数 | `n_pos 327`、`n_neg 2430`、手工最佳单项 AUC `0.5992`、`change_abs` FP `0.9730` / GT-missed `0.8626`、`change_abs` AUC `0.4779` | 同上 + run5_fp_profile |
| run4 层 B/C 门 | MECH_ABS `0.2474 / 0.2396`；MECH_DELTA `+0.05 / +0.05 / +0.01`；GUARDRAIL `97.18/91.07/83.47/95.07`；PAPER `98.00/92.50/85.00/95.00`；BUDGET `5e6` | run4_fet_report |
| 预算门 | STRFusion `3.0e6`；TASS `5.0e6`；Run4 `5_000_000`；Run5 新增 tap 上界 `4,880,190+Δ<5,000,000` | run1_strfusion_budget、run11_tass_budget、run4_fet_report、run5_hf_pair_preflight |
| 数值纪律 | `cudnn.allow_tf32=False`、`matmul.allow_tf32=False`、`cudnn.deterministic=True`、`benchmark=False`；`seed=16` | tvim_diag_common.set_eval_numerics、run7_fold_knife_edge、各 preflight |
| 指标 eps | `float(np.finfo(np.float32).eps)` = 1.1920929e-07 | tvim_diag_common.EPS32、tvim_object_metrics._counts_to_scores |
| 禁止事项（各脚本显式声明） | 不 `train()`、不改 BN、不改 optimizer、不写回历史产物；`--limit>0` 一律不得 PASS（DRY_ONLY/rc=2） | run5/run6 preflight、tvim_* 全部 |

---

## 4. 风险与缺口清单（读码所得，供决策）

1. **`docs/experiment_metrics.xlsx` 无防丢行保护**：整表重建、无备份、无行数守恒断言，也无历史行对比。92 个 log 中任何一个不可读即静默缺行（仅 stdout 打印 `[skip]`）。若该 xlsx 被当作论文指标唯一台账，建议：先写临时文件 + 校验行数 ≥ 上次 + 失败则保留旧文件。
2. **`models_to_txt.py` 会覆盖同名快照**（`open(..., "w")`），docstring 中"不覆盖旧文件"的表述与实际行为在相同 `--tag/--run` 下不符。
3. **`match_objects` 是"先全局分配、后阈值过滤"**：低 IoU 的对会占用行列。若下游希望"仅在高 IoU 子图上匹配"，当前实现会与之不同；报告口径必须按实际实现描述。
4. **`tvim_object_metrics._counts_to_scores` 与 `tvim_diag_common.counts_to_scores` 的 Kappa 处理不同**：前者在 `1-pre==0` 时返回 `nan`，后者直接做除法（可能 `ZeroDivisionError`/`inf`）。两者都声称"与 metric_tool 同口径"，实际在退化情形上不一致。
5. **`run2_zero_cost_diag.component_pr` 的 FP 恒为 0** 是已知缺陷，其 `small/medium/large F1`（历史值 small 0.1796 / medium 0.4620 / large 0.8273）**不得作为对象级 F1 引用**；`tvim_diag_report.py` 与 `tvim_small_error_audit.py` 已在报告中显式标注。
6. **KNIFE_EDGE_ARTIFACT 不是对象指标**：它属于折叠数值等价性证据，判据为严格 `<`（无容差常数），且 `run4_fet_report` 只在「合成随机 batch 分歧非 0」时才要求该证据文件在场。
7. **`tvim_diag_common.build_model` 不覆盖 Run7 变体**（`VARIANT_CFG` 无 `hf_edge*`），故 `run7_object_metrics.py` 自带 `build_run7_model` 显式构造——若未来新增变体，两条构造路径必须同步。
