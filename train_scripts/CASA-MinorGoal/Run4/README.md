# CASA-MinorGoal / Run4 — 四条定案归因跑（GPU1 only）

协议 `casa_minor_run3_attribution_v1`。GPU0 在 Run2 结束后退役，本跑只用 **GPU1**、2 槽。
起跑由 `launch_after_latency.sh` 把关：等 Run2 的 pipeline 进程退出 **且** GPU1 空闲显存
≥ 25000 MiB 才 `exec python pipeline.py`（deadline 24 h，超时 exit 3，日志 `launcher.log`）。

## 任务矩阵（不是笛卡尔积）

| job | module | tar | 额外开关 | seed | 数据集 | 训练/部署参数 | 预计墙钟 |
|---|---|---|---|---|---|---|---|
| `ARM_C/WHU-CD-256` | hyperbolic | spp | — | 2334 | WHU | 2,129,577 / 2,006,998 | 2.3 h |
| `ARM_C/SYSU-CD-256` | hyperbolic | spp | — | 2334 | SYSU | 2,129,577 / 2,006,998 | 3.0 h |
| `SUPP_PIX/SYSU-CD-256` | hyperbolic | spp | `--support pixel` | 2334 | SYSU | 2,129,577 / 2,006,998 | 3.0 h |
| `DETAIL/LEVIR-CD-256` | difference | spp | `--detail-head` | **2333** | LEVIR | 2,132,394 / 2,009,815 | 3.5 h |

四条合计 11.8 GPU-h，2 槽 ≈ 6 h。DETAIL 用 2333 是因为 LEVIR 只有 2333 有精确配对控制
（Run1 DIFF 90.8535 / SPP 91.0612）；前三条用 2334，因为 Run2 的 `SPP_S2334` 已经跑完
（WHU **93.8540**、SYSU **82.2689**；注意不是同名的 2333 臂 93.6925 / 82.7915）。DETAIL 的 detail head 2,689 参数（1×1 32→64 = 2,048，dw3×3 = 576，1×1 64→1 = 65），
推理期参与（`primary + resize(detail)`），因此部署参数同样是 2,009,815，GMAC/延迟需重测。

## 预登记判据（开跑前锁定，跑完不改）

**R4（DETAIL/LEVIR vs Run1 DIFF/LEVIR，同 seed 2333）**
- 总 F1 ≥ 90.8535 + 0.36
- 边界（≤0 px）F1 ≥ 59.486 + 3.0，且正类中边界占 31.41%
- 内部 F1 ≥ 96.349 − 0.18
- 单 seed，结论只写探索性。

**ARM_C（vs Run2 SPP 同库同 seed 2334）**
- WHU：ΔF1 ≥ +0.36 ⇒ 模块存在效应主导；|ΔF1| ≤ 0.18 ⇒ 无模块效应，DIFF 增益归 KV 内容。
- SYSU：ΔF1 ≥ +0.47 ⇒ 同理；|ΔF1| ≤ 0.32 ⇒ 无模块效应。
- 门槛是旧口径（WHU 0.18 / SYSU 0.32 复跑噪声）。若 analyst 给出更大的复跑噪声（>0.36），
  修订判据并记录，但不改已排队列。

**SUPP_PIX（vs ARM_C/SYSU，同 seed 2334）**
- ΔF1 ≥ +0.47 才认"逐尺度像素 support"为第二杠杆。

**R4 的混杂（必须随结果一起写）**：`DIFF` vs `DIFF+detail` 同时改了 (a) 边界分支
(b) 额外一项 detail 深监督。若 R4 过门，需再跑一臂 `--detail-head` 但关闭 detail 深监督的
变体才能区分是本分支还是额外监督带来的。**本跑不加**，只登记。

**detail 损失**：`training_losses` 把 `loss_function((detail_logits,), target)` 直接加到主损失上
（`models/train_CASA.py:90`），权重恒为 1，不做重新归一化 ⇒ 总损失由 4 项变 5 项。实际量级由
`prepare.py` 的 `detail_loss_ratio()` 实测写入 `detail_loss.json`。

## 每图 TP/FP/FN（零成本，图像级 bootstrap 用）

`run_one.py` 固定追加 `--per-image-metrics`（`models/train_CASA.py:362-363`），test 评测在
`result.json` 之外额外写 `<run>/per_image_cm.npz`：`index/tp/fp/fn` 四个 int64 数组，另带
`dataset/split/batch_size/threshold/images/list_entries/list_path/list_sha256/protocol/seed/selection_split`
等采信字段（`train_CASA.py:630-644`）——per-image 行是按位置对齐的，没有 list 哈希就可能静默错位。
只在 test 落盘，val 不动；存 pooled 计数而非 per-image F1（无正样本图 F1 无定义）。`result.json` 原字段不变，`prepare.py` 的 smoke 与
`validate.py` 的 `check_models` 保证参数/折叠口径不变。

## 就绪检查（fail-fast，任一不成立不许开跑）

1. `validate.py` 全绿：矩阵、CLI（`--support {mass,cell,pixel,none}`、`--detail-head`、
   `--per-image-metrics`）、参数表、折叠 `maxdiff ≤ 2e-4` 且二值差 0。`mass`/`cell` 是同一
   先验的两个名字（`cell` 为历史别名），`none` 表示不加先验。
2. `prepare.py` 每臂 one-step smoke，训练/部署参数与上表逐位一致。
3. **DETAIL 未训练时与 DIFF 输出逐位相同**（`maxdiff == 0`，detail 末层零初始化）；不成立就停。
4. `--support pixel` 不是静默 no-op：mass vs pixel 的**概率**图 `maxdiff > 1e-4`
   （`model(a, b)` 返回 sigmoid 概率，不是 logits；smoke 的 BCE 已在 logit 上算）。
5. `scheduling.json` 只含 `"1"`，`gpu_budget_mib {"1": 31583}`。

## 运行

```bash
cd train_scripts/CASA-MinorGoal/Run4
# 本地（Windows）只打包并上传冻结 runtime，不等 GPU
python bootstrap.py
# 服务器
python prepare.py --fold-device cpu     # GPU 忙时 CPU 折叠，与 Run1 CPU 折叠同口径
python pipeline.py --dry-run            # 打印矩阵/预计完成时刻/命令，不 spawn
bash launch_after_latency.sh            # 等 Run2 退出且 GPU1 空闲后 exec pipeline.py
python pipeline.py --resume             # 中断后续跑
python collect_results.py               # 下载 → 抽表 → txt 快照 → compare.py
```

`pipeline.py` 的 `complete()` 只认 `train_log.txt` 里完整的 `=== TEST RESULTS ===` 区块，
半截日志不会被误判为完成；`--resume` 只续跑未完成的 job，不重复启动。
`fit()` 的槽位与显存都是**按卡**计算的（传入的是 `per_gpu[gpu]`），`MINOR_GPUS=0,1` 时不会
退化成全局 2 槽。

## 采集与裁决

`collect_results.py` → `tools/download_training_results.py --tag CASA-MinorGoal --run Run4`
→ `extract_metrics_to_excel.py` → `models_to_txt.py` → `compare.py`。
`compare.py` 按 `config.GATES` 与 `config.LEVIR_DETAIL_GATES` 直接给裁决（`effect confirmed` /
`effect reversed` / `no effect (inside the seed noise)` / `direction only`），并把 DETAIL 的
总 F1、边界 F1、内部 F1、边界占正类比例逐项与门限对照，写 `attribution.{md,json}`。

## 冻结 runtime 的采信

`prepare.py` 把当前 `models/` 复制成 `runtime/models/` 并写 `runtime_manifest.json`
（每个 `.py` 的 SHA-256 前 16 位）；`bootstrap.py` 额外写一份 `runtime_manifest.json` 到运行目录。
服务器上现存的 runtime 来自 Run1/Run2 那一代（`support="mass"` 时代、无 `kv_hierarchy`/`freeze_tau`），
语义上与当前树等价（flat 层级、不冻 τ、`cell == mass`），因此**不重新 staging**；本跑的
provenance 以服务器上那份 `runtime_manifest.json` 为准。本地树的 `--support` 现已同时接受
`mass`/`cell`/`pixel`/`none`，重跑 staging 也不会再因 `invalid choice: 'mass'` 崩。

### 修复记录（2026-11-19）

`ManifoldContext` 初版没有覆写 `prototype_features`，KV 仍由融合特征 `T_s` 池化得到，
TAR-CG 的跨尺度差分只经 `support` 与门控间接进入检索——与设计意图不符。现已加入
`norm_diff` 与 `prototype_features` 覆写（与 `DifferenceContext` 同构），KV 改为
`LN(D_s)` 的类条件原型，并补 `test_manifold_context_reads_the_difference_stream`。
参数因此 +128：`HRA_SPP 2,136,115 / 2,013,536`、`CG_HRA 2,111,287 / 2,061,668`；
两条 manifold 臂的旧权重作废，重新开跑。
