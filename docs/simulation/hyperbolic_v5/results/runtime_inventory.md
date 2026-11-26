# runtime_inventory —— Run10 冻结 runtime 与 checkpoint 清单

生成方式：读 `best_model.pth` 同目录的 `run_manifest.json`/`arch.json`，以及
`/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime_manifest.json`。

## 1 checkpoint 级口径（4 个 R_FULL 全部核对）

| 项 | 值 |
|---|---|
| arm | R_FULL-2334 / R_FULL-2335 |
| dataset | SYSU-CD-256 / LEVIR-CD-256 |
| ckpt | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run10/<arm>/<dataset>/best_model.pth` |
| protocol | `casa_pyramid_40k_testselect_heads_detail_v12r3` |
| selection_split | `val`（`independent_test=True`） |
| selection | best val F1 @ threshold 0.5，逐 epoch 覆盖更新 |
| steps | SYSU 40125 / LEVIR 40182 |
| module | `manifold` |
| str_dim / qkv_dim / heads | 64 / 32 / 1 |
| grid_sizes | [8, 4, 4, 4]（区域数 112，原型数 224） |
| retrieval_scales | [0, 1, 2, 3] |
| support / pooling / aggregate | `mass` / `lorentz` / `lorentz_barycenter` |
| tar_variant / decoder_variant | `cg` / `reparam` |
| deploy_verification | `strict_fp32_fold_v1` |

## 2 源码哈希（与 runtime_manifest.json 一致）

| 文件 | sha256 前 16 位 |
|---|---|
| `models/model/layers/hyperbolic_context.py` | `3bf6ee3a…` |
| `models/model/layers/manifold_context.py` | `9d041b32…` |
| `models/model/layers/lorentz_ops.py` | `b1ac4ce6…` |
| `models/train_CASA.py` | `2aec1d86…` |
| `train_scripts/.../config.py` | 不在该 manifest（NA） |

## 3 protocol 字段的差异（已解释，不是矛盾）

- `run_manifest.json` 的 `protocol` = pipeline 协议 `casa_pyramid_40k_testselect_heads_detail_v12r3`
  （描述训练管线与 40k 步、val 选模）。
- runtime `config.py` 的 `PROTOCOL` = `casa_minor_run10_lean_v1`（Run10 这一轮的"精简臂"协议名，
  决定跑哪些 arm/超参）。
- 两者是不同层级的字段；AGENTS.md 里说的 `casa_minor_run7_manifold_pooling_v1` 属于更早一轮，
  与 Run10 的 config 不是同一字符串。

## 4 可用臂与数据完整性

| 臂 | SYSU | LEVIR | 说明 |
|---|---|---|---|
| R_FULL-2334 | ✅ | ✅ | 本轮 S7c 主用 |
| R_FULL-2335 | ✅ | ✅ | 本轮 S7c 主用 |
| R_COMPACT-2334 | ✅ | ❌ | 2×2 网格（32 原型） |
| R_LEAN-2334 | ✅ | ❌ | 同上 |
| R_RES-2335 | ✅ | ❌（曾缺、后补则需复查） | 粗层 Q |
| D_CG / D_CTRL | ✅ | ✅ | `kv_tokens=0`、`module=none`，无 HRA，只能做下游对照 |

结论：R_FULL 四套齐，可用于同预算内的几何反事实；32 原型臂只有 SYSU，跨库的预算曲线
受限，必须在报告里注明。
