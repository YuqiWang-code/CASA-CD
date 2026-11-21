# Run8 — the pooling axis on its own (results folded into Run7)

> 归档说明：这两条臂的产物已并入 `CASA-MinorGoal / Run7`（服务器日志/权重在 `Run7/POOL_*`，指标表 `Run=Run7`，快照在 `docs/temporary/models_and_metrics_CASA-MinorGoal_Run7/`）。本目录只保留脚本与协议 `casa_minor_run8_pooling_axis_v1` 的定义，便于复跑。

Run7 packed the two modules into a 2×2 and every retrieval arm also carried the manifold
pooling, so it could not answer whether the barycentre itself buys anything. Run8 fixes the
architecture and moves one axis only.

## Two arms

| arm | module | tar | aggregate | prototype pooling | train / deploy (M) |
|---|---|---|---|---|---|
| `POOL_EUCLID` | manifold | cg | lorentz | **euclidean**（切空间加权平均后再 lift） | 2.1113 / 2.0617 |
| `POOL_LORENTZ` | manifold | cg | lorentz | **lorentz**（先将每个成员点 lift，再求加权 Klein 重心） | 2.1113 / 2.0617 |

两臂参数逐位相同（池化是纯几何操作，不含参数），逐位相同的还有 seed、调度、边缘项（关闭）。

## 口径

- 协议 `casa_minor_run8_pooling_axis_v1`，GPU1 单卡两槽，只跑 **SYSU**（检索臂唯一动过的库），seed 2334，val 选模，test τ=0.5。
- 门限：`POOL_LORENTZ` 相对 `POOL_EUCLID` 需 ≥ +0.36 pp，噪声 0.32 pp。
- 参考值（Run7 同架构、同 seed、lorentz 池化）：SYSU 83.016；Run7 的 `TAR_HRA` 与本轮 `POOL_LORENTZ` 配置等价，因此这一轮同时也是那次结果的复现检查。

## 判读

- 若 `POOL_LORENTZ − POOL_EUCLID` ≥ +0.36：流形上的压缩本身有收益，可写"重心优于切空间均值"。
- 若落在 ±0.32 内：压缩位置不影响精度，Run7 的 +1.14/+1.63 只能归给模块存在，不能归给几何。
- 若 ≤ −0.36：重心在压缩阶段有害（与 R/P 换位那条线索一致——重心偏保守）。

## 运行

```bash
python bootstrap.py                     # 打包当前 models/ + 本目录到服务器
# 服务器：
cd $ROOT/runtime/train_scripts/CASA-MinorGoal/Run8
source environment.sh && export CUDA_VISIBLE_DEVICES=1
python prepare.py                       # 审计 + 冒烟 + readiness
python pipeline.py                      # 两项，约 5.5 h
python collect_results.py               # 下载 + 指标表 + 快照
python compare.py --root ../../../../../local/path/to/Run8
```
