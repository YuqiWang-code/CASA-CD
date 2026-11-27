# CASA-MinorGoal / Run11：把 HRA 拆成三个可归因的几何轴（`casa_minor_run11_hra_v1`）

## 这一轮要回答什么

Run7 的 HRA 是"整块双曲"：成员池化、查询—原型距离、V 聚合三处同时用 Lorentz 几何，
一次消融只能说明"这套几何组合有/没有收益"，不能说明收益来自哪一处；Run7 的
`POOL_EUCLID 83.212 vs POOL_LORENTZ 82.824`（SYSU）又提示三个轴未必同向。
Run11 在 Run10 的 `ManifoldContext` 上做最小改造：数据流、`T` 查询、`D` 变化证据 K/V、
guide、类条件压缩、learned-share 先验、gamma 残差回写、DCR/FineRefine 全部不动，
只把三个几何开关拆开：

| 轴 | 取值 | 含义 |
|---|---|---|
| `member_pool` | `euclidean` / `lorentz` | 每格每类的逐像素投影如何聚成原型（K/V） |
| `distance` | `euclidean` / `lorentz` | 查询与原型如何比较（注意力 logits 里的距离项） |
| `value_aggregate` | `euclidean` / `lorentz` | softmax 权重如何聚合 V（`aggregate` 分支） |

坐标约定与 Run10 一致：`z = qk 投影 / sqrt(d)`（d=32）；Lorentz 曲率 −1，
`L(z) = (sqrt(1+||z||²), z)`；成员池化用 `spatial(S/sqrt(⟨S,S⟩_L))`，距离用
`lorentz_ops.pairwise_squared_distance`，V 聚合用 `ManifoldContext.aggregate`。
三轴全 `lorentz` 时与 Run10 `R_FULL`（224 原型、`FLAT_GRIDS`）逐 logit 一致
（本地冒烟 `max|HRA224 − R_FULL| = 0.000e+00`），因此 32 原型臂有 224 参照。

## 臂、对照与门限

| 臂 | TAR | HRA 轴（pool/dist/value） | 原型 | 说明 |
|---|---|---|---|---|
| `CTRL` | cg | 无（guide 在 D 上） | — | 无 HRA 的匹配对照 |
| `LOCAL` | cg | 无（只有局部细节支路） | — | 隔离 TAR+guide+局部残差 |
| `H32` | cg | H / H / H | 32 | Run10 几何放到 32 预算 |
| `E32` | cg | E / E / E | 32 | 参数/数据流完全匹配的 E-HRA |
| `E32_HV` | cg | E / E / H | 32 | 只换 V 聚合一个轴 |
| `H224` | cg | H / H / H | 224 | 高预算参照（`(8,4,4,4)`） |

* `CTRL` 与 `LOCAL` 的 guide 都从 `D` 上算（`--guide-source diff`），与 HRA 臂的 guide
  监督一致；Run10 的 `D_CG` 用的是 `T` 上的 guide，因此 **Run10 数字只作方向参考**，
  同轮结论只用 `CTRL`/`LOCAL`/`H32`/`E32` 配对比较。
* 32 原型 = `(2,2,2,2)`，224 = `(8,4,4,4)`；四级查询尺度全部保留，`retrieval_scales=0,1,2,3`。
* support 固定 learned share（`mass`）；Run10 的 `confidence`（`|2p−1|`）在 `p=0.5`
  梯度阻断，本轮不再使用。
* 种子 2334 / 2335，`--selection-split val`，主阈值 0.5，bs 32，40k step，
  SYSU 先、LEVIR 后；只使用物理 GPU0，两槽（`MINOR_GPUS=0 MINOR_SLOTS=2`）。
* 门限（预登记，跑前写死）：`LOCAL vs CTRL`、`H32 vs LOCAL`、`E32 vs LOCAL` 用
  `max(0.36, 1.5σ) = 0.59 pp`；`E32_HV vs E32` 用一个噪声单位 `0.36 pp`；
  `H224 vs H32` 用 `−0.36 pp`（32 预算不得比 224 掉超过一个噪声单位）。
* 归因问题 `H32 vs E32` 不作为门限：打平也是结论（三个轴不等价于一块几何），
  不因此补跑或改口径。

## 预检与折叠

`prepare.py` 逐臂做 CPU 审计 + 一步 smoke，并在设备上折叠：

* 每臂 4 个 `(1,1,128,128)` 输出、有限、与 `module none/reparam` 控制不同（no-op 守卫）；
* 折叠 `strict_fp32_fold_v1` 首级 `max abs diff ≤ 2e-4`、二值差 0；
* 打开写回（`gamma_context = level_scale = 0.5`）后 `H32`/`E32`/`E32_HV` 输出两两不同，
  证明三个轴真的在起作用；
* 梯度非零：`tar.stage1.*`、`context.qk/v/out/gate`、`mean_gate`、`support_head`、
  `level.0`、`head`、`fine.out`；
* HRA 全 lorentz 与 `ManifoldContext` 同权重同输入 logits 一致（本地冒烟已过，
  服务器预检会重跑参数/折叠项）。

## 产物

* `outputs/CASA-CD/CASA-MinorGoal/Run11/<arm>/<dataset>/train_log.txt`（含
  `=== TEST RESULTS ===`）
* `params.json`（smoke 回填的 trainable/deploy 与 HRA 轴报告）、`comparison.md/json`
* 收尾：`docs/experiment_metrics.xlsx`、`docs/temporary/models_and_metrics_CASA-MinorGoal_Run11/`

## 运行

```bash
# 本地打包并上传（会拒绝覆盖已有 readiness.json）
python train_scripts/CASA-MinorGoal/Run11/bootstrap.py

# 服务器预检（GPU 折叠）
cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run11/runtime/train_scripts/CASA-MinorGoal/Run11
source environment.sh && python prepare.py --fold-device cuda

# 队列（只 GPU0）
MINOR_GPUS=0 python pipeline.py
```

环境见 `.codex/SERVER.md`；连接只走 `.codex/ssh.py`，凭据不出现在输出里。
