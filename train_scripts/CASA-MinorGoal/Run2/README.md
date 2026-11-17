# CASA-MinorGoal / Run2：种子复现与两条隔离臂

Run1 只跑了一个种子，因此 SPP 的 SYSU +1.06 pp 和 DIFF 的 WHU +0.72 pp 都还停在"单次观测"上；同时 DIFF 被硬绑在 `--tar-variant spp` 上，无法回答"差分 KV 的收益是不是靠多尺度池化"。这个跑次把两件事一起补：种子复现 + 把两个变量拆开。

## 协议

- `PROTOCOL = casa_minor_spp_diff_seeds_v2`，与 Run1 同一训练配方（256²、batch 32、40K step、Adam 5e-4、多项式衰减、四级 BCE+Dice、guide 0.1）。
- 全部**验证集选模**、阈值 0.5；`independent_test=true`。
- 数据集只用 WHU / SYSU：这两库是 Run1 出现正负信号的地方，LEVIR 的差值与 CDD 的持平已由 Run1 覆盖。
- 只用物理 GPU1、两槽并发：9 臂 × 2 库 = 18 项，WHU 约 2.3 h、SYSU 约 2.6 h 一项，预计 23–25 h。

## 九条臂

| 臂 | 开关 | 作用 | 训练 / 部署参数 |
| --- | --- | --- | --- |
| `CTRL_S2334` | `--module none --seed 2334` | 2334 的同口径基线 | 2.047125 / 1.935810 M |
| `SPP_S2334` | `--tar-variant spp --seed 2334` | 复现 SPP | 2.118293 / 1.995714 M |
| `DIFF_S2334` | `--module difference --tar-variant spp --seed 2334` | 复现 DIFF | 2.129705 / 2.007126 M |
| `CTRL_S2335` | `--module none --seed 2335` | 2335 的同口径基线 | 2.047125 / 1.935810 M |
| `SPP_S2335` | `--tar-variant spp --seed 2335` | 复现 SPP | 2.118293 / 1.995714 M |
| `DIFF_S2335` | `--module difference --tar-variant spp --seed 2335` | 复现 DIFF | 2.129705 / 2.007126 M |
| `SPP1` | `--tar-variant spp1` | 同结构但池化窗口只有 1×1，隔离"窗口变宽"的贡献 | 2.095509 / 1.983426 M |
| `DIFF_SPP1` | `--module difference --tar-variant spp1` | 差分 KV + 1×1 池化，DIFF 的窗口对照 | 2.106921 / 1.994838 M |
| `DIFF_NOSPP` | `--module difference --tar-variant diff` | 差分 KV 不再经过 SPP，只保留 `|P−Q|` 混合 | 1.955361 / 1.917014 M |

`--tar-variant spp1` 与 `diff` 的差分流支持是本次新增：`SPPTARStage(kernels=(1,))` 与 `DifferenceTARStage.forward_with_difference`，`DifferenceContext` 的硬依赖从"spp"放宽为"任意提供差分流的时间融合"。

## 判读方式

- `SPP_S2334 − CTRL_S2334`、`SPP_S2335 − CTRL_S2335`、`SPP − CTRL(Run1)` 三个配对差给出 SPP 的跨种子均值与标准差；DIFF 同理。
- `SPP1 − CTRL` 与 `SPP − SPP1` 分别回答"1×1 差分池化本身是否有用"和"把窗口放宽到 3/5/7 是否额外有用"。
- `DIFF_SPP1 − SPP1` 是差分 KV 在不带多尺度池化时的净效应；`DIFF_NOSPP` 是同一问题的极限版本（连 SPP 结构都去掉）。
- 噪声参照：Run1 同配置复跑 WHU 0.18 pp、SYSU 0.32 pp；跨种子离散以本跑的 3 seed 实测为准。

## 运行

```bash
bash -lc "cd runtime/train_scripts/CASA-MinorGoal/Run2 && source environment.sh && python prepare.py"
bash -lc "cd runtime/train_scripts/CASA-MinorGoal/Run2 && source environment.sh && MINOR_GPUS=1 python pipeline.py"
```

收集中：

```bash
python tools/download_training_results.py --tag CASA-MinorGoal --run Run2
MINOR_ROOT=outputs/CASA-MinorGoal/Run2 MINOR_LOCAL_IO=1 CASA_REPO=. \
  python train_scripts/CASA-MinorGoal/Run2/collect_results.py
```
