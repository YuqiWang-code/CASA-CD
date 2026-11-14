# Run15 — 三个历史版本的统一口径消融

目的：Run14 只对当前主线（Run12/v12r3）做了配对消融，得到「双曲几何无精度收益、TAR 学习式混合不承载收益、DCR 是唯一稳定起作用模块」。
本轮的追问是：**这些结论是不是只属于当前这一版实现**。Run1（Poincaré）、Run4（直接 Lorentz）、Run10（后验压缩 + matched 欧氏孪生）都曾在某库
给出过正值信号，因此把三个版本的源码树各自原样搬到服务器，用同一套数据、同一预算、同一条 **val 选模** 口径重跑对照臂。

## 代码来源与改动

| 版本 | 目录 | 来源 | 改动 |
| --- | --- | --- | --- |
| Run1 | `models_Run1/`（`runtime/models_Run1/`） | `/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1/source_code.tar.gz` | 无（该版本本来就是 val 选模） |
| Run4 | `models_Run4/` | `.../Run4/source_code.tar.gz` | 新增 `--selection-split {test,val}`，默认 `test` 保持原行为；本轮传 `val` |
| Run10 | `models_Run10/` | `.../Run10/source_code.tar.gz` | 同上；另 `--seed/--context-seed` 固定 2333 |

每个目录下的 `SOURCE.md` 记录远端路径、`source_code.tar.gz` 的 sha256 与文件计数；`run_scripts/` 是该轮实际使用的调度脚本，仅作归档。

两处必须知道的历史差异，写结论时要标注：

1. **Run1 的归一化均值是 BGR 顺序**（`MEAN = [0.406, 0.456, 0.485] * 2`），与 Run4/Run10/主线的 RGB 顺序不同。这是该轮的真实协议，本轮不改，Run1 的数字只能与 Run1 内部对照臂比较。
2. Run1 是 Poincaré 球 + 投影几何的早期实现（`model/layers/poincare_ops.py`）；Run4 换成直接 Lorentz 提升；Run10 增加 `matched` 欧氏孪生（与双曲同参数量、同初始化）。

## 矩阵（20 项）

| 版本 | 臂 | 数据集 |
| --- | --- | --- |
| Run1 | `none`(CTRL) / `hyperbolic`(H_POINCARE) / `qkv`(E_RUN1) | WHU、SYSU |
| Run4 | `none` / `hyperbolic`(H_LORENTZ) / `qkv`(E_RUN4) | WHU、SYSU |
| Run10 | `none` / `hyperbolic`(H_LORENTZ) / `matched`(E_MATCHED) / `qkv`(E_RUN10) | WHU、SYSU |

- 统一口径：`--selection-split val`，每 epoch 在 **val** 上算 F1 选模，末次在 test 上只评一次（`independent_test=true`）。
- 超参：256²、batch 32、40K steps、seed/context-seed 2333、Adam 5e-4（各版本自带的分组权重衰减保持不变）。
- 选 WHU + SYSU 的原因：这两库是历史版本出现正负信号的地方；LEVIR/CDD 在 Run14 已是持平，重复训练的信息量低于成本。

## 参照点

- Run14（主线、val 选模、3 seed 均值）：WHU H 93.19 / E 93.88，SYSU H 81.90 / E 81.84。
- Run13（主线、test 选模）：WHU CTRL 94.6403、SYSU CTRL 82.5512 —— 与 val 选模不是同一口径，只能看组内相对差。
- Run12（历史最好，test 选模）：WHU 94.6–94.8、SYSU 82.8–83.5。

## 执行

```bash
cd /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run15/runtime/train_scripts/CASA-Pyramid/Run15
source environment.sh
python prepare.py     # CPU 审计 + 每版本 1 次端到端 smoke + scheduling/readiness
python pipeline.py    # 20 项，按 RUN15_GPUS/RUN15_SLOTS 调度
python pipeline.py --resume   # 中断后继续
```

本地打包与环境落盘：`python train_scripts/CASA-Pyramid/Run15/bootstrap.py`。

成本：单卡两槽、每项 2.3–2.8 h，20 项约 24–28 h 墙钟；若两卡空出可 `RUN15_GPUS=0,1 RUN15_SLOTS=2` 压到约 13–14 h。

## 产出

- 每项：`<version>/<arm>/<dataset>/{train_log.txt, result.json, run_manifest.json, deploy_check.json, operating_point.json}`。
- 汇总：`comparison.{json,md}`（各版本内 Δ 相对 `none`），`collected.json`，`inference_benchmark.json`。
- 结论口径：同一版本内比较 Δ；跨版本只比较「H−CTRL 的符号与量级」，且必须注明 Run1 的数据顺序差异。
