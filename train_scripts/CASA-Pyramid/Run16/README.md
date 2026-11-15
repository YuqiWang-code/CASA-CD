# Run16 — TAR / DCR 模块移除在三份历史版本上的复现

Run15 只回答了「几何有没有用」。这一轮回答另一半：**TAR 的时相混合与 DCR 的多级解码，它们的作用是否随版本成立。**

三个版本的 `model/str_tar.py`、`model/str_dcr.py` 在 Run1/Run4/Run10 里与 v12r3 原文件**逐字节相同**（git blob `1485930…`、`7945dfc…`），Run13/Run14 的模块移除实验测的就是这段代码。本轮在版本树上重建同样的两条臂，检验同一结论在不同基线、不同 context 实现下是否稳定。

## 两条臂

两条臂都保留「四尺度 → 四级解码 → 单头」拓扑，只替换某一级的可学习内容，因此差值只归因于该级：

| 臂 | 替换内容 | 训练参数 | 部署参数 |
| --- | --- | ---: | ---: |
| `TAR_DIFF` | 学习式时相混合（concat / P+Q / Q−P 三支）→ 固定 `\|P−Q\|` + 1×1 | 1,943,949 | 1,905,602 |
| `DCR_SINGLE` | 多级 coarse→fine 解码（fuse+block ×3 + refine）→ 单层上采样 + 参数自由残差 | 1,974,730 | 1,900,802 |

控制值取 Run15 的**同版本 `none` 臂**（val 选模）：Run1 93.6175 / 81.7839、Run4 93.3574 / 83.0623、Run10 94.0877 / 81.0951（WHU / SYSU）。

## 矩阵

3 版本 × 2 臂 × 2 库 = **12 项**，全部 val 选模：

| 版本 | 臂 | WHU | SYSU |
| --- | --- | :-: | :-: |
| Run1 | `TAR_DIFF` / `DCR_SINGLE` | ✓ | ✓ |
| Run4 | `TAR_DIFF` / `DCR_SINGLE` | ✓ | ✓ |
| Run10 | `TAR_DIFF` / `DCR_SINGLE` | ✓ | ✓ |

WHU 约 41 s/epoch（217 epoch）、SYSU 约 100 s/epoch（107 epoch）；两卡两槽约 7–8 h。

## 源码改动

三个版本树只做了**加法式**补丁，默认行为与归档时完全一致：

- `model/str_tar.py`、`model/str_dcr.py` 换成当前超集（原类未改，新增 `PlainTemporal1x1` / `DifferenceTemporal1x1` / `PlainTARStage` / `DifferenceTARStage` / `PlainLocalBlock` / `PlainFuse1x1` / `SingleLevelDecoder` / `build_decoder`）。
- `model/casa_net.py` 增加 `tar_variant="reparam"`、`decoder_variant="reparam"` 两个默认参数并透传。
- `train_CASA.py` 增加 `--tar-variant {reparam,plain,diff}`、`--decoder-variant {reparam,plain,none}`，manifest 与 `arch.json` 记录取值。

## 运行

```bash
cd train_scripts/CASA-Pyramid/Run16
python bootstrap.py                     # 本地：打包三个版本树 + 套件并上传
# 服务器
cd ~/outputs/CASA-CD/CASA-Pyramid/Run16/runtime/train_scripts/CASA-Pyramid/Run16
source environment.sh
python prepare.py                       # validate + smoke + 显存 profile
python pipeline.py                      # RUN16_GPUS=0,1 RUN16_SLOTS=2 可扩卡
python pipeline.py --resume             # 断点续跑
```

## 结果

完成后由 `collect_results.py` → `compare.py` 生成 `comparison.{json,md}`：每个 (版本, 库) 与同版本 Run15 控制臂配对，给出 Δ 与按臂的平均 Δ。口径提醒：这些 Δ 是 **val 选模**下的 Δ，与 Run13（test 选模）比的是相对量，不是绝对 F1。

2026-11-15 23:56 CST 完成 12/12 项，无失败：

| 版本 | 控制 WHU / SYSU | `TAR_DIFF` | Δ（pp） | `DCR_SINGLE` | Δ（pp） |
|---|---|---:|---:|---:|---:|
| Run1 | 93.6175 / 81.7839 | 93.7435 / 82.1943 | +0.126 / +0.410 | 93.2805 / 80.7502 | −0.337 / −1.034 |
| Run4 | 93.3574 / 83.0623 | 94.1642 / 82.6838 | +0.807 / −0.379 | 93.9531 / 79.9029 | +0.596 / −3.159 |
| Run10 | 94.0877 / 81.0951 | 94.0826 / 82.5632 | −0.005 / +1.468 | 93.5406 / 81.4424 | −0.547 / +0.347 |

汇总：`TAR_DIFF` 六项平均 **+0.405 pp**（换固定 `|P−Q|` 不掉精度，参数 2.047125 M → 1.943949 M 训练 / 1.935810 M → 1.905602 M 部署）；`DCR_SINGLE` 六项平均 **−0.689 pp**（四负两正，SYSU 上最不能少，幅度依赖基线，须逐库报告）。
