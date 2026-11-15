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
