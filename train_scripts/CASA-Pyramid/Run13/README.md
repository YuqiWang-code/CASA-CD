# CASA-Pyramid / Run13

TAR 与 DCR 的单因子消融。协议仍为 `casa_pyramid_40k_testselect_heads_detail_v12r3`，控制支路为 `module=none`、1 头、无细节头（即 Run12 的 `CTRL_*`/`DETAIL_OFF` 配置），因此每一项与控制的唯一差异就是被替换的那一处结构。

| 臂 | 改动的结构 | 替换方式 | 训练/部署参数 |
|---|---|---|---:|
| 控制（Run12 复用） | — | — | 2.047125 M / 1.935810 M |
| `TAR_PLAIN` | `TemporalRep1x1` 的 concat + P+Q + Q−P 三支 | 单个 `Conv1×1(2C→C)+BN`；`RepLocalBlock` 换成普通 DW3×3+PW1×1+BN | 1.974157 M / 1.935810 M |
| `DCR_PLAIN` | `RepPairFuse1x1` 的 concat + sum + diff 三支 | 单个 `Conv1×1(2C→C)+BN`；局部块同样换成普通 DW3×3+PW1×1+BN | 2.010250 M / 1.935810 M |

四条尺度、通道数、DCR 的逐级上采样与融合顺序、头部、损失、增广、预算（40K 步、batch 32、seed 2333）全部不变。两个臂都保留拓扑（4 尺度 → 4 级解码 → 单头），只删掉可折叠的多分支冗余，因此消融问的是"这些分支带来的表达力是否值得它的训练期参数与实现复杂度"，而不是"删掉整块结构会不会变差"。

## 对照与噪声

控制值取 Run12 同配置的正式结果（test 选模、阈值 0.5）：

| 数据集 | 控制臂 | F1 | 同配置复跑跨度 |
|---|---|---:|---:|
| WHU | `CTRL_ANCHOR` | 94.6403 | 94.6403–94.8173（0.177 pp） |
| SYSU | `CTRL_REPEAT` | 82.5512 | 82.5512–82.8699（0.319 pp） |
| LEVIR | `DETAIL_OFF` | 91.0455 | 单一结果 |

`compare.py` 以 `delta_pp = 消融臂 F1 − 控制 F1` 出具表格，并把同配置复跑跨度作为噪声下限：`|delta| ≤ 噪声` 记作"落在种子噪声内"，`delta < −噪声` 记作"删除结构的代价超过噪声"。LEVIR 只有单次控制，结论按探索性处理。

## 结果

北京时间 **2026-11-12 22:00** 全部 6 项完成，无失败，日志与权重留在服务器 `Run13/` 与 `checkpoints/CASA-Pyramid/Run13/`。均为 test 选模、阈值 0.5 的同集复测。

| 数据集 | 臂 | F1（%） | 控制 F1（%） | Δ（pp） | 噪声（pp） | 判定 |
|---|---|---:|---:|---:|---:|---|
| WHU | `TAR_PLAIN` | 94.7843 | 94.6403 | +0.144 | 0.177 | 落在种子噪声内 |
| WHU | `DCR_PLAIN` | 94.7339 | 94.6403 | +0.094 | 0.177 | 落在种子噪声内 |
| SYSU | `TAR_PLAIN` | 82.3698 | 82.5512 | −0.181 | 0.319 | 落在种子噪声内 |
| SYSU | `DCR_PLAIN` | 82.3239 | 82.5512 | −0.227 | 0.319 | 落在种子噪声内 |
| LEVIR | `TAR_PLAIN` | 91.0016 | 91.0455 | −0.044 | — | 探索性：低于单次控制 |
| LEVIR | `DCR_PLAIN` | 91.1427 | 91.0455 | +0.097 | — | 探索性：高于单次控制 |

结论：把 TAR 的 concat/P+Q/Q−P 三支与 DCR 的 concat/sum/diff 三支换成单支 1×1+BN 后，三个数据集上的 F1 变化都在同配置复跑跨度以内（最大 |Δ| 0.227 pp，对应噪声 0.319 pp）。也就是说，这些多分支结构在当前预算与容量下不承载可测的精度收益，训练期参数却分别多 72,968 与 36,875 个。部署侧完全等价：折叠后参数同为 1,935,810、RTX 5090 单张 256² FP32 batch 1 中位延迟 3.678 / 3.680 / 3.672 ms。

因此这批分支应作为"实现可选"而非"精度来源"表述；论文里能主张的是折叠等价性与训练期表达力的冗余，不能主张它提升了 F1。要注意本消融只动 TAR 与 DCR 内部的可折叠分支，上下文检索（H/E）与细节头不在此列，其对照见 Run12。


## 矩阵

6 项：`TAR_PLAIN`、`DCR_PLAIN` × WHU / SYSU / LEVIR。两项在两张 RTX 5090 上各并发 2 路。实测单项 WHU 约 8600 s（2.4 h），SYSU 约 9650 s，LEVIR 约 9480 s；两波并发约 3 h 完成全部 6 项。

## 评价

沿用 v12r3：每轮在 test 以阈值 0.5 的 F1 选权重并跑满预算，训练结束后用 val 的 0.10–0.70（121 点）阈值曲线校准，再在同一 test 集复测。结果标注 `selection_split="test"`、`independent_test=False`，不是独立盲测。

## 执行与恢复

本地先 `python train_scripts/CASA-Pyramid/Run13/bootstrap.py`：把当前 `models/` 与本目录脚本打包成服务器 `Run13/runtime/`，并生成 `source_code.tar.gz`、`source_manifest.json`、`environment.json`。

服务器上在 `Run13/runtime/train_scripts/CASA-Pyramid/Run13` 执行：

```bash
source environment.sh
python prepare.py     # CPU 审计 + 折叠 + 单步端到端 + 两臂各 2 epoch 探针 + 显存实测
python pipeline.py    # 双卡队列
```

中断或失败后在同一目录 `python pipeline.py --resume`：已完成的正式项跳过，未完成项从最后完整 epoch 接续。`prepare.py` 与 `pipeline.py` 都不会覆盖已有的 `readiness.json`/`pipeline_status.json`。

本地回收：`python train_scripts/CASA-Pyramid/Run13/collect_results.py` 下载日志、按完整 `TEST RESULTS` 汇入 `docs/experiment_metrics.xlsx`、冻结源码快照，并调用 `compare.py` 生成 `outputs/CASA-Pyramid/Run13/ablation_table.{json,md}`。
