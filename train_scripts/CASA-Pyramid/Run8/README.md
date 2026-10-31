# CASA-Pyramid / Run8

协议`casa_pyramid_40k_attribution_v8`。仅使用物理GPU1，保留既有进程，按实测显存与1GiB留量并行。用户确认按[Run7分析方案](../../../docs/temporary/CASA_Pyramid_Run7_深度分析_调研与改进方案.md)分阶段开展，当前仅启动检索监督归因阶段。

## 首阶段：4项完整预算对照

| 数据集顺序 | 实验 | guide BCE系数 | retrieval BCE系数 | 区域精炼β |
|---|---|---:|---:|---:|
| SYSU → WHU | H_CLASS | 0.1 | 0.1 | 0 |
| SYSU → WHU | H_NONE | 0.1 | 0 | 0 |

两臂完整224KV、单头32维、共享K/V内容、support、gate、local分支相同，唯一训练目标差异为检索BCE系数。两臂均从MobileNetV2预训练全新初始化，不继承Run7或探针权重。run seed和context seed均固定2333并独立记录，train/eval loader使用独立配对随机流；这一入口变化使两臂需要本轮配对重训，不直接将历史Run7当作完整匹配锚点。

RGB256、batch32、Adam全网lr5e−4、200warmup/poly^.9、40K按完整epoch上取整；SYSU107epoch/40125step、WHU217epoch/40145step。衰减、增强与四级主分割损失保持共同设置。沿用每epoch test F1选模，结果须标明test参与选模、非独立盲测；完整val用于冻结干预及辅助诊断，不参与本阶段选模。

## 模型接口与后续阶段

`models/train_CASA.py`支持`--guide-weight`、`--retrieval-weight`、`--seed`、`--context-seed`及`--refinement-beta {0,1}`。`--module matched`为严格匹配欧式对照，保持同一投影与原型，仅替换空间距离。该入口不等同Run5普通QKV。

一次精炼按区域类别初始P计算cosine兼容性，以exp(β cos)重加权，再用同一更新权重计算P及support。β=0精确复现原压缩，β=1为预先固定候选，无额外参数；首阶段保持β=0。

首阶段完整结果收集后，比较val上的真实内容、内部/对象召回和F1，确定共同辅助目标，再开展匹配H/E完整预算。只有内容混合与有害读出得到证据支持，才启动原压缩/一次精炼对照。四库多seed推广在归因通过后安排，尚未加入当前队列。

## 检查、启动与产物

`analyze_frozen.py`读取冻结Run7源码和best权重；SYSU另测last。完整val记录全局/局部2×2开关、平均query注意力、按源尺度固定/置乱半径、路由分量、前景连通域覆盖及逐图配对记录。内容诊断独立保留名义注意力和GT池化占比，GT在前向后用于统计。

```bash
bash train_scripts/CASA-Pyramid/Run8/validate.sh
bash train_scripts/CASA-Pyramid/Run8/probe.sh
bash train_scripts/CASA-Pyramid/Run8/launch.sh
```

validate完成模型/梯度/折叠/调度检查和两臂profile；probe在SYSU、WHU分别对两臂完成2个train/val epoch，保存任务梯度。readiness通过表示数值和梯度健康，不表示精度提升。pipeline.py为唯一正式调度器，仅GPU1，首阶段共4项。

日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run8`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run8`。源码、环境、配置和初期观察独立归档；分析在analysis/，不写为正式指标。每项结束后记录best/last在完整val的guide/retrieval BCE（auxiliary_validation.json），与主分割选模loss区分。

```bash
python .codex/status.py --tag CASA-Pyramid --run Run8
python train_scripts/CASA-Pyramid/Run8/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run8
bash train_scripts/CASA-Pyramid/Run8/test_one.sh H_CLASS SYSU-CD-256
```

collector只采纳完整TEST区块，至少253条历史记录逐单元格保留，首阶段4项收齐后至少257条；另有新增历史全部保留。冻结TXT来自实际源码归档。单独复测放retests/，不重复记为正式实验。

<!-- RUN8_STATUS -->
2026-10-31 01:03:56 CST完成首阶段4/4项，无失败；源码基线dea8421，实际运行归档已保留。日志/JSON、Excel、冻结及当前TXT已更新，完整训练记录见docs/experiments/CASA-Pyramid/Run8。检索监督相对关闭时SYSU F1 +0.1589pp、WHU −0.2072pp；后续共同监督、匹配几何与β1精炼对照待安排，不重复启动已完成队列。
