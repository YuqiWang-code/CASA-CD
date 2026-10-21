# tools/ — 通用分析工具

从 `tools/archive/` 改名而来（2026-10-21）。分三层：

```
tools/
  *.py            通用工具：不绑定某一轮 run，换 run/数据集可直接复用
  tests/          通用工具的单测（对象指标 + 日志解析器逐字对拍）
  archive/        run 专用脚本（Run1–Run11 / Exp1），只为备查，不属通用层
```

## 通用工具

| 文件 | 用途 |
|---|---|
| `extract_metrics_to_excel.py` | 扫 `outputs/` 的 `train_log.txt` → 写 `docs/experiment_metrics.xlsx`（增量、按 key 去重） |
| `models_to_txt.py` | `models/` 代码快照 + 指标 → 一个 txt（用于交付/回溯） |
| `param_breakdown.py` | 参数量 / FLOPs 分解（主干 / 非主干 / 各模块） |
| `tvim_object_metrics.py` | **正确口径**的对象级小指标：GT 面积分组 Recall / ObjectHit / 对象级 P-R-F1 / 边界带（4 连通、native 256² mask） |
| `tvim_diag_common.py` | 诊断基础设施 + D0 对拍审计（`audit` / `protocol` 子命令） |
| `tvim_stage_recoverability.py` | 分阶段 hook 的可恢复性分析（编码器 A/B cosine proxy + CAACP 内部） |
| `tvim_stage_raw_stats.py` | cosine margin 的配对 image-level bootstrap CI |
| `tvim_linear_probe.py` | 冻结线性 probe（`--encoder-input absdiff\|concat`） |
| `tvim_probe_stratified.py` | 按 GT 面积分层的 probe AP / per-object margin |
| `tvim_diag_report.py` | 汇总报告 + 预注册判据自动评估 + `RUN_MANIFEST.json` |
| `vit_pretrain_audit.py` | 预训练权重与模型的 key/shape 对齐审计 |

## 单测

```bash
cd <repo>
python -m unittest discover -s tools/tests -p 'test_tvim_*.py' -v
```

## archive/

按轮次命名的专用脚本（`run1_*` … `run11_*`、`tvim_run4_*` / `tvim_run5_*` / `tvim_run6_*`、
`tvim_caacp_counterfactual.py`、`tvim_small_error_audit.py` 等）及其对应单测。
它们**依赖当时的产物布局与模型结构**，现在多半已经跑不通，只用于复盘。

> 迁移说明：原 `tools/archive/` 下的路径在历史文档里仍写作 `tools/archive/…`。
> 历史叙述已集中在 `../docs/temporary/历史版本结论与指标_2026-10-21.md`，
> 其中提到的 `tools/archive/xxx.py` → 现为 `tools/xxx.py`（通用）或 `tools/archive/xxx.py`（专用）。
