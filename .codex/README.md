# CASA-CD 桌面工作目录

此目录保存 Codex / ChatGPT 桌面任务使用的项目工具和上下文。仓库根目录 `AGENTS.md` 承接项目工作约定；本目录不改变桌面应用的全局配置。

| 文件 | 用途 |
|---|---|
| `ssh.py` | 从本地配置连接 RSML-3，执行命令或本地命令文件 |
| `status.py` | 本地入口，读取指定轮次的远程队列、日志和 GPU 占用 |
| `server_status.py` | 由 status.py 通过 SSH 执行的只读状态查询 |
| `SERVER.md` | 服务器路径、环境、部署和结果收集约定 |
| `STATE.md` | 当前轮次、源码基线及待办任务 |
| `RUN7_TASK.md` | Run6 结果与 Run7 分析、设计及启动任务 |

在仓库根目录执行：

```powershell
python .codex/status.py --tag CASA-Pyramid --run Run9
python .codex/ssh.py "nvidia-smi" 60
python .codex/ssh.py outputs/remote_command.sh 120
python train_scripts/CASA-Pyramid/Run6/collect_results.py
```

连接信息保存在未提交的 `.vscode/sftp.json`，Python 需要 `paramiko`。状态查询只读，不会启动、停止或恢复训练。命令文件可放入 `outputs/`；可重复使用的实验启动、停止和检查脚本放在 `train_scripts/<tag>/<Run>/`。

研究原则见根 README。数据与图像分析产物放在 `outputs/<tag>/<Run>/analysis/`，正式指标在 `docs/experiment_metrics.xlsx`，源码快照在 `docs/temporary/`。

旧工作文件归档在 `outputs/tooling_archive/legacy_claude_20261023.zip`。其中 `_r5_out`、`_r5_fp_out`、`_r6_out`、`_r7_obj_out` 属于早期 CASA-TViM；不能按文件夹编号当作当前 CASA 轮次。近期诊断原文包括 `_run3_baseline_validation.log`、`_run4_checkpoint_audit.log`、`_run4_trend_summary.log`、`_run5_assignment_audit.log`。已整理的 Run5 梯度分析位于 `outputs/CASA/Run5/assignment_gradient_analysis.json`。
