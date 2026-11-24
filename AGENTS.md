# CASA-CD 工作约定

项目定位与研究原则见 `README.md`。当前任务和服务器状态见 `.codex/STATE.md`，服务器操作见 `.codex/SERVER.md`。

- 面向普通 RGB、全监督遥感二值变化检测，在 LEVIR、SYSU、WHU、CDD 上比较准确率与推理、可学习参数开销。模块设计须对应可验证的问题，保持完整 Q、压缩 KV，优先简洁且收益可归因的结构。
- 当前主线是 CASA-MinorGoal / Run7（协议 `casa_minor_run7_manifold_pooling_v1`：`--tar-variant cg` + `--module manifold`）；Run2–Run6 的 `spp`/`difference` 开关仍在树上可复现。`models/` 只放当前工作树；`models_Run1/`、`models_Run4/`、`models_Run10/` 三棵历史版本树已删除，需要时从 git 历史（删除前最后提交 `a3e839d`）取回；MoE 探索线在 `models_v1/`。改模型前先确认属于哪一条线、哪一轮，运行中的源码不直接覆盖。
- 每个新臂必须同时给出：训练/部署参数、部署折叠一致性（≤2e-4）、no-op 守卫（改动后的输出必须与对照不同）、被替换层梯度非零，以及 test 侧的 per-image TP/FP/FN。缺任一项不算跑过。
- 选模口径由 `--selection-split {test,val}` 决定，manifest 记 `selection_split` / `independent_test`。test 选模的结果必须写明"test 参与选模"，不得称独立盲测；跨轮比较只用同一口径，不同口径只作参考。
- 使用本项目 `tools/` 处理日志下载、指标汇表和源码快照。正式指标只取完整 `TEST RESULTS` 区块；保留全部历史记录，直接比较已有单元格。中断留下的 `.dup-*` 目录没有 TEST 块，汇表时会跳过，不要手工补行。
- 服务器操作统一走 `.codex/ssh.py`（单条命令）、`outputs/remote_cmds/run.py`（把本地 py 送上远端执行）与 `outputs/remote_cmds/launch_remote.py`（后台长任务）。连接信息只从 `.vscode/sftp.json` 读取，禁止输出凭据。MSYS/bash 会把命令里的 `/home/...` 参数改写成 `C:/...`，远端路径一律放进命令字符串内部。
- 上机前先 staging：把 `models/` 与本轮脚本打包进 `runtime/`，写逐文件 SHA-256 的 `runtime_manifest.json` 和 `source_manifest.json`；已冻结的 runtime 不再改动，确需改动必须重跑预检并在 STATE 里记原因。
- 正式训练按当前用户授权使用 GPU。按实测显存安排并行，不重复启动队列，不停止无关进程。GPU 被外部任务占满时排队等（如 `wait_and_start.sh`），不抢卡；失败时保留日志和权重，先确定具体原因再重跑。
- 分析、图像和中间特征放入 `outputs/<tag>/<Run>/analysis/`；训练与验证数据用于方案设计。历史冻结 TXT 保持对应轮次的实际源码。
- Git 只提交源码、说明、指标表和规定快照；数据集、权重、凭据与 `outputs/` 不提交（`.gitignore` 已覆盖 `.tmp/`、`.deps/`、`.agent-teams/`、`.dsh-matlab-runner/`、`pip-*/`）。提交前逐项确认 `git status`，不把他人未完成的改动带进去；推送用 `GIT_TERMINAL_PROMPT=0 git push`（凭据助手已在 `.git/config` 指向 Git Credential Manager）。
- 文档与结果同步：每轮跑完更新根 `README.md` 的对应小节、`models/README.md` 的开关与参数表、`docs/experiment_metrics.xlsx` 和 `docs/temporary/` 的 TXT 快照。模型结构图由 `outputs/pptx/build_*.py` 生成（不入库），成图放 `docs/`。
- 定时任务使用桌面应用的自动化工具管理。`.codex/` 是项目工具和说明目录；项目规则由本文件承接，自动化状态以应用中的任务为准。
