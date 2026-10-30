# CASA-MoE / Run1

共享 MobileNetV2 → 四级 TAR → 动态专家上下文 → DCR。TAR 原始变化特征作为基础专家，双曲检索、欧式检索和 Haar 频率分支提供增量，像素级 softmax 路由采用不同的增量。此轮为 dense MoE，所有启用专家都执行，不宣称稀疏计算加速。

## 实验矩阵

| 变体 | 模式 | LEVIR | SYSU | WHU | CDD |
|---|---|---|---|---|---|
| M4_MOE | 四专家动态路由 | ✓ | ✓ | ✓ | ✓ |
| M0_CTRL | TAR 基础路径 | ✓ | ✓ | ✓ | ✓ |
| M4_EQUAL | 四专家固定等权 | ✓ | ✓ | ✓ | ✓ |
| M1_HYPER_QKV | TAR + 双曲增量 | ✓ | ✓ | — | — |
| M2_CA_QKV | TAR + 欧式增量 | ✓ | ✓ | — | — |
| M3_FREQUENCY | TAR + 频率增量 | ✓ | ✓ | — | — |
| M4_NO_HYPER | 动态路由移除双曲 | ✓ | ✓ | — | — |
| M4_NO_QKV | 动态路由移除欧式 | ✓ | ✓ | — | — |
| M4_NO_FREQUENCY | 动态路由移除频率 | ✓ | ✓ | — | — |

合计 **24 组**。LEVIR 和 SYSU 的九组实验检验专家贡献、动态路由相对固定融合的收益以及删除专家的影响；WHU 和 CDD 比较基础路径、等权融合与动态路由。按 LEVIR → SYSU → WHU → CDD 优先，在每库中先启动动态 MoE、CTRL 和等权组，允许容量补位。

## 共同协议

- 协议 `casa_moe_40k_dense_v1`，物理 **GPU1**，每个进程看到逻辑 GPU0。
- batch 32、seed 2333、40,000 step 按完整 epoch 上取整；每组从 MobileNetV2 预训练独立初始化。
- 主体训练设置和输入预处理由 `models_v1/train_CASA.py` 固定。四级深监督和 guide 权重保持共同配置。
- 正式训练每 epoch 在 test 选模，正式 TEST 为所选 checkpoint 的复测；**test 参与选模，不是独立盲测**。三个 epoch 的健康探针使用 val，不写正式指标。
- 九种变体均实测训练峰值显存和步耗时；调度保留 1 GiB 显存余量，同时检查当前实际可用显存。只有本探索线的进程受调度管理。
- 完成模型检查、剖析和 LEVIR/SYSU 的三轮真实数据探针后启动。探针确认损失有限且改善、分割有有效输出，三个增强专家与路由器均得到分割任务梯度。

## 文件与运行

服务端输出 `/home/yqwang/outputs/CASA-CD/CASA-MoE/Run1`，checkpoint 位于 `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run1`。`runtime/<时间>/` 冻结本轮 `models_v1` 和启动脚本；测试、探针和正式训练均使用该副本。`source_code.tar.gz`、`runtime.json` 保留源码及启动环境。

```bash
bash train_scripts/CASA-MoE/Run1/validate.sh
bash train_scripts/CASA-MoE/Run1/probe.sh
bash train_scripts/CASA-MoE/Run1/launch.sh
bash train_scripts/CASA-MoE/Run1/test_one.sh M4_MOE LEVIR-CD-256
```

`stop.sh` 停止此探索队列与其子进程。未完成训练不自动重启，日志和权重保留。`evaluate.py` 使用冻结源码复测指定 checkpoint，结果写入 `retests/`。

本地收集：

```powershell
python train_scripts/CASA-MoE/Run1/collect_results.py
```

调用项目现有下载、Excel 和 TXT 工具，仅收集完整 `TEST RESULTS`；旧指标逐单元格比较。快照从本轮源码归档提取 `models_v1`，不会取后来修改的工作目录代码。完整日志保存在 `outputs/CASA-MoE/Run1`。

<!-- CASA_MOE_RUN1_RESULTS_START -->

状态更新：2026-10-30 17:32:19 CST；队列 `complete`；正式结果 **24/24**。

| 数据集 | 实验 | 状态 | F1 (%) | IoU (%) |
|---|---|---|---:|---:|
| LEVIR | `M4_MOE` | 完成 | 90.91 | 83.33 |
| LEVIR | `M0_CTRL` | 完成 | 91.11 | 83.67 |
| LEVIR | `M4_EQUAL` | 完成 | 91.19 | 83.81 |
| LEVIR | `M1_HYPER_QKV` | 完成 | 91.16 | 83.75 |
| LEVIR | `M2_CA_QKV` | 完成 | 91.00 | 83.49 |
| LEVIR | `M3_FREQUENCY` | 完成 | 91.04 | 83.56 |
| LEVIR | `M4_NO_HYPER` | 完成 | 91.04 | 83.56 |
| LEVIR | `M4_NO_QKV` | 完成 | 90.98 | 83.45 |
| LEVIR | `M4_NO_FREQUENCY` | 完成 | 91.18 | 83.79 |
| SYSU | `M4_MOE` | 完成 | 82.75 | 70.58 |
| SYSU | `M0_CTRL` | 完成 | 83.43 | 71.58 |
| SYSU | `M4_EQUAL` | 完成 | 83.14 | 71.14 |
| SYSU | `M1_HYPER_QKV` | 完成 | 83.07 | 71.04 |
| SYSU | `M2_CA_QKV` | 完成 | 82.78 | 70.62 |
| SYSU | `M3_FREQUENCY` | 完成 | 82.47 | 70.17 |
| SYSU | `M4_NO_HYPER` | 完成 | 83.26 | 71.32 |
| SYSU | `M4_NO_QKV` | 完成 | 82.76 | 70.59 |
| SYSU | `M4_NO_FREQUENCY` | 完成 | 82.92 | 70.82 |
| WHU | `M4_MOE` | 完成 | 94.43 | 89.45 |
| WHU | `M0_CTRL` | 完成 | 94.63 | 89.81 |
| WHU | `M4_EQUAL` | 完成 | 94.63 | 89.81 |
| CDD | `M4_MOE` | 完成 | 96.89 | 93.97 |
| CDD | `M0_CTRL` | 完成 | 96.93 | 94.05 |
| CDD | `M4_EQUAL` | 完成 | 96.91 | 94.01 |

<!-- CASA_MOE_RUN1_RESULTS_END -->
