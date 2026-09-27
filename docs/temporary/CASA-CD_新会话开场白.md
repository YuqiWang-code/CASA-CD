# CASA-CD 新会话开场白

> 用途：在 DSH Web GUI 里新开会话（同一工作区）后，把下面这段作为第一条消息粘贴，
> 让新会话快速恢复项目上下文。旧会话历史仍在左侧会话列表可随时切回。

```text
继续 CASA-CD 项目（遥感图像二值变化检测，全监督；baseline ChangeViT-T 已复现完成，4 数据集指标见 README）。

先读以下文件恢复上下文：
- README.md
- docs/RSML-3_服务器环境与变化检测数据统一说明.md
- docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md
- docs/temporary/models_and_metrics_baseline_Run1.txt（当前 models 源码快照 + 指标）
- others/SAT/saa.py 与 others/SAT/README.md（SAT 核心机制提取，CASAA 的模板）

常用约定：
- 方法创新导向，不做工程任务；损失 BCE+Dice、poly LR、max_steps=80000、seed 16、test 集当验证集
- 服务器 RSML-3：环境 casacd，代码 /home/yqwang/projects/CASA-CD，
  数据集 /share_datasets/CD，checkpoint /share_datasets/yqwang/checkpoints/CASA-CD，
  日志 /home/yqwang/outputs/CASA-CD；本地部署用 .claude/_deploy.py，监控用 .claude/_monitor.py
- 每个 GPU（RTX 5090 32GB）最多一个 batch16 任务（ChangeViT-T 单任务 ~15.7GB）

本次任务：<在此填写新任务>
```
