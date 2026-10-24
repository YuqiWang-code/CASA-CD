# 当前状态

更新：2026-10-24。

## CASA-Pyramid / Run1

协议`casa_pyramid_40k_guide02_v1`。用户授权去HFEA、原生四尺度经TAR融合后做联合跨尺度QKV。新脚本`train_scripts/CASA-Pyramid/Run1`，入口`models/train_CASA.py`。

三组均有预测评分辅助监督；每尺度32个KV，联合128个，完整Q、四头16维，普通欧氏/双曲几何匹配预算。双曲与普通训练2,063,581、部署1,952,266参数，CTRL训练2,047,125、部署1,935,810参数。

结构、几何、梯度、共同初始化、优化器/RNG、部署折叠与显存检查通过。真实探针在两GPU运行，各模块LEVIR3epoch、SYSU4epoch。四探针通过后正式从MobileNetV2预训练全新初始化，12项，LEVIR→SYSU→WHU→CDD优先，两卡按实测显存补位。不得重复启动。

本轮日志`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`，本地`outputs/CASA-Pyramid/Run1`。状态用`python .codex/status.py --tag CASA-Pyramid --run Run1`。

## 历史与指标

旧CASA/Run7十二项完成并收集，指标159条，旧147条逐单元格保留。旧源码基线2e60998，原服务器models保存在`/home/yqwang/outputs/CASA-CD/CASA/Run7/archive/server_models_before_Pyramid_Run1`；对应source_code.tar.gz和冻结TXT保留。旧队列不恢复。

本轮收齐后应171条正式记录，其他新增一并保留。新冻结TXT用本轮归档，当前TXT用工作区。没有活动自动化，不自动创建监控。用户后续指令优先。
