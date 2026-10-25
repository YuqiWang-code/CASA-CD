# 当前状态

更新：2026-10-24。

## CASA-Pyramid / Run2

协议casa_pyramid_40k_qk_euclideanv224_v2。用户授权启动四库三组共12项，LEVIR→SYSU→WHU→CDD优先，两张GPU按实测显存补位。当前224-token代码已部署；结构、梯度、池化等价、衰减分组、折叠和profile检查通过，正在各模块LEVIR3epoch、SYSU4epoch真实探针。正式训练必须四项passed后从MobileNetV2预训练新初始化，不使用探针/旧权重，不重复启动。

网格(8,4,4,4)、KV(128,32,32,32)，普通K/V先池化再投影，双曲Q/K加欧式V。投影weight WD1e-4，其余context WD0；主体WD1e-4。脚本train_scripts/CASA-Pyramid/Run2，日志/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run2，checkpoint/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run2，本地outputs/CASA-Pyramid/Run2。

服务器旧Run1模型保存在/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1/archive/server_models_before_Run2。旧171条指标与所有历史源码、日志、权重保留；Run2收齐后正常183条，其他新增一并保留。没有活动自动化，不自行创建。

## CASA-Pyramid / Run1 已完成

协议`casa_pyramid_40k_guide02_v1`，模型与训练源码基线`5730d13`。2026-10-24 15:03:59 CST启动，22:30:52 CST完成，12/12正式测试齐全，无失败。最后核查两GPU均空闲，队列与子进程结束。

原生MobileNetV2四尺度经TAR后做联合跨尺度QKV，不含HFEA。每尺度32个KV、联合128个、四头16维；三组有相同评分辅助监督。QKV训练2.063581M/部署1.952266M，CTRL训练2.047125M/部署1.935810M。

全部日志与JSON已下载至`outputs/CASA-Pyramid/Run1`。指标表171条，原159条历史记录逐单元格保留；本轮冻结TXT使用source_code.tar.gz内实际模型源码，current TXT使用当前工作区。旧CASA/Run6和Run7冻结快照保持原轮次。

本轮相对CTRL，双曲/普通模块在LEVIR分别提高0.031/0.191个百分点，在SYSU提高0.359/0.714个百分点；WHU分别降低0.114/0.297个百分点，CDD降低0.052/0.045个百分点。普通版在LEVIR与SYSU高于双曲版，本轮尚未证明双曲几何具有稳定精度优势。两个模块部署参数均为1.952266M，精度与开销需共同评价。

全部部署折叠检查通过，首级最大绝对差不超过3.25e-6，二值差异均为0。Run1已归档，当前活动轮次见Run2；不恢复历史队列。连接与操作见SERVER.md。

## 产物

- 日志：`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run1`
- 权重：`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run1`
- 本地结果：`outputs/CASA-Pyramid/Run1`
- 当前/冻结TXT：`docs/temporary/models_and_metrics_CASA-Pyramid_current.txt`、`models_and_metrics_CASA-Pyramid_Run1.txt`
- 旧CASA/Run7服务器模型归档：`/home/yqwang/outputs/CASA-CD/CASA/Run7/archive/server_models_before_Pyramid_Run1`

## GitHub公开训练记录

docs/experiments/CASA-Pyramid/Run1保存12项完整日志、逐轮CSV、实际最佳epoch、完整训练/验证曲线、逐轮模块诊断CSV/JSONL、成功探针及环境。导出入口tools/export_training_records.py。模块诊断是每轮验证首个batch的实际测量值，epoch从0开始。
