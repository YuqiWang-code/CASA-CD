# CASA-MoE / Run3 审查与启动记录

协议`casa_moe_40k_candidate_evidence_v3`。LEVIR/SYSU各动态、TAR-only、静态三项，共六项，GPU0单路串行。历史Run2已完成并冻结。

## 结构依据

Run2的动态版相对TAR在SYSU提高0.9981pp，LEVIR提高0.0405pp。LEVIR边界F1提高，但召回降低。H/E在SYSU最佳checkpoint四层残差余弦0.99815–0.99947，当前内容高度相似。证据来自首个选模batch，不等于全数据集统计。

路由选择X为底座的三个候选和独立TAR，却只读取X与粗层DCR；Run3将LN(TAR)加入路由输入，新增4096参数。该改动让候选选择直接依赖TAR内容，是否提高精度由同协议对照评价。

H/E共享无偏置输出投影和逐通道增益，故

`gH·RH+gE·RE = gamma·out((gH·AH+gE·AE)·V − (gH+gE)·Vq)`。

正常训练和推理采用一次AV/out，诊断按专家分开读出。保留对路由、QK、V、out、gamma和温度的梯度。小波与共同训练设置保持一致。

训练数据处理、损失与选模实现审查未发现明确错误。SYSU后期训练损失下降、选模损失上升在三配置均出现，继续使用最佳checkpoint。几何距离、Haar布局与部署路径数值检查通过。

## 检查

CPU合并/分离读出最大差1.67e-16，全输入/参数/路由梯度最大差3.34e-15；三配置部署折叠最大差1.20e-7。TAR新增输入的任务梯度、四级输出、共享初始化和小波重建均通过检查。详见`model_tests.json`。

动态版训练/部署参数2.224247M/2.112676M。GPU0 batch32进程峰值13156MiB，包含训练任务梯度与整batch评价诊断。GPU与其他任务共享，profile的延迟不能作为独占硬件性能结论。

六项3epoch train/val探针全部通过，结果见`probe_summary.json`和各探针实际记录。动态版主损失在LEVIR由2.140降至1.083，SYSU由2.344降至1.524；末轮val F1为85.71%/79.90%。四层TAR路由输入、H/E及小波均有非零任务梯度，未见非有限值。探针不作为正式结果。

同一随机权重部署模型、非零gamma、batch1双时相256、FP32、六轮交替顺序测速，合并/分离读出的轮中位数汇总为9.279/9.620ms。运行时与其他任务共享GPU，仅作为本次读出实现对照，详见`readout_benchmark.json`。

## 正式启动

2026-10-31 13:21:47 CST启动，supervisor PID3230536，首项LEVIR动态融合PID3230540。固定GPU0单路，DYNAMIC/TAR/STATIC按LEVIR→SYSU顺序执行。GPU0预算16012MiB，动态/TAR/静态预留13156/9164/12966MiB。源代码提交`734940a`，运行使用冻结副本。

正式训练从预训练骨干重新初始化，不使用探针权重。每epoch test F1选模，最终结果不是独立盲测。当前正式结果0/6。

LEVIR动态版正式epoch0–3主损失2.144638/1.192744/1.075096/1.029927，选模F1为80.1962/83.7849/85.8983/83.7938%。早期评价有波动，训练损失下降，检索、小波及router任务梯度均非零且有限，无NaN或OOM。前三轮诊断见`initial_health.json`，初始日志保存为`M_D_DYNAMIC/LEVIR-CD-256/initial_train_log.txt`；它不是完整训练日志。

## 产物

源码与环境见`runtime.json`；实际源码归档位于`outputs/CASA-MoE/Run3/source_code.tar.gz`。服务器日志`/home/yqwang/outputs/CASA-CD/CASA-MoE/Run3`，checkpoint`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MoE/Run3`。

[训练与收集脚本](../../../../train_scripts/CASA-MoE/Run3/README.md) · [探索线说明](../../../../README_MoE.md)
