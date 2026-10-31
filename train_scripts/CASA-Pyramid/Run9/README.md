# CASA-Pyramid Run9

四库检验一次区域内内容关联精炼（β=1）的泛化及其几何贡献。

| 变体 | 模块 | β | 数据集 |
|---|---|---:|---|
| H_REFINE | Lorentz Q/K、欧式 V | 1 | LEVIR、SYSU、WHU、CDD |
| E_REFINE | 匹配的欧氏空间坐标 Q/K、欧式 V | 1 | 四库 |
| M0_CTRL | TAR 直通、共同 guide 辅助头 | 0 | 四库 |
| E_BASE | 匹配欧氏 Q/K、欧式 V | 0 | SYSU、WHU |

14 项全新正式训练。SYSU/WHU 的 Run8 H_NONE 作为外部 H β=0 参照，保留原轮次和记录；Run9 不重复计入指标。

## 固定协议

`casa_pyramid_40k_refinement_v9`；MobileNetV2 预训练、独立初始化，seed/context seed=2333，batch32、Adam、40K步按整epoch上取整、encoder LR倍率1。所有组 guide=0.1、retrieval=0；数据处理、学习率与权重衰减沿用共同设置。正式每epoch在 test 选模，最终结果标注 test 参与选模。短训使用 train/val，检查数值与主分割任务梯度。

优先级 LEVIR→SYSU→WHU→CDD；只用物理 GPU1，按四变体实测显存和1GiB余量并行补位。每项使用自己的日志、checkpoint、placement.json；异常保留产物并停止本队列，禁止自动重新开始失败训练。

## 启动

在服务器 `/home/yqwang/projects/CASA-CD` 执行：

```bash
mkdir -p /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9
bash train_scripts/CASA-Pyramid/Run9/validate.sh > /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9/validation.log 2>&1
bash train_scripts/CASA-Pyramid/Run9/probe.sh > /home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9/probes.log 2>&1
bash train_scripts/CASA-Pyramid/Run9/launch.sh
```

验证覆盖双曲几何、H/E 匹配、β=1 梯度、时相交换/同输入、部署折叠、续训和调度矩阵，并测量四种配置。随后运行六项2完整epoch探针：SYSU/WHU H_REFINE/E_REFINE，以及 SYSU E_BASE/CTRL。所有项目通过后写 `readiness.json`；正式训练不加载探针权重。首次启动归档实际模型和本轮脚本到 `source_code.tar.gz`，保存环境记录。日志在 `/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run9`，权重在 `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run9`。

```bash
bash train_scripts/CASA-Pyramid/Run9/stop.sh
bash train_scripts/CASA-Pyramid/Run9/test_one.sh H_REFINE SYSU-CD-256
```

复测写入独立 `retests/`，不替代正式训练日志。

冻结权重分析采用对应构造参数；`analyze_frozen.py` 和 `content_diagnostics.py` 均需显式传入 `--module hyperbolic --refinement-beta 1`（H_REFINE）、`--module matched --refinement-beta 1`（E_REFINE）或 `--module matched --refinement-beta 0`（E_BASE），其余路径与采样参数见各脚本 `--help`。

## 收集与比较

本地保留服务器的 `source_code.tar.gz` 后执行：

```powershell
python train_scripts/CASA-Pyramid/Run9/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run9
```

Collector 仅采纳完整 TEST RESULTS，逐单元格保留263条既有正式记录；本轮14项齐全时应至少277条。冻结TXT从实际源码归档提取，README使用 `CASA_PYRAMID_RUN9_RESULTS_START/END` 区域。矩阵以 `config.py` 的 `experiments()` 为准，pipeline状态写同一名单。

H_REFINE 对 E_REFINE 比较相同细化下几何差异；E_REFINE 对 E_BASE 衡量欧氏细化；SYSU/WHU H_REFINE 对 Run8 H_NONE 检查双曲细化收益；四库模块对 CTRL 评估整体上下文价值。单seed结果反映本轮效果，不能替代多seed稳定性结论。
