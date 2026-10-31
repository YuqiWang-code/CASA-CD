# CASA-Pyramid Run10

正式配置采用空间中心化读出与区域后验监督，协议`casa_pyramid_40k_supervisedposterior_v10`。H_POSTERIOR、E_POSTERIOR、M0_CTRL各四库，共12项；物理GPU1按实测容量填位，优先LEVIR→SYSU→WHU→CDD。

## 模型与共同训练设置

完整Q、224个KV。P0用guide平方置信权重，局部几何修正得到`r=sigmoid(guide_logit+delta)`，P1按r/1−r一阶池化，K/V共享P1。读出采用`R−mean_spatial(R)`，去掉直接减query value的路径；原TAR残差保留。H/E在后验修正及最终检索两处分别使用双曲/匹配欧式距离。

- H/E：`main + .05 raw-guide BCE + .05 posterior BCE`；CTRL：`main + .1 raw-guide BCE`。
- retrieval=0，压缩温度初值固定0.5；H/E温度参数参与学习，CTRL无该参数。
- 预训练全新初始化、batch32、seed/context seed2333、40K按整epoch取整、Adam lr5e-4、encoder倍率1。
- 正式每epoch在test选模，结果标注test参与选模。参数衰减、增强和预处理沿用共同配置。

## 六项配对短训来源

| 探针 | 实际源码 | guide目标 |
|---|---|---|
| H_CENTER_RAW | selection/centered/raw/models | .1 raw BCE |
| H_CENTER_SUP | selection/centered/supervised/models | .05 raw + .05 posterior BCE |
| E_CENTER_SUP | selection/centered/supervised/models | 同上，匹配欧式 |

每组LEVIR/SYSU各4完整epoch，train/val划分，seed2333，train loader2333/val loader2334。H05/E05历史对照保留在`selection/classposterior/probes`，CTRL在`selection/neighborhood/probes`；不重训历史对照。实际运行的隔离脚本位于`selection/centered/scripts`。

公开`probe.py`、`probe_config.py`、`health_probe.py`复现这6项来源和预算；读取归档实测profile安排GPU1最多3路。调用`bash train_scripts/CASA-Pyramid/Run10/probe.sh`；完成项直接引用，活动的隔离runner存在时拒绝重复启动。

history分别记录raw/posterior train BCE及完整val辅助BCE。guide_loss是源码实际objective分量，RAW与SUP含义不同；应比较分开的BCE字段。task_gradients记录主分割loss到4层实际r及g+delta节点的梯度，posterior_gradients另外记录posterior BCE到Q/K、温度、guide的梯度，并标明实际训练权重。

## 正式验证与启动

```bash
bash train_scripts/CASA-Pyramid/Run10/validate.sh
bash train_scripts/CASA-Pyramid/Run10/launch.sh
```

validate检查几何、类别压缩、中心化恒等式、监督预算/梯度、模型、部署折叠与续训，并以固定温度0.5和选中SUP源码profile三组。正式启动要求`selection/centered/probes`下6份实际health及manifest完整，以及根目录`review_decision.json`明确记录：

```json
{
  "protocol": "casa_pyramid_40k_supervisedposterior_v10",
  "guide_objective": "half_raw_half_posterior_area_bce",
  "compression_temperature_init": 0.5,
  "passed": true
}
```

review还应保存配对实测和机制分析证据。该文件由结果判断产生；短训runner不会自动创建正式通过决策。readiness同时记录实际探针、对照及SUP profile，原始RAW/SUP协议保持各自归属。不存在温度搜索或由单个F1阈值自动启动的路径。

唯一正式调度器`pipeline.py`，允许跨库填充GPU1空闲容量，留量1GiB。首次正式启动保存实际模型和Run10脚本的源码归档及环境。日志根目录`/home/yqwang/outputs/CASA-CD/CASA-Pyramid/Run10`；checkpoint根目录`/share_datasets/yqwang/checkpoints/CASA-CD/CASA-Pyramid/Run10`。

```bash
bash train_scripts/CASA-Pyramid/Run10/stop.sh
bash train_scripts/CASA-Pyramid/Run10/test_one.sh H_POSTERIOR LEVIR-CD-256
```

## 正式结果收集

```powershell
python train_scripts/CASA-Pyramid/Run10/collect_results.py
python tools/export_training_records.py --tag CASA-Pyramid --run Run10
```

只采纳完整TEST RESULTS。263条历史逐单元格保留，本轮12项收齐应至少275条；探针和未进入正式训练的候选不入表。冻结TXT使用实际`source_code.tar.gz`，README使用`CASA_PYRAMID_RUN10_RESULTS_START/END`区域。

## 比较范围

短训用于区分中心化读出和后验监督的作用。正式H/E对照匹配结构、预算和监督目标，CTRL保留相同总guide预算。单seed、test选模与四轮短训的局限分别记录；短训数值健康不等于完整预算下的精度优势。
