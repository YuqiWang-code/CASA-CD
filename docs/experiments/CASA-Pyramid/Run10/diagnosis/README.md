# Run10 设计依据：Run9 冻结验证诊断

日期：2026-10-31。冻结权重为 Run9 `H_REFINE/LEVIR-CD-256/best_model.pth`，运行源码为该轮 `runtime_source/models`。使用固定 256 张 LEVIR val 图像；这些诊断不作为正式 TEST 指标。

| 文件 | 内容 | 评价口径 |
|---|---|---|
| [hyperbolic_interventions.json](hyperbolic_interventions.json) | 关闭全局/局部路径、固定/打乱半径及注意力干预 | batch8，FP32 |
| [hyperbolic_content.json](hyperbolic_content.json) | 压缩内容、像素纠错及新增错误、样本列表 | batch1，FP32 |

完整路径 F1 为 91.0789%，关闭全局路径为 91.1318%，关闭局部路径为 90.8255%，固定半径为 91.0876%。逐像素统计中，全局路径纠正 6072 个错误、新增 6694 个错误。两份记录的 batch 不同，阈值附近存在微小数值差异，计数和 F1 分别按各自文件解释。

此 checkpoint 的全局上下文尚未形成净纠错收益，局部路径的收益更明确。诊断描述停止时权重在该 val 子集上的行为，不代替完整训练结果。方案与数学定义见[邻域几何压缩](../../../../temporary/过去的想法/CASA_Pyramid_Run10_邻域几何压缩方案.md)。
