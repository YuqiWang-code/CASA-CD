# CASA-Pyramid Run10：几何归属与空间检索

协议 `casa_pyramid_40k_supervisedposterior_v10`。GPU1，四库H_POSTERIOR/E_POSTERIOR/M0_CTRL共12项，优先LEVIR→SYSU→WHU→CDD。正式队列待启动，0/12项完整TEST。

## 配对证据

六项四轮train/val对照通过数值、任务梯度、部署与显存检查。[完整比较与原始记录](selection/centered/probe_comparison.md)保存末轮、epoch2–4均值、边界、损失及冻结干预。

| 数据集 | H_SUP末轮F1 | CTRL末轮F1 | H_SUP边界1px F1 | CTRL边界1px F1 |
|---|---:|---:|---:|---:|
| LEVIR | 86.5958 | 85.6555 | 70.4766 | 66.4200 |
| SYSU | 80.4059 | 79.7279 | 22.4425 | 21.8636 |

H_SUP相对匹配E的末轮F1分别高1.0935/0.4084pp。固定256张val，H_SUP完整模型相对关闭全局路径高0.0344/0.2275pp；打乱注意力的查询位置后低0.1012/0.2374pp。

## 结构和训练

P0由guide平方权重构造；区域内变化/背景原型的相对距离修正guide logits得到r。P1采用r及1−r的一阶权重，K/V共享P1。完整Q检索224个跨尺度KV，写回`gamma * Wout[gate * (AV−mean_hw(AV))]`，保留TAR与局部细节路径。

H/E为主损失＋0.05原guide BCE＋0.05后验BCE；CTRL为主损失＋0.1原guide BCE。压缩温度初值0.5，旧跨尺度检索BCE为0。40K整epoch预算、batch32、seed2333、预训练骨干全新初始化。正式每epoch test选模，非独立盲测。

## 推理开销

RTX5090、双时相256×256、batch1、FP32、卷积TF32开启，随机权重架构基准：

| 组 | 部署参数M | 中位延迟ms | 峰值allocated MiB |
|---|---:|---:|---:|
| CTRL | 1.935810 | 3.875 | 54.76 |
| H | 1.947095 | 9.452 | 74.26 |
| E | 1.947095 | 8.527 | 62.90 |

[原始基准](inference_benchmark.json) · [显存profile](profile.json) · [验证记录](validation_result.json) · [完整预算准入判断](review_decision.json)。

## 已知限制

四轮单seed结果不代表最终收益。H_SUP的epoch2–4均值略低于H_RAW；SYSU末轮仍低于旧H_BASE。后验关闭与压缩度量替换的独立影响较小。LEVIR全局路径减少漏检但增加误报，F1改善并不等于总错误数下降；batch8干预和batch1内容统计单独记录。参数增量小，推理延迟仍明显高于CTRL。

[Run9停止时诊断](diagnosis/README.md) · [模型公式](../../../../models/HYPERBOLIC_DESIGN.md) · [训练脚本](../../../../train_scripts/CASA-Pyramid/Run10/README.md)。正式结果仅从完整TEST RESULTS入表，现有263条记录保留。
