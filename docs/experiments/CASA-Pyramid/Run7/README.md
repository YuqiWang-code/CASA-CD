# CASA-Pyramid / Run7

2026-10-28 16:00:08 CST启动，源码基线`8a54fa2`，协议`casa_pyramid_40k_retrieval_v7`。物理GPU0运行四库双曲/CTRL共8项，按实测显存补位；正式从预训练骨干全新初始化，每epoch test F1选模，最终复测非独立盲测。

16项train/val短训完成后采用224KV、一个32维直接Lorentz投影、同内容K/V、独立内容门控和类别检索监督。[筛选结果](selection/selection_summary.md) · [检索内容证据](selection/retrieval_evidence_summary.md) · [模型公式](../../../../models/HYPERBOLIC_DESIGN.md)。

## 初期观察

记录时间：2026-10-28T16:07:07+08:00。三项首批实验均已完成至少3个训练与test选模epoch；损失有限，未记录NaN、CUDA OOM或进程错误。双曲各层有非零半径差异和残差输出。

| 实验 | 已完成训练epoch | 最新已完成test选模F1 (%) | PID |
|---|---:|---:|---:|
| M1_HYPER_QKV/LEVIR-CD-256 | 6 | 86.6271 | 2274059 |
| M0_CTRL/LEVIR-CD-256 | 7 | 88.3312 | 2274086 |
| M0_CTRL/SYSU-CD-256 | 4 | 82.7918 | 2274118 |

这些是初期学习和数值健康记录，尚无完整正式测试。SYSU后期泛化与四库增益继续由完整训练检验。

## 效率

RTX5090、双时相256×256、batch1、FP32，cuDNN TF32开启、matmul TF32关闭，30次预热和100次测量中位数；随机权重结构成本。

| 变体 | 训练参数M | 部署参数M | 中位延迟ms | 峰值allocated MiB |
|---|---:|---:|---:|---:|
| CTRL | 2.047125 | 1.935810 | 3.863 | 54.76 |
| 双曲 | 2.058409 | 1.947094 | 6.918 | 73.90 |

双曲保留约2M部署参数；相对CTRL仍有执行成本，精度收益以完整测试评估。[原始测量](inference_benchmark.json) · [训练显存](profile.json) · [部署检查](validation.log) · [环境](environment.json) · [初期记录](startup_health.json)。

## 产物

运行源码归档和完整产物保存在`outputs/CASA-Pyramid/Run7`及服务器同轮目录；当前正式指标0/8。Run7源码TXT使用实际source_code.tar.gz，current TXT使用工作源码，历史221条指标保留。
