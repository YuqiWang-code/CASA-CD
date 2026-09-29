# CASAA Run2 — 冻结 ViT 下的 Oracle 机制诊断（重大前置发现后修订）

> 决策依据：`docs/temporary/CASA-CD_CASAA_Run1复盘与下一步实验决策.md`（§4 Oracle 实验），
> 并经 2026-09-28 服务器实测修订（见下「前置发现」）。

## ⚠️ 前置发现：ChangeViT 官方协议会把 ViT 训练成零权重（Run1 全部结果受影响）

用 snapshot 训练实测（真实 LEVIR、官方协议、baseline 模式）：

| step | pos_embed norm | patch_embed norm | blocks.0.qkv norm | resnet.conv1 norm |
|---|---:|---:|---:|---:|
| 0 | 4.43 | 8.01 | 19.17 | 12.58 |
| 400 | 1.70 | 7.53 | 14.97 | 12.51 |
| 800 | 0.055 | 5.58 | 8.79 | 12.49 |
| 1200 | 0.012 | 1.10 | 3.26 | 12.46 |
| 1600 | **0.0007** | **0.029** | **1.12** | 12.43 |

- ViT 在 ~1600 steps（约 3.6 epoch）内被训练成**精确零权重**（Adam 对小梯度做
  全步长更新 + 权重归零后梯度消失 = 吸收态）；ResNet/decoder 完全不受影响。
- 保存/加载往返无损（in-mem == file），是**训练动力学崩溃**，不是文件损坏。
- 核对 Run1 全部 checkpoint：LEVIR/CDD/WHU（baseline+A1+A2）ViT 150/150 键全零、
  optimizer 矩也全零（从未收到梯度）；SYSU best（epoch 9）健康、last（epoch 107）
  106/150 键已零——与 SYSU-A2 的 F1 在 epoch 9 后从 0.8235 衰减到 0.8106 吻合
  （崩溃对 SYSU 是有害的）。
- **结论**：Run1 在 LEVIR 上的 A0/A1/A2 全部是「死 ViT」模型（ViT 输出恒零，
  模型退化为 ResNet+decoder），LEVIR 的 CASAA 结论无效；SYSU 仅 best 检查点有效。
  Run1 的「A1≈baseline 压缩无损」结论必须用活 ViT 重新验证。

## Run2 设计（修订后）

- **冻结 ViT**（`--freeze_vit 1`）：路由机制作用于健康的 pretrained DeiT-Tiny 特征，
  保证 CASAA 的「上下文选择」确实改变了 ViT 的前向（注意力在冻结权重下仍被
  routing 改变）。ResNet18/decoder 照常训练。
- **4 个 run**（同设置、唯一变量 = router）：
  - `A1_SAA_FROZEN`：content-only 对照（K=64 全部内容聚类）——新的有效对照；
  - `A3_ORACLE_FROZEN`：**DIAGNOSTIC-ONLY**，change score = GT patch occupancy
    （label gray≥128 二值化后 avg_pool 16×16），训练/测试都用 GT，**不可部署**。
- 其余不变：blocks 8-11、N=256、K=64、Kc=32、Kb=32、keep_ratio 0.25、
  change_share 0.50、共享确定性 density-peak 聚类、qkv/proj 预训练原位继承、
  BCE+Dice、poly、80000 steps、batch 16、256×256、seed 16、test-as-val。
- 已实现的等价优化：qkv 切片投影（只算 full-Q 与 compressed-KV，smoke 等价误差
  <1e-6），并新增 `analyse/casaa_router_diagnostic.py`（Router Audit）。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/CASAA/Run2/{A1_SAA_FROZEN,A3_ORACLE_FROZEN}/<dataset>/`
- 日志：`/home/yqwang/outputs/CASA-CD/CASAA/Run2/{A1_SAA_FROZEN,A3_ORACLE_FROZEN}/<dataset>/train_log.txt`
  （oracle 日志头部含 `[DIAGNOSTIC-ONLY] oracle routing uses GT and is not deployable`）

## 启动（GPU1 专属；GPU0 让给其它项目）

```bash
cd /home/yqwang/projects/CASA-CD/train_scripts/CASAA/Run2
nohup bash run_queue.sh > /dev/null 2>&1 &
# 顺序：A1_LEVIR -> A3_LEVIR -> A1_SYSU -> A3_SYSU（单卡串行）
```

## 成败判据（Oracle 相对 A1_FROZEN，而非 Run1 的旧数字）

- **通过**：至少一个数据集 F1 ≥ +0.30 且另一数据集不降超 0.15，IoU 同向，
  LEVIR 上 Recall 改善 → 机制有价值，瓶颈在 deployable change signal → 可部署 v2
  （detail score 或 adaptive quota，二选一）。
- **失败**：Oracle ≈ A1_FROZEN（两数据集都 ≤ +0.15）→ **停止 change-aware router
  迭代，转主线二**。
