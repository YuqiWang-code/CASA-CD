# CASA-CD Run4：R4-2d PSD_DETAIL 最终结果与轻量 detail 路线终止记录

> **日期**：2026-09-30（R4-D0 判定 C 后按修订方案执行的唯一一次 R4-2d 训练）
> **审查基准**：本地 repo = GitHub `YuqiWang-code/CASA-CD` main（本次会话将提交为 `update code`）
> **固定协议**：BCE+Dice、Adam 2e-4、poly+200 warmup、80000 steps、batch 16、256×256、
> seed 16、test-as-val、GPU1、ViT frozen。
> **结果纪律**：正式数字只认 train_log.txt 最后一个完整 TEST RESULTS 区块。

---

## 0. 结论先行

```text
R4-2d PSD_DETAIL / SYSU（唯一变量 = detail branch：PSD-Detail 0.078M）
F1 = 0.8202   IoU = 0.6952
Gate（F1 ≥ 82.47 且 IoU ≥ 70.11，相对 R4-1 82.77/70.61）：FAIL
ΔF1 = −0.75，ΔIoU = −1.09
```

按预注册停止规则 **Stop-3**：

> **停止当前轻量 detail 路线。不启动 SABI。也不启动 DFPD。**
> 前端 detail 已经没有足够性能 margin，继续连续砍 FI/decoder 不具有合理成功概率。

---

## 1. 正式结果（最后一个完整 TEST RESULTS 区块）

```text
[MODE] baseline   [FREEZE-VIT] 1   [VIT-LR-RATIO] 1.0
[TOTAL-PARAMS] 5.491 M   [EFFECTIVE-PARAMS] 5.491 M   [TRAINABLE-PARAMS] 3.514 M
[FLOPS] 11.5239 G   (input 2x3x256x256, unsupported_ops=9)
Recall=0.7978 | Precision=0.8439 | OA=0.9175 | F1=0.8202 | IoU=0.6952 | Kappa=0.7668
[BEST-F1] 0.8202 (epoch 64)
```

## 2. Run4 detail 换代完整对照（全部 SYSU、frozen ViT4、唯一变量 = detail branch）

| Run | detail branch | F1 | IoU | ΔF1 vs R4-1 | ΔIoU vs R4-1 | Params | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| R4-1 | ResNet18 C2-C4（ImageNet 预训练） | **82.77** | 70.61 | — | — | 8.195M | 24.10G |
| R4-2 | LightDetail 32/64/128 + 1×1 adapters | 82.30 | 69.92 | −0.47 | −0.69 | 5.492M | 10.73G |
| R4-2b | LightDetail 48/96/160 + 1×1 adapters | 82.32 | 69.95 | −0.45 | −0.66 | 5.533M | 10.98G |
| R4-2d | **PSD-Detail 0.078M**（pretrained stem + residual DS） | 82.02 | 69.52 | **−0.75** | **−1.09** | 5.491M | 11.52G |

## 3. 训练证据（决策文档 §26 要求补查）

- best_epoch = 64 / 106（60% 处），best-last gap = 0.8202 − 0.8056 = **0.0146**；
- F1 曲线从 epoch 10 起就在 0.79–0.82 带内震荡，**最后 20% steps 无单调上升**；
- 0 retry、无 OOM、冻结 ViT checksum 正常（dry run 已验证）。

→ **「80K steps 不足」假设不被支持**：这是表达能力/接口的 plateau，不是欠训练。

## 4. 解读：PSD 为什么反而低于 LightDetail（−0.28 F1）

R4-D0 audit 在 feature 层面预测「深层 raw 表达力不足 → pretrained stem + residual
+ MixDown 应能修复」。端到端结果相反：PSD（82.02）比随机初始化 LightDetail
（82.30）还低 0.28。未验证的候选解释（**不构成新实验授权**）：

1. **可学习 adapter 的作用被低估**：LightDetail 的 43K 1×1 adapters 提供了 detail→FI
   的接口学习自由度；PSD 直接输出 64/128/256 裸特征，接口统计（零均值/负值比例）
   与 FI 的 LayerNorm 假设不匹配（audit 已显示 adapter 会把特征变成 ~50% 负值，
   而 ResNet/PSD raw 输出几乎全非负——两者都没有学到这个归一化）。
2. **pretrained stem 在统一 lr=2e-4 + wd=1e-4 下漂移**：ImageNet 先验可能在前几
   千步就被重写（本轮未设独立 stem LR，决策文档 §14 明确不引入该技巧）。
3. **通道分布差异**：PSD 深层 256ch 在 1/8 尺度 + 无 adapter，与原 FI 训练过的
   统计结构不匹配。
4. **单 seed 噪声**：±0.1–0.2 F1 属本协议噪声量级（R4-2 vs R4-2b 差异 +0.02 已
   显示单 seed 分辨率有限），0.28 的差距接近但仍超出噪声。

无论如何：**连续三个 detail 设计（Light32 / Light48 / PSD）都无法在旧 FI+decoder
接口下把 detail 换代损失压回 ≤0.30**，证据已经足以否定「<0.25M 自定义 DW/residual-DW
detail 分支可以保留足够精度」这一路线假设。

## 5. 停止执行内容（Stop-3，严格遵守）

- ❌ 不启动 R4-3 SABI；
- ❌ 不启动 R4-4 DFPD；
- ❌ 不做 PSD 96/192 / PSD deeper / 更多 pretrained block / adapter 修补 / width sweep；
- ❌ 不改训练协议（loss/LR/steps）。

## 6. 下一步：Run5 方向候选（需重新预注册后再动手）

按决策文档 §25，证据已支持结束自定义轻量 detail 搜索，候选：

1. **重新分配 semantic/detail 参数预算**（例如更宽的 shallow ViT + 更薄的解码路径）；
2. **使用有 ImageNet 预训练的成熟超轻 backbone 子层**作为 detail 来源；
3. **去掉独立 detail branch**，让轻量 decoder 从 ViT 浅层 token / patch feature
   重建细节（16×16 token 上采样 + difference-first 重建）。

同时保留 Run4 已确认的两个正面资产：
- R4-0 健康 full12 frozen 参考（83.14）与 R4-1 depth-4 证据（−0.37 F1、−3.56M 参数）；
- R4-D0 的 feature 级诊断方法（无训练、可复用于 Run5 候选筛选）。

## 7. 留存资产（代码已进仓库）

- `models/model/psd_detail.py`（PSD-Detail 78,464 参数，smoke T-PSD0..T-PSD5 全过）；
- `analyse/run4_detail_interface_audit.py`（R4-D0，可复用于任何 detail 候选）；
- `train_scripts/UltraLight/Run4/{audit_R4_D0,train_R4_2d_PSD_DETAIL,dryrun_R4_PSD}_SYSU.sh`；
- `train_R4_2c_ADAPTER_ALIGN_SYSU.sh` 与 `--detail_mode light_bnrelu` 代码路径
  （预注册选项，audit 判定后未触发，留存备查）。
- 日志：`outputs/UltraLight/Run4/R4_2d_PSD_DETAIL/SYSU-CD-256/train_log.txt`；
  audit 报告：`outputs/UltraLight/Run4/R4_D0_DETAIL_AUDIT/audit_report.txt`。
