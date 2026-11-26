# S13-final：support 先验的梯度路径与隔离

## 1. 真实 checkpoint 的 support_head 状态

直接读 `best_model.pth` 的 `context.support_head.{weight,bias}`：

| checkpoint | weight 范数 | max\|w\| | bias 范数 | max\|b\| |
|---|---|---|---|---|
| R_FULL-2334 / SYSU | 1.1551 | 0.2607 | 0.0526 | 0.0483 |
| R_FULL-2334 / LEVIR | 1.0702 | 0.2943 | 0.1818 | 0.1788 |
| R_LEAN-2334 / SYSU（confidence） | **0.0** | **0.0** | **0.0** | **0.0** |
| R_COMPACT-2334 / SYSU（confidence） | **0.0** | **0.0** | **0.0** | **0.0** |
| R_LEAN / R_COMPACT × LEVIR | 该轮未训练（checkpoint 不存在） | — | — | — |

⇒ confidence 分支的 support_head 在真实训练中**一次都没有离开零初始化**，这与"零点与 clamp 双重
断梯度"的机制预测一致；mass 分支（R_FULL）的同一层学到了非零权重（范数 ≈1.1）。

## 2. 梯度路径单元测试（`s13_support_final.py`，同一输入/损失/种子）

损失构造为 `log_softmax(logits + logprior, -1)` 对前一半 key 的负对数似然，backward 后读
`support_head.weight.grad`：

| 臂 | log_prior 范围 | grad 范数 | 非零元素 | 一步更新后 attention L1 |
|---|---|---|---|---|
| OLD（`abs(2·share−1).clamp_min(1e-6)`，零初始化） | −13.82 / −13.82 | **0.000** | 0 | 0.000 |
| SHARE（`share.clamp_min(1e-6)`） | −0.693 | 0.202 | 16 | 2.39e-3 |
| NONZERO（confidence，kaiming 初始化） | −3.23 / −2.88 | 2.419 | 16 | 4.59e-1 |
| LOGSHARE（`logsigmoid(pooled_logits)`） | −0.474 | 0.0344 | 16 | 8.30e-5 |
| OFF（无先验） | 0 | 0 | 0 | 0 |

⇒ 断点来自 `abs` 在 0 处的次梯度为 0 且 `clamp` 在边界处梯度为 0（**双重断**），不是零初始化本身：
把 `abs` 去掉（SHARE）或改 `logsigmoid` 都能恢复梯度。`logsigmoid` 的一步影响最小，是最保守的修复。

## 3. 冻结先验反事实（在 S9a-final 里，与 BASE 同图同权重）

| 臂 | SYSU 2334 | SYSU 2335 | LEVIR 2334 | LEVIR 2335 |
|---|---|---|---|---|
| SUPPORT-OFF（`log s = 0`） | −0.44 | −0.01 | +0.02 | +0.09 |
| COUNT-OFF（去掉 `−log N`） | **+0.29\*** | **+0.29\*** | −0.02 | −0.01 |
| SCALE-OFF（去掉学习尺度偏置） | **+0.29\*** | **+0.29\*** | −0.02 | −0.01 |
| GUIDE（先验换成 guide 区域类均值） | −0.75 | −0.01 | −0.31 | −0.11 |
| DISTONLY（三项全关） | −0.35 | +0.04 | +0.01 | +0.06 |

`*` = 配对 bootstrap CI 不含 0。

## 结论

1. **confidence 先验在真实训练中是死参数**（权重恒为 0），所以 R_COMPACT/R_LEAN 的
   "confidence 支持度"从未生效；这是实现缺陷，不是"先验无用"的证据。
2. **mass 模式下 support 项在推理时几乎不影响预测**：直接置零只动 0.0–0.4 pp（噪声内）。
   原因可从 S9a 的注意力侧看出：support 是"每槽一个标量"，在 softmax 里相当于一个常数偏置，
   模型已经把这段信息吸收进 `scale_bias`/温度/距离项。
3. **`−log N_s` 与学习尺度偏置在 SYSU 上是有害的**（去掉 +0.29 pp，两 seed 同号且 CI 排除 0），
   LEVIR 上无害。合理解释：SYSU 的四个尺度 token 数量差异大（128/32/32/32），
   `−log N` 会系统性抬高小尺度槽的权重，而 SYSU 的变化更多由大尺度上下文决定。
4. **换先验（GUIDE）不能改善伪变化**：LEVIR −0.3 pp、SYSU −0.75/−0.01 pp，方向为负。
   所以"用 guide 类均值代替 support_head"不是可部署的候选。
5. 与 S9a 的 UNIFORM/PRIORONLY 合起来看：**先验项的净贡献 ≈0，距离项的贡献 1–2 pp（LEVIR）**，
   这与任务书 §7 的判据（"若 NOCTX 改善而先验替换无改善 → 不能归咎 support"）完全一致：
   本轮 NOCTX 是负贡献（去掉变差 1.6–16.7 pp），所以不能把问题归到 support 上。

## 建议（不启动训练）

- 若将来要修 confidence 分支，最小改动是把 `abs` 换掉（`share` 或 `logsigmoid`），并给
  `support_head` 非零初始化；这是独立工程任务，不构成本轮的结构候选。
- `−log N` 与学习尺度偏置在 SYSU 上有害，但只在 SYSU 上成立、幅度 0.3 pp，暂不作为改动项。
