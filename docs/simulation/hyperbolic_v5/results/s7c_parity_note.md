# s7c parity 说明

任务书 §3 A2 要求：插桩前向 vs 原前向要逐级对齐，首级 `max|Δ| ≤ 2e-4`、二值差必须报。
本文件记录实际做法与结果，避免"parity 全 0"被误读成"所有运行都开了全量 parity"。

## 1 比较项

`tools/simulation/s7c_capture.py` 在 `--parity 1` 时对每张图做两次前向：

1. 正常前向（不插桩）；
2. 插桩前向（Recorder 替换 `manifold_pool` / `project_queries` / `attention_logits`，
   原方法在 wrapper 内部被调用）。

逐项比较：

| 键 | 内容 |
|---|---|
| `logit` | 首级（256²）输出 logit 的 `max|Δ|` |
| `logit_levels` | 其余三级（128/64/32）logit 的 `max|Δ|` |
| `aux` | 递归比较 dict/list 中的 aux 张量（实测 `n_aux=8`） |
| `binary` | 用 `logit > 0` 得到的二值图，比较不同像素数 |
| `kv` | 两次插桩运行之间 `manifold_pool` 的 points/weights/out（实测 `n_kv=16`） |
| `support` | 每尺度 support 的 `max|Δ|` |

## 2 结果

- 3 图 smoke（`--parity 1`，R_FULL-2334/SYSU）：
  `logit 0.0 / logit_levels 0.0 / aux 0.0 / binary 0.0 / kv 0.0 / support 0.0`
  （`n_aux=8`、`n_kv=16`），即**逐位相同**。
- 正式 4 套 checkpoint × 2 库的采集用 `--parity 0` 关闭双前向（每套运行时间减半，
  且代码路径与 smoke 完全一致）；因此 `summary.json` 里 `n_aux=0 / n_kv=0`，
  这不表示"没有比较过"，而是"这次没有重复比较"。

## 3 已知限制

- `binary` 用的是 `logit > 0`（等价于 `sigmoid(logit) > 0.5`），不是对 float 概率取阈值；
  两者在 0.5 处等价，但边界像素可能因浮点顺序不同而不同——实测为 0，故不构成问题。
- 没有把 `occupancy`、`mean_gate`、`level_scale` 等回写中间量纳入 parity；它们由
  同一 forward 产出，若 logit 逐位相同则这些量必然相同（同一 kernel 顺序），故未单列。
