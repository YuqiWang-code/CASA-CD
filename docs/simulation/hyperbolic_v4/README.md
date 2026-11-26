# hyperbolic_v4 — 本轮（A1/A2 已落地，S7b/S8b/S9 待做）

对应任务书 `docs/temporary/CASA-CD_V3仿真结论与下一轮实验执行方案.md`。
V1–V3 全部保留不覆盖；本目录只放本轮被要求重做的部分。

| 文件 | 修的是任务书哪一条 | 状态 |
|---|---|---|
| `c2c_retrieval_v4.m` | §3 A1（平方距离 + 真实温度 + 共享场景/扰动 + 真实半径档）与 A2（每 p 重生成特征标签、四档先验 × 两几何） | ✅ 已跑通，日志 `results/log_c2c.txt` |
| `CORRECTIONS_V4.md` | §3 A5（纠错表与状态） | ✅ 已写 |
| `../tools/simulation/s13_support.py` | §7.2 S13a-0（confidence 梯度阻断复现） | ✅ 已跑通，日志 `results/log_s13a.txt` |
| A3（c3b 3 seed × 3 σ 交叉 + 模型匹配层） | §3 A3 | ⬜ 未做 |
| A4（r1b/r2b 口径核对） | §3 A4 | ⬜ 未做 |
| S7b / S8b / S9a / S9b / S13a-1 | §4–§7 | ⬜ 未做 |

## 复现

```bat
rem 本地 MATLAB 会话（matlab_run）本轮失联，改用 -batch；脚本自己写 UTF-8 日志
"D:\Program Files\MATLAB\R2022a\bin\matlab.exe" -batch "run('F:/Code_Repositories_2/CursorCode/CASA-CD/docs/simulation/hyperbolic_v4/c2c_retrieval_v4.m')"

python tools/simulation/s13_support.py
```

c2c 约 55 s；s13a 约 3 s。

## 两条口径提醒（否则会读错数）

1. **JS 距离不是百分比**：`sqrt(JS)` 取值 0–1，0.056 = 5.6% 的分布距离，不能说成"注意力变了 5.6%"。
2. **"温度+行偏置残差"才是几何差异的实质**：它把 logit 差对 `Le` 做单标量回归后取残差 /
   该行 logit 跨度。残差小（真实半径档 0.9%–2.0%）说明差异的主体是一个距离尺度因子，
   于是"几何是否有用"落到"τ 是否按几何标定"上——这正是 S9a 拟合温度臂要回答的。
