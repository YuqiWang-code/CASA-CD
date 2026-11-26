# hyperbolic_v6 — V5 收尾轮的仿真结果

本目录只放**文档与小型表格**（`.csv`/`.md`）；逐图 CSV、checkpoint 指标、张量都在服务器
`/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/{S9aFinal,S8bFinal,S7c}`。
所有阈值判定、温度拟合、臂选择只用 **val**；test 不参与任何选择。

## 文件 ↔ 任务书条款

| 文件 | 对应任务书条款 | 状态 |
|---|---|---|
| `S9A_FINAL_REPORT.md` | §3 A1/A2/A3 + §4 S9a-final（16 臂） | 完成 |
| `S9B_BUDGET_REPORT.md` | §5 S9b（224/128/56/32 tokens 预算曲线） | 完成（全量展开参考未做，见文末） |
| `S8B_FINAL_REPORT.md` | §6 S8b-final（同预算自适应划分，B=32） | 完成（表示保真层） |
| `S13_FINAL_REPORT.md` | §7 S13a-0b/1（support 梯度与先验隔离） | 完成 |
| `CORRECTIONS_FINAL.md` | §1/§3 A5（旧表述逐条改口） | 完成 |
| `FINAL_DECISION.md` | §9/§10 九问与结构决策 | 完成 |
| `results/table_metrics.csv` | 16 臂 × 4 checkpoint 的全局混淆矩阵指标 | 由 `finalize_summary.py` 生成 |
| `results/table_paired_delta.csv` | 每条臂相对 BASE 的配对 ΔF1 + bootstrap CI | 同上 |
| `results/table_budget_curve.csv` | 预算曲线 | 同上 |
| `results/table_s8b_summary.csv` + `s8b_final_*.csv` | 自适应划分的保真度 | 由 `s8b_final.py` 生成 |

## 复现命令（服务器，conda env `casacd`，CPU）

```bash
cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation

# S9a-final: 16 arms, 4 checkpoints
for arm in R_FULL-2334 R_FULL-2335; do for ds in SYSU-CD-256 LEVIR-CD-256; do
  python -u s9a_final.py --arm $arm --dataset $ds --mode fit  --out ./S9aFinal
  python -u s9a_final.py --arm $arm --dataset $ds --mode arms --images 48 --out ./S9aFinal
done; done

# S9b: budget curve
for gs in b128 b64 b32; do python -u s9a_final.py --arm R_FULL-2334 \
  --dataset SYSU-CD-256 --mode grids --gridset $gs --images 48 --out ./S9aFinal; done

# S8b-final: adaptive partition at equal budget
python -u s8b_final.py --arm R_FULL-2334 --dataset SYSU-CD-256 --images 12 --out ./S8bFinal

# tables
python finalize_summary.py --dir S9aFinal --iters 2000
```

## 三个必须记住的口径

1. **F1 只用整数混淆矩阵算**：每图存 TP/FP/FN/TN，先求和再算 F1/IoU/Kappa；逐图 F1 平均
   （V5 的做法）会把空图记 0，方向都可能反过来。
2. **Q/K/V 的半径与量纲不同**：K 的成员是 `qk(LN(D))/sqrt(d)`，V 的成员是 `v(LN(D))` 不除
   `sqrt(d)`；两侧的绝对 SSE 不可跨侧比较，只比同侧。
3. **本轮的 F1 是冻结权重反事实**：换几何/换预算都用同一套权重，没有重训。改善只是"值得试"
   的证据，不能当作训练后的精度增益；同理变差也不能证明重训后仍然差。

## 样本分母（不得省略）

- S9a-final：LEVIR 每个 checkpoint 48 张 val，SYSU 每个 checkpoint **36** 张（分层配额里
  bg/small/mid/large 某层不足 12 张）；2 个 seed（2334/2335），共 4 个 checkpoint。
- S9b：同上清单，SYSU 36、LEVIR 48 张。
- S8b-final：每 checkpoint 12 张 val（4 个 checkpoint × 4 个尺度 × 8 个臂）。

## 未做项（明确记录）

- S9b 的"组质量匹配全量展开参考"（F-MATCH/F-RAW 与 dense-vs-streaming）未实现：预算曲线已经
  显示 224→32 任务层无差异，展开参考不能再改变这个判断，故留作后续。
- S8b 的下游反事实（S8b-B/S9b 用自适应分区重跑 readout）未做：S9b 显示固定 32 与 224 无差异，
  下游没有可回收的损失。
- S10/S11/S12 按任务书触发条件未启动（触发条件均未满足）。
