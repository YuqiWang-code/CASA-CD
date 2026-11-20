# CASA-MinorGoal / Run6 快照

协议 `casa_minor_run6_edge_tap_v1`；val 选模、test τ=0.5、seed 2334。

| arm | dataset | F1 | Recall | Precision | best ep | val F1 | train/deploy (M) | 秒 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| EDGE_ALL | SYSU-CD-2 | 82.258 | 77.99 | 87.02 | 33 | 81.56 | 2.111417 / 2.061668 | 10294 |
| EDGE_FINE | SYSU-CD-2 | 82.351 | 78.48 | 86.62 | 25 | 82.20 | 2.111417 / 2.061668 | 10289 |
| EUCLID_ALL | SYSU-CD-2 | 82.504 | 80.55 | 84.56 | 7 | 81.39 | 2.111417 / 2.061668 | 9635 |
| EDGE_TAP_CUT | SYSU-CD-2 | 83.090 | 79.66 | 86.83 | 33 | 81.13 | 2.111417 / 2.061668 | 10348 |
| EDGE_FULL | SYSU-CD-2 | 82.006 | 78.97 | 85.28 | 19 | 81.69 | 2.111417 / 2.061668 | 10349 |
| EDGE_LEVIR | LEVIR-CD- | 91.009 | 89.83 | 92.23 | 162 | 91.59 | 2.111417 / 2.061668 | 8552 |

## 读法

- `EDGE_ALL` 与 `EDGE_TAP_CUT` 逐位同配置，复跑差 0.832 pp ⇒ 第一批的 +1.085 pp 只比复跑带宽高 0.25 pp，方向可信、幅度不可信。
- `EDGE_FINE`（只监督 D1,D2）+0.093 pp、`EUCLID_ALL`（欧氏均值）+0.246 pp 都在噪声内：尺度与几何都不是杠杆。
- 重心的稳定痕迹是操作点换位（P 87.02/R 77.99 vs P 84.56/R 80.55），不是可分性。
- LEVIR 三条同 seed 配置 90.990 / 91.008 / 91.009，差 0.02 pp：作用域只在 SYSU 边界带。

代码快照与指标行见同目录 `models_and_metrics_CASA-MinorGoal_Run6.txt`；源文件与逐图 TP/FP/FN 在服务器 `outputs/CASA-CD/CASA-MinorGoal/Run6/`。
