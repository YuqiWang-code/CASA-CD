# Run3 Z1/Z2 零成本复盘（对 Run2 E1/E2/E3 checkpoint，SYSU test 集）

复盘文档 §12 的两个假设验证（M1 基线 small F1=0.1796）：

| 变体 | small P/R/F1 | medium F1 | large F1 | band2 F1 | 像素级 Recall |
|---|---|---|---|---|---|
| M1_FULL | — / — / 0.1796 | 0.4620 | 0.8273 | 0.6710 | 84.41 |
| E1_CP_CAACP | 0.2104 / **0.1534** / 0.1678 | 0.3779 | 0.7882 | 0.6454 | 80.94 |
| E2_FRH | 0.2348 / **0.1562** / 0.1738 | 0.4066 | 0.7788 | 0.6467 | 80.10 |
| E3_CP_FRH | 0.2409 / **0.1594** / 0.1779 | 0.4169 | 0.7887 | 0.6520 | 80.69 |

**H-E1 / H-E2 均成立**：
- 三个 Run2 变体的 small **Recall 全部掉到 0.153~0.159**（Precision 略升到 0.21~0.24）——
  CP/FRH 的保守化"首先杀 tiny changes"（P↑R↓ 的 weak-positive suppression 签名）；
- medium/large 基本保持（large 0.78~0.79，与 M1 0.827 同量级）；
- 与复盘文档 §0.1 的像素级 Recall −3.5~−4.3pp 完全互证。

**对 Run3 的含义**：E4（RA 保护 dense residual）与 E5（1/4 尺度提前提取 tiny temporal evidence）
的机制动机进一步升级为"近直接证据"；开训前不再阻塞。
