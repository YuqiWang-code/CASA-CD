# Run9 backbone matrix: within-block verdict against CTRL

protocol `casa_minor_run9_backbone_v1`, val-selected, threshold 0.5, seed 2334, 
| arm | dataset | F1 (%) | R / P | best ep | control | delta (pp) | verdict |
|---|---|---|---|---|---|---|---|
| VM_F_CTRL | SYSU | 79.03 | 82.93 / 75.47 | 14 | — | — | baseline |
| VM_F_TAR_ONLY | SYSU | 78.65 | 79.43 / 77.89 | 22 | 79.03 | -0.375 | worse (<= -0.36) |
| VM_F_HRA_ONLY | SYSU | 74.78 | 75.14 / 74.43 | 24 | 79.03 | -4.244 | worse (<= -0.36) |
| VM_F_TAR_HRA | SYSU | 78.92 | 78.66 / 79.18 | 22 | 79.03 | -0.110 | inside noise (0.32) |
| VM_T_CTRL | SYSU | 81.00 | 84.51 / 77.76 | 9 | — | — | baseline |
| VM_T_TAR_ONLY | SYSU | 81.21 | 78.07 / 84.61 | 13 | 81.00 | +0.211 | inside noise (0.32) |
| VM_T_HRA_ONLY | SYSU | 82.31 | 80.86 / 83.82 | 9 | 81.00 | +1.314 | pass (>= 0.36) |
| VM_T_TAR_HRA | SYSU | 80.57 | 78.31 / 82.96 | 9 | 81.00 | -0.427 | directional |
| VIT_F_CTRL | SYSU | 82.52 | 80.95 / 84.16 | 17 | — | — | baseline |
| VIT_F_TAR_ONLY | SYSU | 81.91 | 80.02 / 83.88 | 7 | 82.52 | -0.619 | worse (<= -0.36) |
| VIT_F_HRA_ONLY | SYSU | 79.40 | 77.07 / 81.88 | 17 | 82.52 | -3.120 | worse (<= -0.36) |
| VIT_F_TAR_HRA | SYSU | 82.49 | 79.53 / 85.68 | 17 | 82.52 | -0.037 | inside noise (0.32) |
| VIT_T_CTRL | SYSU | 78.37 | 77.77 / 78.99 | 15 | — | — | baseline |
| VIT_T_TAR_ONLY | SYSU | 78.89 | 78.61 / 79.18 | 13 | 78.37 | +0.522 | pass (>= 0.36) |
| VIT_T_HRA_ONLY | SYSU | 77.71 | 77.55 / 77.87 | 8 | 78.37 | -0.661 | worse (<= -0.36) |
| VIT_T_TAR_HRA | SYSU | 78.72 | 77.06 / 80.45 | 14 | 78.37 | +0.350 | directional |
| VM_F_CTRL | LEVIR | 81.43 | 78.97 / 84.04 | 44 | — | — | baseline |
| VM_F_TAR_ONLY | LEVIR | 81.63 | 80.13 / 83.19 | 42 | 81.43 | +0.197 | inside noise (0.32) |
| VM_F_HRA_ONLY | LEVIR | 63.18 | 64.72 / 61.70 | 39 | 81.43 | -18.253 | worse (<= -0.36) |
| VM_F_TAR_HRA | LEVIR | 81.66 | 79.34 / 84.12 | 42 | 81.43 | +0.231 | inside noise (0.32) |
| VM_T_CTRL | LEVIR | 90.70 | 89.34 / 92.09 | 43 | — | — | baseline |
| VM_T_TAR_ONLY | LEVIR | 89.99 | 88.71 / 91.30 | 37 | 90.70 | -0.711 | worse (<= -0.36) |
| VM_T_HRA_ONLY | LEVIR | 90.43 | 89.60 / 91.28 | 38 | 90.70 | -0.267 | inside noise (0.32) |
| VM_T_TAR_HRA | LEVIR | 90.67 | 90.05 / 91.29 | 42 | 90.70 | -0.030 | inside noise (0.32) |
| VIT_F_CTRL | LEVIR | 85.44 | 84.71 / 86.18 | 41 | — | — | baseline |
| VIT_F_TAR_ONLY | LEVIR | 85.97 | 85.37 / 86.58 | 42 | 85.44 | +0.531 | pass (>= 0.36) |
| VIT_F_HRA_ONLY | LEVIR | 75.06 | 75.49 / 74.64 | 41 | 85.44 | -10.381 | worse (<= -0.36) |
| VIT_F_TAR_HRA | LEVIR | 86.21 | 85.67 / 86.74 | 42 | 85.44 | +0.768 | pass (>= 0.47) |
| VIT_T_CTRL | LEVIR | 86.04 | 85.12 / 86.98 | 44 | — | — | baseline |
| VIT_T_TAR_ONLY | LEVIR | 86.64 | 85.38 / 87.93 | 44 | 86.04 | +0.600 | pass (>= 0.36) |
| VIT_T_HRA_ONLY | LEVIR | 86.21 | 85.26 / 87.18 | 44 | 86.04 | +0.173 | inside noise (0.32) |
| VIT_T_TAR_HRA | LEVIR | 86.83 | 86.03 / 87.63 | 44 | 86.04 | +0.788 | pass (>= 0.47) |

## Pairing summary

- LEVIR: mean -2.196 pp over 12 pair(s)
- SYSU: mean -0.600 pp over 12 pair(s)

### SYSU: F1 (%) and delta against the block CTRL

| block | CTRL | TAR_ONLY | HRA_ONLY | TAR_HRA |
|---|---|---|---|---|
| VM/F | 79.03 | 78.65 (-0.38) | 74.78 (-4.24) | 78.92 (-0.11) |
| VM/T | 81.00 | 81.21 (+0.21) | 82.31 (+1.31) | 80.57 (-0.43) |
| VIT/F | 82.52 | 81.91 (-0.62) | 79.40 (-3.12) | 82.49 (-0.04) |
| VIT/T | 78.37 | 78.89 (+0.52) | 77.71 (-0.66) | 78.72 (+0.35) |

### LEVIR: F1 (%) and delta against the block CTRL

| block | CTRL | TAR_ONLY | HRA_ONLY | TAR_HRA |
|---|---|---|---|---|
| VM/F | 81.43 | 81.63 (+0.20) | 63.18 (-18.25) | 81.66 (+0.23) |
| VM/T | 90.70 | 89.99 (-0.71) | 90.43 (-0.27) | 90.67 (-0.03) |
| VIT/F | 85.44 | 85.97 (+0.53) | 75.06 (-10.38) | 86.21 (+0.77) |
| VIT/T | 86.04 | 86.64 (+0.60) | 86.21 (+0.17) | 86.83 (+0.79) |

A delta below the seed noise (0.32 pp on both libraries) is not an effect.
Frozen blocks train only the bridge, the decoder and the head (0.21-0.58 M),
so a frozen arm differs from its trainable twin by the encoder update alone.
