# CASA-MinorGoal / Run9 快照

换骨干矩阵的收尾材料。Run9 只替换编码器（MobileNetV2 → VMamba-Tiny 29.94 M / ViT-B/16 前 4 块 ≈30.9 M），
TAR·SPP、跨尺度检索、DCR 的拓扑与 Run8 相同，因此没有另画总体结构图；`docs/CASA_总体模型结构.pptx`
的编码器一栏仍写 MobileNetV2，若日后需要对外展示，改那一格即可。

- `models_and_metrics.txt` — 生成时的 `models/` 代码快照（46 个文件）+ 正式指标（428 行，其中本 tag/Run9 32 行）。
  训练实跑用的是 `runtime/` 冻结副本，源码身份以 `source_code.tar.gz` 为准。
- `comparison.md` / `comparison.json` — 按 (backbone, freeze) 分四块的 F1 与块内 Δ，含 12 对均值。

结论摘要：冻结骨干时检索模块全面失效（SYSU −3.12/−4.24、LEVIR −10.38/−18.25）；冻结本身比换骨干更贵
（同 CTRL：SYSU 79.03→81.00、LEVIR 81.43→90.70）；ViT-B 全参微调在 SYSU 掉 4.15 pp、LEVIR 反而 +0.60；
两库最优骨干不同（SYSU=ViT-B 冻结 82.52，LEVIR=VMamba 解冻 90.70）。该路线已被放弃，不再重训。
