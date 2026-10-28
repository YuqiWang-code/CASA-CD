# CASA-Pyramid / Run6 训练记录

协议`casa_pyramid_40k_taskgeometry_v6`。仅物理GPU0，四库各双曲与CTRL共8项，优先LEVIR→SYSU→WHU→CDD。正式test结果尚未完成；此目录先保存启动健康探针，探针使用train/val，不作为正式指标。

| 双曲探针 | epoch / step | 主训练损失首→末 | 验证F1首→末(%) |
|---|---:|---:|---:|
| LEVIR | 6 / 1332 | 1.9582→0.9867 | 82.84→88.02 |
| SYSU | 4 / 1500 | 2.4060→1.4956 | 76.29→79.75 |

所有轮次任务梯度有限，各层分支梯度非零。两个外观子原型未出现整体重合；LEVIR最粗层采用门控接近1，主任务梯度仍非零，后续需结合纠错与泛化评价。

[运行说明](../../../../train_scripts/CASA-Pyramid/Run6/README.md) · [模块设计](../../../../models/HYPERBOLIC_DESIGN.md)
