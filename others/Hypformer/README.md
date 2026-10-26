# Hypformer 核心参考

- 论文：*Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space*，ACM SIGKDD 2024，DOI [10.1145/3637528.3672039](https://doi.org/10.1145/3637528.3672039)。本地PDF为作者2025-08修订的arXiv版本，会议年份仍为2024。
- 官方仓库：[Graph-and-Geometric-Learning/hyperbolic-transformer](https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer)。
- 提取版本：`768fcec099bcb039d0d838e3e4ebe83ef6bb7426`；下载日期：2026-10-26。
- 许可证：MIT，完整声明见[LICENSE](LICENSE)。

## 保留内容

| 文件 | 内容与来源 |
|---|---|
| `manifolds/layer.py` | 官方`Hypformer/manifolds/layer.py`中的`HypLinear`、`HypLayerNorm`；保留空间坐标变换后重建时间坐标的公式 |
| `manifolds/lorentz.py` | 官方`Hypformer/manifolds/lorentz.py`中的交叉Lorentz内积与加权质心；基础流形操作由Geoopt继承 |

仅保留上述核心类及所需导入。质心中的`inner`调用使用Geoopt原生的`inner(x,u,v)`接口；公式不变。依赖PyTorch、Geoopt；不包含训练、数据、完整Transformer、分类器或权重，不由CASA运行代码导入。

## 对CASA的直接参考

论文Eq.9与Eq.13在固定曲率−1时使用`(sqrt(1 + ||z||²), z)`满足Lorentz流形约束。它允许从学习到的空间坐标重建时间坐标，不必反复Exp/Log。此时距原点的测地半径为`asinh(||z||)`，随输入幅度连续变化。

Hypformer的完整输入链仍包含Exp映射。CASA将欧氏投影直接作为空间坐标，是上述合法坐标参数化的简化应用，不是整套Hypformer的复现。官方注意力示例的负Lorentz弦距也不能写成严格测地距离平方。核心公式成立不意味着在任意FP32范数下都无数值误差，实际半径及梯度仍由运行诊断判断。
