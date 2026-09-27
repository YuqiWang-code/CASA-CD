# SAT — Selective Aggregation Transformer（CVPR 2026）核心代码提取

> 上游：https://github.com/PhuTran1005/SAT
> 论文：SAT: Selective Aggregation Transformer for Image Super-Resolution
> （arXiv:2604.07994）
> 许可：上游仓库 MIT License（本目录仅保留核心算法代码，训练框架未收录）

## 为什么保留这些

SAT 的完整仓库是一个 SR 训练工程（basicsr 框架、上采样重建、训练选项等）。
CASA-CD 只需要它的**核心机制**——「非对称、token 聚合注意力」，即：

> **完整 Query + 压缩 K/V**：保留所有查询位置（逐位置判别不丢细节），
> 只压缩提供上下文的 Key/Value（冗余背景被聚类合并）。

## 文件

| 文件 | 内容 |
|---|---|
| `saa.py` | **SAA（Selective Aggregation Attention）+ cluster_and_merge**，从上游 `basicsr/archs/sat_arch.py` 提取并去框架化（仅依赖 torch），附冒烟测试 |

### saa.py 算法要点（与上游逐行一致）

1. **cluster_and_merge(x, K)**：密度峰值聚类把 N 个 token 合并成 K 个压缩 token
   - 在 S=min(N, max(2K, 4K)) 个子采样点上算余弦相似度 → 局部密度 ρ（top-k 相似度均值）
   - δ = 1 − max(与更高密度点的相似度)；γ = ρ×δ 取前 K 为簇心
   - 全部 token 按余弦相似度分配给簇心，簇内加权平均 → 压缩 token
2. **范数保持**：压缩 token 缩放到原 token 的最大范数量级
3. **交叉注意力**：`softmax(Q_full @ K_comp^T / √d) @ V_comp`，Q 通道按 `c_ratio` 压缩（默认 0.5），K/V 的 token 数压缩到 `M*N`（默认 M=0.03）
4. 复杂度 **O(N²) → O(NK)**，K≪N

### 在 SAT 原架构中的使用方式（供参考）

SAT 的 Block 里 **偶数层用 L_SA（局部窗口注意力，双分支 H/V 平移窗口 + DRPE），
奇数层用 SAA（全局选择性聚合）**，交替堆叠；MLP 带 SimpleGate 门控（通道减半的轻量 FFN）。
L_SA / DynamicPosBias / Gate / MLP 等未收录，需要时从上游仓库取
（`basicsr/archs/sat_arch.py` 的 `L_SA`、`DynamicPosBias`、`Gate`、`MLP`）。

## 与 CASA-CD 的映射

| SAT 机制 | CASA-CD（CASAA）用法 |
|---|---|
| SAA：完整 Q | 不变——ChangeViT 的 ViT 里 Query 保持全部 patch token |
| cluster_and_merge：背景 token 聚类 | 改为**变化感知**：疑似变化 token 尽量保留、稳定背景 token 强聚合（引入双时相差分/注意力线索引导聚类） |
| 范数保持 + c_ratio 通道压缩 | 可选：保留预训练 Q/K/V 能力时按需裁剪 |

详见 `docs/temporary/CASA-CD_研究路线_ChatGPT方案记录.md`。

## 验证

```bash
cd others/SAT && python saa.py
# SAA 输出形状 OK: (2, 256, 192)，压缩 token 数 NF=7（N=256）
# 全注意力 FLOPs ≈ 50.33 M
# SAA FLOPs     ≈ 1.38 M（不含聚类开销，O(N²)→O(NK)）
```
