# CASA-CD：双曲流形 / 流形空间用于变化感知与 CAACP-SS2D 的系统调研

> **研究调研与可执行设计报告**｜2026-10-13  
> **研究范围**：CDD-CD-256、LEVIR-CD-256、SYSU-CD-256、WHU-CD-256，**全监督遥感二值变化检测**。  
> **仓库**：[YuqiWang-code/CASA-CD](https://github.com/YuqiWang-code/CASA-CD)；检索及代码审查所见 `main` 提交：[`23eb250cd48bb012ffaafc908dd1daafa560aec2`](https://github.com/YuqiWang-code/CASA-CD/tree/23eb250cd48bb012ffaafc908dd1daafa560aec2)，提交日期 2026-10-12。  
> **材料**：仓库 README、核心 `models/model/` 源码、Run7 结果节、Diag1 审计报告；用户提供的《SAT: Selective Aggregation Transformer for Image Super-Resolution》（CVPR 2026 Findings 相关论文 PDF）和 RSML-3 服务器环境说明；2024–2026 年论文/出版社页面及官方 GitHub。  
> **证据标记**：**[F]** 代码/文件可直接核对的事实；**[R]** 论文与出版社可核对的事实；**[I]** 从证据得出的推断；**[H]** 待验证假设；**[M]** 缺失证据。  
> **审计边界**：本报告通过 GitHub 连接器读取了仓库目录树（620 个节点）、README、CAACP/SS2D/TinyViM/STR 关键源码、Diag1、Run7 等材料；当前执行环境不能直接联网 `git clone`。因此不声称逐字审阅了仓库所有二进制/历史产物，也**未直接访问 RSML-3 上的最新完整训练日志与模型权重**。README 中历史指标是“仓库报告数值”，不是本次独立复算的正式实验结果。

---

## 0. 执行摘要：先讲结论

**结论 A：双曲空间值得研究，但“将 `1−cos` 换成双曲距离”不构成充分创新，甚至可能数学上严格没有新增排序信息。** 如果单位特征以**相同固定半径**映射到双曲面，则双曲测地距离与原余弦差异严格单调对应；CASA-CD 的 `rank_normalize_2d` 随即产生相同排序，原 `rank`-CAACP 的权重因此不变。换言之，不先解决这个退化问题，投入一次 80K 训练很可能只是在给已知余弦分数换数学外衣。

**结论 B：已有直接先例，不能主张“首次把双曲空间用于遥感变化检测”。** 最关键的是 **HES-Net（IEEE JSTARS 2024）**：双时相特征 → 双曲映射 → 双曲相似性注意力；其次 **HyperCLIP-CD（IEEE TGRS 2026）**：双曲图文对齐用于遥感变化检测；再有 **HSN（CVPR 2025）**：无人机树木长期细粒度变化监测。它们与 CASA-CD 的真正差异，在于**双曲几何用于什么操作**：已有工作用于特征相似性、注意力、跨模态对齐/表征；CASA-CD 可以探索**低频 SS2D 扫描前、规则 2×2 cell 内的双时相几何证据分配**，并保留预训练与部署上界。但这个差异是潜在的创新切入点，不等于有效性已经被证明。

**结论 C：最值得做的是“局部上下文调制半径的双曲变化证据”，而不是将整个 TinyViM 改成双曲网络。** 推荐的首选候选为 **CRH-CAACP（Context-Radial Hyperbolic Change-Aware Context Pooling）**：计算每时相像素相对其 2×2 邻域的方向离散度，将**局部结构差异编码为双曲径向坐标**，再用双时相双曲测地距离构建**共享、对称、参数自由**的 cell 权重；其余 SS2D 四向扫描、高频 dense 分支、STR 折叠图全部不动。它有**可证伪的几何增量**，而非简单把 cosine 改叫 hyperbolic。

**结论 D：但必须先判断 CAACP 的门控是否实际上“有话语权”。** 仓库 Diag1 曾报告，既有 M1 的 `||βΔC||/||c_avg||≈0.19%`，临时关闭 β 后 F1 变化约 `−2.2e−6`，小目标 Hit@25 不变；这支持**当时那个 checkpoint 对 CAACP 修正的局部依赖很弱**，并不支持“只换一个更高级分数就必然有效”。如果 β 仍接近不起作用，改评分机制很难成为决定性的主贡献。

**结论 E：建议从无训练的“几何是否带来额外信息”诊断开始，但按照当前 Run7 项目纪律，不将这种代理诊断当作可以任意替代完整训练的判决。** 先完成恒等性、非退化性、几何特异性诊断，再实施同期三组必要对照，并在四库 80K 的统一协议下看 F1/IoU/Recall、对象级召回、Params/FLOPs 和 train↔deploy 等价性。只有在**双曲版优于具有相同径向信息的欧氏版**且跨库有可复核收益时，论文才可以谈“曲率”的必要性。

---

## 1. 仓库与 CAACP 的代码级审计：你的变化感知目前究竟做了什么

### 1.1 已查看的核心源码及证据索引

| 文件 / 材料 | 本次直接确认的事实 | 对本次研究的意义 |
|---|---|---|
| [`models/model/casa_tvim_str_net.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/casa_tvim_str_net.py) | 两时相沿 batch 合并共享 TinyViM；在 Stage3 最终 TViM 前拆出 `xa,xb`；`torch.no_grad()` 中计算 `1−cos` 和逐图 rank；权重沿 batch 再复制供 A/B 共用 | **新分数必须接在这里**；只能改变打分，不要偷偷引入时间方向偏置 |
| [`models/model/layers/caacp_ss2d.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/layers/caacp_ss2d.py) | `rank`：`w_i∝ε+rank_i`；`cp`：`w_i∝1+s_i rank_i`；`c=c_avg+β(c_ca−c_avg)`；`β=0` 初始化 | 这是唯一推荐修改的模块接口和核心逻辑 |
| [`models/model/layers/ss2d.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/layers/ss2d.py) | Stage3 `index=2` 的官方低频 `AvgPool2d(2)`，后接 `cross_selective_scan`；高频经 `local_conv`；残差含 `x0−Upsample(c)` | 不改变 lattice / scan / 高频路径；否则不再是同一个实验变量 |
| [`models/model/tinyvim_s_slim.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/tinyvim_s_slim.py) | Stage3 末块 `TViMBlock.op` 被替换为 `CAACPSS2D`；继承原 SS2D 参数，仅新建零初始化 β | 保持 ImageNet EMA 预训练键加载与 epoch-0 恒等 |
| [`models/model/str_tar.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/str_tar.py)、[`str_dcr.py`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/models/model/str_dcr.py) | TAR/DCR 的可折叠卷积与双时相解码路径独立于 CAACP score | 新方法不能让 TAR/DCR 的 deploy 折叠失效 |
| [`README.md#Run7`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/README.md) | Run7 HFAE 三臂×四库、严格 80K，仓库记录 E9 `aux_only` 为判定胜者；性能收益主要在 SYSU/WHU | 当前实际项目状态已超过早期 M1/Run3；不能按旧 Run1 结果当作最新性能 |
| [`Diag1/DIAGNOSIS_REPORT.md`](https://github.com/YuqiWang-code/CASA-CD/blob/23eb250cd48bb012ffaafc908dd1daafa560aec2/docs/temporary/CASA-TViM_Diag1/DIAGNOSIS_REPORT.md) | 记录 CAACP β 局部反事实、cell 熵、small/large 召回及深度可读性坍塌 | 先有证据再谈新机制；谨防已有 bottleneck 不是评分 |
| 用户附件《SAT》(2026) | SAT 的 DTA 采用**密度引导中心选择 + 相似度加权聚合**，保留 full Query、压缩 Key/Value；不是双曲方法 | CAACP 的结构化压缩故事可参考，但**不能把 SAT 的非规则聚类机制照搬进 SS2D** |

### 1.2 实际数据流（不是概念图）

```text
T1 [B,3,256,256] ─┐
                   ├─ concat batch ─ TinyViM-S-Slim shared stem / stages
T2 [B,3,256,256] ─┘
                                      │
                      Stage3 prefix features X_A, X_B
                         each [B,168,16,16]
                                      │
                         torch.no_grad()
                                      │
                         s = 1 − cos(X_A,X_B)
                         r = rank(s) per image
                                      │
                       repeat for A/B [2B,16,16]
                                      │
                        CAACPSS2D stage3-final
                             split low/high
                             │             │
                      low-frequency       high-frequency
                             │             └─ RepDW dense local path unchanged
                  AvgPool2x2 and weighted 2x2
                             │
                   c = c_avg + β(c_ca-c_avg)
                             │
                  8×8 structured low-frequency tokens
                             │
                  four-direction selective scan
                             │
              upsample 16×16 + original low-frequency residual
                             │
               concat high-frequency output → Stage4
                             │
                         multiscale TAR
                             │
                            DCR
                             │
                   256×256 binary probability map
```

**关键约束**：CAACP 的 score 不选择 TopK、不改 8×8 的二维格点顺序、不做方向专属权重；这使 `CrossScan` 的输入仍是标准规则网格。对单独 Stage3 低频分支而言，score 的计算输入是 Stage3 前缀 **16×16 的全通道特征**，而真正被加权的是该 TViM 内部 `split` 后的低频通道，这两个张量并非完全同一个表征层。新方法必须在同一输入位置做对照，以免引入混杂变量。

### 1.3 已有评分的两种版本

令同一像素的两个时间特征为 `a_i,b_i∈R^C`。

**原始余弦变化证据**：

\[
 s_i^{\cos}=1-\frac{a_i^{\top}b_i}{\|a_i\|\|b_i\|},\qquad
 r_i=\operatorname{Rank}_{\text{image}}(s_i^{\cos})\in[0,1].
\]

- `rank` 权重：\(w_{k,i}=(r_i+\varepsilon)/(\sum_{j\in\mathcal C_k}(r_j+\varepsilon))\)；
- `cp` 权重：\(w_{k,i}=(1+s_i^{\cos}r_i)/(\sum_{j\in\mathcal C_k}(1+s_j^{\cos}r_j))\)；
- 两者都满足 \(\sum_{i\in\mathcal C_k} w_{k,i}=1\)，cell `2×2` 共四个位置；
- 对每个时相 \(t\)，\(c^t_k=c^t_{\text{avg},k}+\beta(\sum_iw_{k,i}x^t_i-c^t_{\text{avg},k})\)，**A/B 共享同一组 \(w_{k,i}\)**。

这里的 rank 是**整张 16×16 特征图上的排序**，不是逐个 2×2 cell 内排序；只有加权归一化在 cell 内完成。数学分析和复现实验不能混淆。

### 1.4 诊断证据与“为什么不能直接开训”

仓库 Diag1 报告的一个 SYSU M1 checkpoint 中：

| 量 | 仓库诊断记录 | 合理解释 | 不得做的外推 |
|---|---:|---|---|
| `β` | `0.005570` | β 从零被优化更新 | “变化评分已产生显著预测贡献” |
| `||βΔC||/||c_avg||` | `0.001858` ≈ **0.19%** | CAACP 实际改动幅度在这组权重中较小 | “所有数据集/所有 Run 都有相同幅度” |
| cell 权重熵 | `1.2651`（均匀 `ln4≈1.3863`） | rank score 形成非均匀权重 | “更不均匀就更好” |
| 仅 β 置零的 F1 差 | `≈−2.2e−6` | **该 checkpoint** 对 β 局部依赖很弱 | “删掉 CAACP 从头训练肯定无损” |
| SYSU small pooled Recall | `0.1974`；large `0.8477` | 小变化对象召回弱；是尺寸选择性瓶颈 | “瓶颈一定是 Stage3 CAACP 造成” |
| small margin 1/16→1/32 | `0.0152→0.0022` | Stage4 下采样后 small 可读性明显坍塌 | “只调 Stage3 cell 权重能自动修复 1/32 瓶颈” |

该诊断早于 Run7，是**历史机制证据**。Run7 的 E9 并非同一 checkpoint；必须重做 β、cell 权重与反事实检查，不能直接把 M1 数字套到 E9 上。

Run7 仓库 README 的同期 TEST 摘要（**本次未独立复核原始 `train_log.txt`**）：

| 数据集 | E9_HFAE_A F1 | 同期 M1_R7CTRL F1 | 差 |
|---|---:|---:|---:|
| CDD | 0.9705 | 0.9717 | −0.0012 |
| LEVIR | 0.9134 | 0.9115 | +0.0019 |
| SYSU | 0.8331 | 0.8268 | +0.0063 |
| WHU | 0.9507 | 0.9484 | +0.0023 |

仓库记录 E9 `aux_only` 部署参数为 **4,880,190**，FLOPs 为 **2.6747G**；但其 FLOPs 统计提示 `unsupported_ops=19`，**这不是扫描核全部实际运算已被完整计入的严格上界**。E9 的边缘头属于训练期模块，部署时删除；本次提案不把这个辅助损失包装为新增创新。所有正式实验指标必须在对应 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块重新核对。

---

## 2. 基础概念：流形、球面、双曲空间、测地距离怎样用于遥感变化

### 2.1 什么是“流形”？

**通俗解释**：看起来在高维欧氏空间中的遥感特征，可能只沿较低维的“弯曲表面”变化。例如同一栋建筑在不同光照、成像角度、季节下的特征变化，可能近似落在一条/一片连续轨迹上。“沿表面走多远”与“直接穿空间的直线距离”可能不同。**这只是有待验证的建模假设，不等于卫星成像物理空间天然就是双曲空间。**

区分四类常被混叫为“流形”的东西：

1. **欧氏 \(\mathbb R^d\)**：零曲率；最便宜，L2 差异。
2. **球面 \(S^{d-1}\)**：正曲率；L2 归一化后特征位于单位球面；球面测地距离 \(\arccos(\cos\theta)\)。注意：只要是两点距离，这仍与 `1−cos` 单调等价。
3. **双曲空间 \(\mathbb H^d\)**：常负曲率；适合表示**经过实验证实的层次/树状关系**，常见 Poincaré 球与 Lorentz 双曲面。其空间体积随半径增长快，可编码细化的层次。
4. **Grassmann / SPD（对称正定）等黎曼流形**：分别描述特征子空间、局部协方差等结构；它们与双曲几何不是一回事，也不等于简单的 GNN/超图。

遥感中的“几何变化”还可能指**地理配准、相机视角变化、建筑轮廓几何**；这些概念也不能自动替代黎曼流形中的曲率。

### 2.2 Poincaré 球的基本公式

设曲率为 \(-\kappa\)（\(\kappa>0\)），Poincaré 球为
\[
\mathbb B^d_\kappa=\{z\in\mathbb R^d:\kappa\|z\|^2<1\}.
\]

从原点的指数映射（切向输入 \(v\)）可写成
\[
\operatorname{Exp}^{\kappa}_0(v)
=\frac{\tanh(\sqrt\kappa\|v\|)}{\sqrt\kappa\|v\|}v,
\]

两点测地距离可用稳定的闭式计算：
\[
 d_\kappa(p,q)=\frac{2}{\sqrt\kappa}\operatorname{asinh}\left(
 \frac{\sqrt\kappa\|p-q\|}{\sqrt{(1-\kappa\|p\|^2)(1-\kappa\|q\|^2)}}\right).
\]

在同曲率的 Lorentz 模型中，用空间方向 \(u\)、半径 \(r\) 表示点 \(h=(\cosh(r),\sinh(r)u)\)（此处取 \(\kappa=1\)）则
\[
\cosh d_{\mathbb H}(h_A,h_B)=
\cosh r_A\cosh r_B-\sinh r_A\sinh r_B\langle u_A,u_B\rangle.
\]

最后一式无需显式构造 \(C+1\) 通道双曲张量，非常适合 CASA-CD 的 16×16 稠密变化评分：只需每像素的两个半径与方向点积。数值实现应在 **FP32、`torch.no_grad()`** 下进行，`acosh` 的自变量最小裁为 `1.0`，必要时控制半径上界；不能在 AMP FP16 下直接做接近单位球边界的 `atanh`。

### 2.3 一个必须先证明的反例：固定半径双曲差异 ≡ 余弦差异排序

当每个点被单位归一化、两时相使用同一个固定半径 \(r_A=r_B=r\)，上式化简为
\[
 \cosh d_\mathbb H
 =\cosh^2r-\sinh^2r\cos\theta
 =1+\sinh^2r(1-\cos\theta).
\]

对任意固定的 \(r>0\)，\(\operatorname{acosh}\) 严格递增，因此
\[
 d_\mathbb H=\operatorname{acosh}\big(1+\sinh^2r\,s^{\cos}\big),
 \quad s^{\cos}=1-\cos\theta.
\]

**数学结论**：双曲距离与原余弦变化分数具有**相同像素排序**。在不存在浮点并列误差时，
\[
\operatorname{Rank}(d_\mathbb H)=\operatorname{Rank}(s^{\cos}).
\]

因此对当前 `score_mode=rank`：`w_hyperbolic == w_cosine`，从而 `c_ca` 与最终预测均没有新增信息；对 `cp` 而言，单调数值变换可能改变权重幅度，但那首先是**置信度刻度变换**，不是已证明的曲率优势。改变曲率常数或固定半径之后，若变化排序仍完全相同，也不能声称“显著强化了变化感知”。

**这条反例是本报告最重要的否证基准。** 所有候选必须与“等秩的固定半径双曲打分”比较，并报告 *rank-disagreement / Kendall tau*。

### 2.4 “二值变化”不意味着数据必然只有两层层次

遥感图像里的纹理→局部构件→建筑/道路→地物区域，具有可能的**尺度/部件层次**；而模型监督标签仅为“变/不变”。可以尝试借助无附加标签的局部上下文形成隐式结构证据，但**不允许把没有标注的层次结构描述成真实的地物类别树**。一个有效的科学问题是：

> **背景一致性与局部结构改变能否改变双时相的最优几何度量，从而更好地决定“一个 2×2 cell 该保留哪个像素的低频上下文”？**

若数据的有效结构近似平坦、双曲半径只是噪声、或者已有 cosine 已足够区分，则欧氏/球面方法可能更好。

---

## 3. 2024–2026 文献地图：优先读哪些？已经有人做过哪些？

> **严格分级**：CVPR/ICCV 主会属于 CCF-A；CVPR **Findings** 与主会应区分；IEEE TGRS、IEEE JSTARS、ISPRS JPRS 是**期刊**，不称“CCF-A 会议”。只列本次确实检索到题名、年份和论文/出版来源的条目。出现「未发现官方代码」仅表示**本次未核验到**，不代表绝对不存在。论文链接优先会议官方 CVF、出版社 DOI 或 arXiv；官方代码仅给确有证据的地址。

### 3.1 A组：与遥感变化检测**直接重合**（必读）

#### [A1] HES-Net（最直接、必须正面比较）

- **题名**：*Hyperboloid-Embedded Siamese Network for Change Detection in Remote Sensing Images*。
- **作者**：Qian Yang, Shujun Zhang, Jinsong Li, Yukang Sun, Qi Han, Yuanyuan Sun。
- **发表**：**IEEE JSTARS 2024，Vol.17，pp.9240–9252**，SCI 期刊，**非 CCF-A 会议**。
- **论文**：[出版社 DOI](https://doi.org/10.1109/JSTARS.2024.3390427)；[dblp 核验](https://dblp.org/rec/journals/staeors/YangZLSHS24.html)。
- **代码**：本次**未核验到作者官方公开代码**，请勿借用其他 HESNet 项目名称误链。
- **已做的核心机制**：Siamese+UNet，把双时相特征映射到双曲空间，以 **HSAM/HASM（文中缩写版本存在不一致，按正式正文 HSAM）双曲相似性注意力**融合；另有多尺度 DFEM 差分增强；在 CDD、LEVIR-CD 上实验。
- **与 CASA-CD 的实质差异**：它以**双曲距离构建融合注意力**；我们拟以**双曲距离决定低频 scan 前 cell 压缩的输入分配**，同时不改 scan 拓扑、保证预训练恒等和轻量预算。
- **审稿风险**：如果仅 `d_H→attention/weight→sum`，很容易被认为是 HES-Net 的评分换位置；必须证明**结构性池化分配、固定 lattice、与 TinyViM 的互补**。

#### [A2] HyperCLIP-CD（2026 年 TGRS：最需要防范的近年先例）

- **题名**：*HyperCLIP-CD: Hyperbolic Multimodal Representation Enhancement for Remote Sensing Change Detection*。
- **作者**：Qian Yang, Shujun Zhang, Xiao Li, Yunuo Wang, Yuanfei Zhou。
- **发表**：**IEEE TGRS 2026，Vol.64**；DOI `10.1109/TGRS.2026.3670127`；出版社记录入口为 [IEEE Xplore 11419177](https://ieeexplore.ieee.org/document/11419177)。本次浏览 IEEE 正文时遇到 JS/反爬墙，题名/DOI/期刊可经交叉书目核对，**未独立通读全文**。
- **论文**：[出版社 DOI](https://doi.org/10.1109/TGRS.2026.3670127)；[期刊参考书目核验](https://www.mindat.org/reference.php?id=19774808)。
- **代码**：本次未发现可核验的官方 GitHub；不提供猜测链接。
- **可核对摘要**：双分支 CLIP；用于变化区域的层次化文本提示 HTP；双曲空间图文对齐 HMSA；在 LEVIR-CD、CDD 和 SECOND 上验证（其中 SECOND 是**语义变化检测**，不在本项目任务范围）。
- **不同点**：该工作通过视觉—语言/图文对齐学习变化语义；CASA-CD 不引入 CLIP、文本提示或推理期重模型，研究的是 tiny SSM **scan 前低频信息压缩**。
- **审稿风险**：论文中绝不能说“首个将双曲结构应用于遥感图像变化感知”；只能主张具体操作层级及成本约束下的增量。

#### [A3] HSN：真实“变化监测”任务中的双曲 Siamese

- **题名**：*Deep Change Monitoring: A Hyperbolic Representative Learning Framework and a Dataset for Long-term Fine-grained Tree Change Detection*。
- **作者**：Yante Li, Hanwen Qi, Haoyu Chen, Xinlian Liang, Guoying Zhao。
- **发表**：**CVPR 2025 主会（CCF-A）**，pp.27346–27356。
- **论文**：[CVF 官方页](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Deep_Change_Monitoring_A_Hyperbolic_Representative_Learning_Framework_and_a_CVPR_2025_paper.html)；[arXiv 2503.00643](https://arxiv.org/abs/2503.00643)。
- **代码**：[作者官方 GitHub](https://github.com/liyantett/Tree-Changes-Detection-with-Siamese-Hyperbolic-network)。**本次已核验仓库存在，但 README 内容极简；不能未经执行就声称完整可复现。**
- **核心**：针对一年内树木生理状态变化及成像干扰，构造 UAVTC 与双曲 Siamese HSN；Poincaré 表征、双曲分类学习与 δ-hyperbolicity 分析。
- **可迁移启示**：要让“变化具有层次”的论点有**具体证据与可度量实验**，而非仅靠双曲负曲率的概念。
- **差异**：HSN 是长期树木个体的细粒度变化监测，不是本项目四个 `256×256` 的通用遥感二值变化像素分割协议；不能直接横比指标。

#### [A4] PID-CD：虽非严格双曲，但确实提出“manifold structures”用于变化特征修正

- **题名**：*PID-CD: PID Control is What Change Detection Needs*。
- **发表**：**IEEE JSTARS 2025，Vol.18，pp.10388–10402**，SCI 期刊。
- **论文**：[出版社 DOI](https://doi.org/10.1109/JSTARS.2025.3556096)；[DOAJ 期刊记录](https://doaj.org/article/cd1a694d59e0406ca88d2647423159ac)；[dblp](https://dblp.org/rec/journals/staeors/YeSHWXSFMYL25.html)。
- **代码**：本次未核验到作者官方仓库。
- **机制**：PIDFA（特征对齐）在摘要里明确采用 manifold structures 与自适应策略，配 PIDFF 多级融合、边界精化。
- **边界**：摘要没有充分证明它采用恒负曲率双曲空间；正式读全文前只能称**流形结构/几何启发**，不得把它误记为“双曲变化池化论文”。

### 3.2 B组：遥感图像中的双曲/流形表征（跨到变化检测很有价值）

#### [B1] HSAE：TGRS 2024，双曲背景—异常结构

- **题名**：*Hyperbolic Space-Based Autoencoder for Hyperspectral Anomaly Detection*。
- **作者**：He Sun, Lizhi Wang, Lei Zhang, Lianru Gao。
- **发表**：**IEEE TGRS 2024，Vol.62**。
- **论文**：[出版社 DOI](https://doi.org/10.1109/TGRS.2024.3419075)；[书目记录](https://www.mindat.org/reference.php?id=17500028)。
- **代码**：未核验到作者官方开源。
- **核心**：把 HSI 特征嵌入双曲空间，通过背景分布的层次建模与双曲距离提升异常检测；涉及双曲隐变量先验。
- **启发**：在遥感场景中，几何可以先用于**“正常背景”与离群变化证据分离**；但 HSI 光谱异常不等价于两时相 RGB 二值变化，不能直接拿其效果当支持证据。

#### [B2] Manifold-Geometric Regularized GCN：TGRS 2025

- **题名**：*A Nonlinear Weighted Graph Convolution Network Based on Manifold Geometric Regularization for Hyperspectral Image Classification*。
- **发表**：**IEEE TGRS 2025**，DOI `10.1109/TGRS.2025.3584561`。
- **论文**：[出版社 DOI](https://doi.org/10.1109/TGRS.2025.3584561)。
- **代码**：未核验到官方公开代码。
- **机制启示**：局部空间—光谱邻域的**非线性流形几何正则**帮助学习关系；可启发“邻域结构变化”而非纯点差异。
- **适配警告**：完整 GCN 或逐图动态图代价不适合 CASA-CD 的 ≤5M/规则 scan；不是首选模块。

#### [B3] ChangeCLIP：ISPRS JPRS 2024（非双曲，作为多模态边界对照）

- **题名**：*ChangeCLIP: Remote sensing change detection with multimodal vision-language representation learning*。
- **发表**：**ISPRS Journal of Photogrammetry and Remote Sensing 2024，208:53–69**，SCI 权威期刊。
- **论文**：[Elsevier 出版社](https://www.sciencedirect.com/science/article/pii/S0924271624000042)。
- **代码**：[官方 GitHub](https://github.com/dyzy41/ChangeCLIP)。
- **意义**：HyperCLIP-CD 的视觉语言路线有直接背景；不应与纯轻量双时相 CAACP 搞混。

### 3.3 C组：CVPR / ICCV 2024–2025 几何模块与反例（重点关注可复用思想）

| 编号 | 经核实的论文（年份/级别） | 实质机制、对 CASA-CD 的启发 | 论文 | 官方代码 |
|---|---|---|---|---|
| C1 | **Hyperbolic Anomaly Detection**（CVPR 2024 主会，CCF-A） | 欧氏特征→Poincaré 双曲距离，评估异常；最接近“以几何距离做变化证据”，但不是时相配对 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hyperbolic_Anomaly_Detection_CVPR_2024_paper.html) | **未核验** |
| C2 | **Flattening the Parent Bias: Hierarchical Semantic Segmentation in the Poincare Ball**（CVPR 2024 主会，CCF-A） | 双曲层次语义分割；也提示“层次设计未必带来精度收益”，需要域外和校准检验 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Weber_Flattening_the_Parent_Bias_Hierarchical_Semantic_Segmentation_in_the_Poincare_CVPR_2024_paper.html) | [tum-vision/hierahyp](https://github.com/tum-vision/hierahyp) |
| C3 | **Improving Visual Recognition with Hyperbolical Visual Hierarchy Mapping**（CVPR 2024 主会，CCF-A） | Hi-Mapper 以视觉层次关系建模支持稠密预测；CASA 可学习其“先证实层次，再用双曲”方法论 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kwon_Improving_Visual_Recognition_with_Hyperbolical_Visual_Hierarchy_Mapping_CVPR_2024_paper.html) | [kwonjunn01/Hi-Mapper](https://github.com/kwonjunn01/Hi-Mapper)，**仓库页面曾标注完整代码尚待发布，须审计可复现范围** |
| C4 | **Hyperbolic Category Discovery**（CVPR 2025 主会，CCF-A） | HypCD 同时处理双曲距离与角度；提示“径向+角向”解耦可以有价值，但任务是开放类别发现，不是本项目 | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.html) | [Visual-AI/HypCD](https://github.com/Visual-AI/HypCD) |
| C5 | **Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space**（CVPR 2025 主会，CCF-A） | HyperCLIP 研究**双曲半径与语义层次粒度**的对应；启发 CAACP 使用非恒定半径。但它属于 CLIP 分割，且**不是 HyperCLIP-CD** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Peng_Understanding_Fine-tuning_CLIP_for_Open-vocabulary_Semantic_Segmentation_in_Hyperbolic_Space_CVPR_2025_paper.html) | [SJTU-DeepVisionLab/HyperCLIP](https://github.com/SJTU-DeepVisionLab/HyperCLIP) |
| C6 | **Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval**（ICCV 2025 主会，CCF-A） | 在没有显式层次标签时约束视觉多层关系；提示如何**检验隐式局部层次是否真实存在** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Learning_Visual_Hierarchies_in_Hyperbolic_Space_for_Image_Retrieval_ICCV_2025_paper.html) | **未核验** |
| C7 | **Accept the Modality Gap: An Exploration in the Hyperbolic Space**（CVPR 2024 主会，CCF-A） | 指出直接拉近双曲模态特征可能破坏层次，提醒不能把“距离越近”当作普遍目标 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ramasinghe_Accept_the_Modality_Gap_An_Exploration_in_the_Hyperbolic_Space_CVPR_2024_paper.html) | **未核验** |
| C8 | **Hyperbolic Learning with Synthetic Captions for Open-World Detection**（CVPR 2024 主会，CCF-A） | 双曲视觉—文本层次对齐；仅作为跨领域支持，不导入重型 VLM | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kong_Hyperbolic_Learning_with_Synthetic_Captions_for_Open-World_Detection_CVPR_2024_paper.html) | **未核验** |

**阅读优先级**：`HES-Net → HyperCLIP-CD → HSN → HypAD → HSAE → HyperCLIP(CVPR) → HypCD → Poincaré hierarchical segmentation`。先读与双时相变化直接相关的工作，再补学数学；不建议一上来照搬图神经网络、双曲 Transformer 或跨模态大模型。

### 3.4 D组：与当前 CAACP 的“压缩发生在哪里”有关（不是流形论文）

- **SAT** — *SAT: Selective Aggregation Transformer for Image Super-Resolution*，2026 年论文；本次用户附 PDF 标示 **CVPR 2026 Findings**，**不是 CVPR 主会同义词**。论文：[arXiv 2604.07994](https://arxiv.org/abs/2604.07994)；[官方 GitHub](https://github.com/PhuTran1005/SAT)。其 DTA 根据局部密度与孤立程度选择中心、聚类相似像素，再加权汇聚 Key/Value、保留 full Query。**CAACP 的区别不是“新提出加权聚合”**，而是双时相变化证据、严格 2×2 规则 lattice、SS2D scan 前低频重构与 TinyViM 预训练继承。PDF 实测宣称 97% K/V token reduction，但不是 CASA 的 2×2 压缩率。
- **TinyViM** — *TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba*，**ICCV 2025 主会（CCF-A）**；[官方项目](https://github.com/xwmaxwma/TinyViM)。低频扫描/高频局部运算是被继承的骨干机制，**不能计为 CASA 的原创**。
- **本项目 STR** — TAR/DCR 训练丰富、部署可折叠，当前应用与 CAACP 属两条创新主线；**不要因为使用双曲分数就宣称整个 TAR/DCR 是双曲可折叠层**（双曲 `acosh/exp` 本身不可与卷积精确折叠，只能作为细小的推理期打分操作保留）。

### 3.5 文献覆盖与证据限制

检索覆盖了 `hyperbolic/poincare/Lorentz/manifold/Riemannian/geodesic` × `remote sensing/change detection/segmentation/hyperspectral/anomaly` 的多组组合，核对了 2024–2026 官方 CVF 页面、IEEE DOI/书目、ISPRS 出版页、作者代码。**这不能证明整个学界不存在遗漏论文**。特别是 HyperCLIP-CD 的 IEEE 正文无法在当前网络直接逐页访问，其精细模块数学式、消融和参数量有待通过作者论文全文复核；因此报告避免写未经证实的具体数字和内部细节。

---

## 4. 具体应用路径：流形究竟能在哪些遥感变化问题上发挥作用？

| 变化问题 | 可采用的几何观点 | 预期适用场景 | 主要失败风险 |
|---|---|---|---|
| 光照/色彩/季节引起伪变化 | 比较相对局部结构或背景流形偏移，而非只比较两点颜色 | CDD，LEVIR/WHU 非变化纹理 | 纹理/阴影本身也改变局部结构 |
| 小建筑、小道路、零碎变化 | 细尺度局部关系/角向突变；将非均匀变化证据保留在 cell 上 | LEVIR/WHU/SYSU 小对象 | Stage3 1/16 已是 sub-token，小对象信息可能在更早期消失 |
| 地物部件—整体层次 | 双曲径向编码“局部 vs 区域结构粒度” | 建筑边缘、道路网络、树冠局部 | **是否真的近似树状未知**；若只是普通纹理，高维双曲可能无益 |
| 空间图/超图关系 | 邻域图关系、图测地/流形 Laplacian | 遥感复杂连接结构 | 需要图构建/邻域重排，破坏极轻量与规则 scan |
| 局部统计分布改变 | SPD 协方差/子空间距离 | 光谱丰富、区域足够大时 | 2×2 只有 4 点；168 通道协方差严重奇异，矩阵对数/SVD 开销高 |
| 时相配准/几何错位 | 先做空间几何对齐，再度量变化 | 多传感器和高视差 | 当前只能严格比较统一配准的四库；不得为了新模块偷偷改变输入对齐协议 |

**重要区分**：所谓“曲率鲁棒于伪变化”不是数学公理。局部上下文半径也可能更敏感于阴影、错位、树冠摆动，因此应把这些作为**反例切片**，用图像级场景分组和对象级结果检验，而不是预设成功。

---

## 5. 2–4 个候选机制横向比较：只选一个进入首轮改动

| 候选 | 具体改动 | deploy 参数增量（预期） | 推理运算/兼容性 | 与既有论文的增量 | 判断 |
|---|---|---:|---|---|---|
| **A. 固定半径双曲距离** | `s=1−cos`→`acosh(1+const·s)` | 0 | 每点 `acosh`；rank 等价 | 与 HES-Net 距离思路近，**rank 退化** | **否决作为创新**；只能做数学负对照 |
| **B. CRH-CAACP：上下文径向双曲变化评分** | 半径由同一时相的 2×2 局部一致性产生；角向由 A/B 特征方向产生；距离给同一 cell 共享权重 | **0**（参数自由打分；保留原 β） | 16² 上少量向量规约、`cosh/sinh/acosh`；规则网格、权重、预训练不变 | 区别于 HES-Net 的全特征双曲注意力；**直接面向 SS2D 前压缩** | **首选，先验证非退化性** |
| **C. 球面结构关系差异** | 比较 A/B 两时相在 2×2 cell 内的方向/Gram 结构变化 | 0 | 4×4 Gram，稳定且廉价 | 可能已被欧氏局部相似性覆盖；曲率贡献弱 | **必要欧氏/球面对照，不单独当主创新** |
| **D. Grassmann/SPD 邻域流形** | 邻域协方差或子空间距离驱动 2×2 聚合 | 0–少量 | batched SVD/eigh；2×2 样本太少；算子部署复杂 | 数学性强，但与轻量 CAACP 不合 | **默认否决**，除非先证明大幅收益且成本可控 |

**最小可证伪主张**：B 的“非恒定径向局部结构”应该在固定方向相似度下**改变重要性排序**；且这种改变应该比在相同半径数据上使用普通欧氏距离更有益。如果前者不发生或后者同样好，论文就不能归因为双曲曲率。

---

## 6. 首选方案：CRH-CAACP（Context-Radial Hyperbolic Change-Aware Context Pooling）

### 6.1 不涉及损失创新的机制动机

现有 CAACP 只问：“A/B 同一个 1/16 位置的特征方向是否不同？”

CRH-CAACP 进一步问：“**它在各自时相的局部组织中扮演的角色，是否也发生了变化？**”

例如一个 2×2 cell 中，原本四个位置都属于同质背景，后来其中一格出现屋顶/边缘；除了其 A/B 方向不同，该位置与邻域的关系也改变。我们将**方向**与**局部相对位置**分别写入双曲空间的角向与径向，再决定哪个像素在低频汇聚里得到更多权重。这是**假设**，不是已获实验证实的机理。

### 6.2 数学定义（参数自由、严格 A/B 交换对称）

给定 `X_A,X_B∈R^{B×168×16×16}`，对时相 \(t\in\{A,B\}\)、2×2 cell \(\mathcal C_k\) 内第 \(i\) 个空间位置：

**（1）单位方向：**
\[
 u_{t,i}=\frac{x_{t,i}}{\max(\|x_{t,i}\|_2,\varepsilon)}.
\]

**（2）局部中心方向：**
\[
 m_{t,k}=\operatorname{Normalize}\left(\frac14\sum_{j\in\mathcal C_k}u_{t,j}\right).
\]

**（3）局部径向离散度（每个时相独立计算；不引入 GT、类原型或额外损失）：**
\[
 q_{t,i}=\operatorname{clip}\left(\frac{1-u_{t,i}^{\top}m_{t,k}}2,0,1\right),
 \qquad r_{t,i}=r_0+a\,q_{t,i}.
\]

建议**首轮预先固定** `r0=0.5, a=0.5, curvature=−1`，不是事后搜索出来的最优超参。这样 \(r\in[0.5,1]\)，远离双曲数值边界。参数取值本身**没有理论最优保证**，若性能未达到目标不得靠扫曲率/半径来包装收益。

**（4）双时相 Lorentz 测地距离：**
\[
 t_i=\max\Big(1,\ \cosh r_{A,i}\cosh r_{B,i}
 -\sinh r_{A,i}\sinh r_{B,i}(u_{A,i}^{\top}u_{B,i})\Big),
\quad d_i^H=\operatorname{acosh}(t_i).
\]

这一步只需方向点积和标量运算，无需显式保留 `C+1` 个双曲空间通道。

**（5）变化强度与置信度：**
\[
 s_i^H=1-\exp(-d_i^H),\quad
 \rho_i=\operatorname{Rank}_{\rm image}(s_i^H),\quad
 \widetilde q_i=s_i^H\rho_i.
\]

**（6）保持 CP-CAACP 的无变化退化规则和 2×2 共享权重：**
\[
 w_{k,i}=\frac{1+\widetilde q_i}{\sum_{j\in\mathcal C_k}(1+\widetilde q_j)};
\quad c^t_{H,k}=\sum_{i\in\mathcal C_k}w_{k,i}\,z^t_i;
\quad c^t_k=c^t_{\mathrm{avg},k}+\beta(c^t_{H,k}-c^t_{\mathrm{avg},k}),
\]
其中 \(z^t_i\) 是 **SS2D 内部低频分支**待池化特征，不是原始 `x` 的全部 168 通道。

由于公式中的 A/B 始终对称，交换时相会保持 \(s_i^H, w_{k,i}\) 不变；因为 A/B 都使用同样的 `w`，不会产生不一致 cell assignment。当局部以及像素特征完全不变时，\(r_A=r_B,u_A=u_B,d_H=0\)，所有权重退化为 `1/4`；再加上 β=0 初始化，**epoch-0 必须逐位恢复旧模型行为**。

**注意**：半径只是我们定义的**局部结构离散度代理**，尚不能声称它已学出“地物语义层次”。如果后续实验没有证据说明径向信息不同于简单的边缘/纹理能量，就应将方法表述为“上下文径向测地评分”，避免夸张地叫“语义层次发现”。

### 6.3 同构的欧氏径向对照（鉴别曲率是否真有价值）

用**完全相同的** `u_A,u_B,r_A,r_B` 和 `q`，仅将双曲测地距离改为切向欧氏距离：
\[
 d_i^E=\sqrt{r_{A,i}^2+r_{B,i}^2-2r_{A,i}r_{B,i}\cos\theta_i},
\qquad s_i^E=1-\exp(-d_i^E).
\]

其余 rank、CP 权重、β、训练图完全一致。这个对照不可省略：

- **H 与 E 同样提升**：得到的是“局部结构径向信息有用”，**不是双曲曲率被证实有用**；
- **H 明显优于 E**：为曲率的增量价值提供初步证据，仍需可视化、统计与跨库验证；
- **H≃当前 cosine**：没有充分理由引入 `acosh` 推理成本；
- **H 退步**：说明当前局部径向代理可能放大了伪变化或损坏了预训练信息，停止扩展。

### 6.4 训练图与部署图

```text
train / inference, both:
  Shared TinyViM Stage3 prefix X_A, X_B
       │
       ├── original score (baseline)
       │              or
       └── CRH-score (no_grad, zero new parameters)
                 │ 16×16 scalar map
             shared 2×2 weights
                 │
             existing CAACP beta gate
                 │
             existing SS2D scan
                 │
             existing TAR/DCR/Head
                 │
             change probability

additional train-only HFAE aux head (only if E9 is the fixed parent):
  unchanged; removed by existing switch_to_deploy()
```

- **新 scorer 训练可学习参数**：`+0`；**deploy 参数**：相对相同 E9 父模型**预期 `+0`**，即目标仍为 **4,880,190 ≤5M**；必须由真实构图逐层复数，不能只依据公式口头确认。
- **FLOPs**：并非零增量；`normalize / 2×2 neighborhood mean / exp / acosh / rank` 都是在**部署**中实际执行的运算。应报告：`MACs/Conv FLOPs + scorer 实际算子与 wall latency`；固定 16×16 尺度预计远小于主干，但尚未实测。普通 FLOPs 工具对 `acosh`、排序、selective scan 的统计可能漏项，须用 CUDA event、同步 warmup/重复测试补充 **ms/张、ms/batch32、显存**。
- **结构重参数化**：CRH 分数不是训练多分支卷积，也**不试图将双曲函数折叠成卷积**。不影响 TAR/DCR 已有的单卷积折叠；训练期/部署期的 score 函数保持相同。
- **预训练**：不新增 SS2D 权重或修改尺寸，原 TinyViM-S-Slim 参数映射不变；β=0 使模型输出一致，不代表 scorer 与原余弦分数相同。

### 6.5 与现有文献相比，能真正写成创新的是什么？

**不能写**：“首次用双曲几何处理遥感变化 / 首次使用双曲相似性 / 第一次把两时相映射到 Poincaré 球 / 双曲空间必然抗伪变化”。这些均与既有文献或数学事实冲突。

**可以提出、但必须实验证实**的主张是：

1. **Change-conditioned geodesic evidence for structured pre-scan aggregation**：以双时相几何差异决定低频扫描前固定 cell 的信息保留权重，而不是在解码阶段做一般融合注意力。
2. **Radial-angular decomposition beyond monotonic cosine remapping**：将局部空间相对关系编码为可变径向，刻意避开“固定半径双曲=余弦排序”的退化；用相同径向信息的 Euclidean comparator 鉴别真正的曲率贡献。
3. **Pretraining/deploy constrained geometry**：不改变 TinyViM 核心 SS2D 权重、lattice、A/B 共享性和 STR 可折叠路径；增量 scorer 不引入训练或部署参数，并量化其真实推理代价。

**这三个不是三个任意堆叠模块**，而是同一设计选择的*位置、非退化几何与约束*三方面定义。若曲率本身没有增量，最好如实退回更易部署的欧氏局部结构评分，不强行保留“双曲”故事。

---

## 7. 逐文件修改清单：以最小改动落到 CASA-CD

> 此处是**实施设计**，并非已经在用户仓库改完的代码。所有实验写入新 Run 目录，未授权不动现有 checkpoints/log。

| 优先级 | 文件 | 最小改动 | 需要验证的边界 |
|---|---|---|---|
| **P0** | `models/model/layers/caacp_ss2d.py` | 新增纯函数 `change_score_crh_2d(xa,xb)` 和 `change_score_radial_euclidean_2d`；`score_mode` 新增 `crh_rank` / `radial_euclidean_rank`；首轮严格沿用现有 rank 权重公式（CP 仅作后续候选） | `16×16→8×8`，无 NaN/Inf，权重非负、和=1，`β=0` 输出逐位一致 |
| **P0** | `models/model/casa_tvim_str_net.py` | 在 `with torch.no_grad()` score 分支按 `score_mode` 选择原 cosine / CRH / Euclidean comparator；沿 batch 复制同一个 score | A/B 共享权重、score swap-invariant，输入与原 code 完全同层 |
| **P0** | `models/model/tinyvim_s_slim.py` | 放宽 `score_mode` 枚举断言，不更改网络宽度、Stage4 裁剪、SS2D state_dict 继承 | 744 等预训练键统计对照，不丢已有权重 |
| **P0** | `models/train.py`、`models/eval.py` | CLI 选项与 checkpoint sidecar 加入明确 `score_mode`；复制现有严格 80K、同评价协议 | 同期 CTRL 与 H/E 代码身份、超参数冻结 |
| **P0** | `models/smoke_test.py` 或新建 `models/test_crh_caacp.py` | 退化证明数值测试、交换性、零变化、梯度、部署折叠 | train↔deploy 二值 disagreement=0，真实 batch 也要测 |
| **P1** | `analyse/tvim_caacp_counterfactual.py`（尽量复用接口） | 支持新 checkpoint 的 β ON/OFF，固定每张图记录 ΔF1、small Recall、cell 熵、`||βΔC||` | **不能**把同一权重 β ON/OFF 当成从头无 CAACP 的消融 |
| **P1** | `analyse/` 新增 `caacp_geometry_preflight.py` | 原 cosine、固定半径 H、CRH、radial-E 四种分数同输入可比；rank 相关与 GT AP 仅作诊断 | 与 train split/test split 独立；不把 GT 注入 inference |
| **P1** | `train_scripts/CASA-TViM/RunH1/`（新建） | `README.md`、`SOURCE_IDENTITY.json`、`run_all.sh`、`check_run.py`、冻结 `SELECTION_LOCK.json` | 产物和日志不得覆盖 Run7；失败保留完整原因 |
| **P2** | `docs/参考文献/文献索引.md` | 将 HES-Net/HyperCLIP-CD/HSN/HSAE/CVPR 双曲作品登记（注明 tier 与代码状态） | 仅做论文索引，不应改评价协议 |

**实现伪代码（核心约 20 行逻辑）**：

```python
# Score only: inputs are the SAME stage3-prefix features as existing cosine.
# Shapes: xa, xb [B,168,16,16]; output score [B,16,16].
@torch.no_grad()
def contextual_radial_score(xa, xb, geometry="hyperbolic"):
    def describe(x):
        u = torch.nn.functional.normalize(x.float(), dim=1, eps=1e-8)
        mean_u = torch.nn.functional.avg_pool2d(u, 2)
        mean_u = torch.nn.functional.interpolate(mean_u, scale_factor=2, mode="nearest")
        mean_u = torch.nn.functional.normalize(mean_u, dim=1, eps=1e-8)
        q = ((1 - (u * mean_u).sum(1)) / 2).clamp(0, 1)
        radius = 0.5 + 0.5 * q
        return u, radius

    ua, ra = describe(xa)
    ub, rb = describe(xb)
    cos_ab = (ua * ub).sum(1).clamp(-1, 1)
    if geometry == "hyperbolic":
        z = torch.cosh(ra)*torch.cosh(rb) - torch.sinh(ra)*torch.sinh(rb)*cos_ab
        distance = torch.acosh(z.clamp_min(1.0))
    elif geometry == "euclidean":
        distance = (ra*ra + rb*rb - 2*ra*rb*cos_ab).clamp_min(0).sqrt()
    else:
        raise ValueError(geometry)
    return (1 - torch.exp(-distance)).clamp(0, 1)
```

上述仅定义 `s`，`rank`、权重、池化、β、SS2D 由现有类实现。**注意：不是可直接复制替换整个 `caacp_ss2d.py` 的完整 patch。** 实际落地时应使用 2×2 元素索引一致性测试，确认 `avg_pool2d→nearest upsample` 所有空间位置确实复用其 cell 中心，且 fp32→输入 dtype 的流程不会导致 `c_avg+βΔ` 在 β=0 时产生非有限数值。

---

## 8. 可证伪假设、最小实验和明确失败判据

### 8.1 预注册假设（先写，后看数据）

**H0：几何退化基线。** 固定半径 H 与余弦 score 的 rank 一致（浮点边界例外），在 `rank`-CAACP 中 cell 权重相同。若不成立先查实现/并列 rank 处理，不能拿误差冒充增量。

**H1：非退化性。** CRH 在某些图像/对象边界附近产生可解释、非数值误差所致的 rank 变化；其变化和局部结构差异相关。若 rank 几乎完全一致，CRH 没有充分机制增量。

**H2：准确性。** 在其他条件相同下，CRH 的四库正式 TEST F1 相比同期父模型得到受控提升，尤其不牺牲 SYSU small Recall / Hit@25；不过 CDD 的伪变化 Precision 不能显著倒退。

**H3：曲率特异性。** 当局部径向证据、权重映射和训练预算固定时，CRH 的提升明显超过欧氏径向对照；如果二者同样有效，否定“曲率必要”。

**H4：实际 CAACP 被使用。** 新 scorer 导致的 \(\beta\Delta C\) 会在网络输出中产生可测差异；否则不能将端到端结果归因为 CAACP 的推理期几何路径。训练中的变化也可能表现为优化轨迹效应，需控制。

**H5：轻量与等价性。** 部署有效参数 ≤5,000,000，TAR/DCR 折叠后单卷积、真实 batch 阈值 0.5 disagreement=0，TinyViM 预训练读取逐位保留。

### 8.2 第一阶段：无需 80K 的机制诊断（不作正式结果）

复用已有 Run7 父模型和四数据集 split，固定样本/文件名，不重新抽样直到显著：

1. **Score 等价性检查**：cosine vs fixed-radius H，报告 Spearman/Kendall、rank 不同数、cell 权重 `max_abs`；正常应近似 0 不同。
2. **结构性来源**：cosine、CRH、radial-E 在相同 A/B 特征上比较；报告 `rank-disagreement`、每个 cell `top1 index` 改变率、score 与原 rank 的相关性。
3. **真实变化定位能力**：诊断性计算 16×16 GT 占有率与 score 的 PR-AUC、boundary PR-AUC、small-object 命中；由于 Stage3 已 1/16，必须说明 `256→16` 的掩码降采样方法，并将**每 cell GT occupancy** 与 pixel GT 区别开。这个结果不等于最终 CD F1。
4. **环境干扰组**：CDD 伪变化样例、LEVIR/WHU 小建筑、SYSU 小目标，记录权重热图与失败例，不允许只展示漂亮案例。
5. **β 强度基线**：先在 E9 父模型重跑 `||βΔC||/||c_avg||` 和 β OFF 反事实，确定与旧 Diag1 的差别。
6. **成本**：单独 scorer 在 RTX 5090 FP32/实际模型 dtype 的 GPU event 时延，前向/反向 peak VRAM；评估时不应只看 fvcore 估算。

**诊断通过只说明“有研究价值”，不是性能保证。** 如果只发现“CRH 换了排序但 GT AP 退步”，依照当前用户“完整训练是正式裁决”的 Run7 工作方式，可以继续一个预注册 80K 作为正式否证；绝不可把代理指标替代 TEST，也不能后续随意换 radius/λ 反复刷诊断。

### 8.3 第二阶段：唯一变量、四库同预算 80K

建议沿用 **Run7 胜者 E9_HFAE_A 作为固定父结构**，因为当前进度已进入 Run7，不能退回早期 M1 再宣称是新主线。HFAE 的原有 train-only 辅助监督在所有组**逐字相同**，不是本次创新，不改其 `λ=0.2`。

| 实验组 | 与同期 E9 父模型相比的唯一变量 | 作用 | 建议预算 |
|---|---|---|---|
| **G0：E9 + 原 cosine rank** | 无 | 同期强基线；同代码身份、同实验脚本 | **4 × 80K** |
| **G1：E9 + CRH-CAACP** | `score_mode=crh_rank`；沿用 G0 的 rank 权重，其余不变 | 唯一主实验 | **4 × 80K** |
| **G2：E9 + radial-E** | 仅 `geometry=euclidean`，与 G1 全部径向量/权重函数一致 | 检验“曲率是否必要”的关键对照 | **优先 SYSU+LEVIR 2 × 80K**；若论文主张跨库曲率效应，再补 CDD+WHU 2 × 80K |

**P0 纠偏**：如果 G0 采用当前仓库 E9 的 `rank` 模式而 G1 使用本方案所写的 `1+s·rank`（CP 权重），这就**同时改变了分数和加权公式**，不是唯一变量。正式启动前须选择以下一种并冻结：

- **方案 I（备选，改变共同的权重基准）**：三组统一采用 `w∝1+s·rank`；G0 以原 cosine 作为 s，G1/G2 分别采用双曲/欧氏径向 s。此时 G0 是“E9 + 同期 CP”而非历史 E9 原 `rank`，必须如实命名；可另外把不训练的旧 rank 权重做审计参照。
- **方案 II（首轮推荐，最保守）**：三组统一采用现有 `rank` 权重；G0 用 cosine rank，G1 用 CRH rank，G2 用 radial-E rank。该方案更加严格隔离“距离引起的排序变化”，也更直接对应本报告第 2.3 节的退化定理。**首轮建议方案 II**：最小改动、不改原 score-to-weight、无新增超参。

**推荐首轮锁定方案 II**。正文 §6.2 的 CP 数学式将保留为**后续可验证版本**，**不进入首轮 G0/G1/G2**。每个 run 的唯一开关为 `score_mode ∈ {rank, crh_rank, radial_euclidean_rank}`，`CAACPSS2D` 用同一个 `(rank+eps)/sum`。先验证纯几何非退化性，避免“置信度变换”混杂。若这一路线全部失败，再根据正式证据另立新课题与预注册；不得把 CP 变体并入同一实验中挑赢家。

**固定协议**：数据集四库、`A/B/label + list/*.txt`；`gray≥128`；共享几何增强；TinyViM-S-Slim ImageNet EMA；batch 32、256²、seed 16、严格 80,000 次 optimizer.step、Adam `(2e-4,0.9/0.99,wd1e-4)`、200 iter warmup+poly 0.9；与父模型一致的 BCE+Dice 和 HFAE 固定训练辅助项；沿用当前 `test` 当验证集、依 test F1 选 best 的历史协议，**需在论文披露这项测试集模型选择的评估局限**。禁止用新 loss、阈值、增广、多种子挑最优来解释成功。

**形式化判据建议在 `SELECTION_LOCK.json` 中提前冻结**：

- **技术硬门**：epoch-0 恒等逐位通过；预训练继承核验；train↔deploy 二值 disagreement **=0**（随机+真实 batch）；有效 deploy Params ≤5M；运行不产生非有限分数；严格 80K；四库协议一致。
- **性能主门**：参考 Run7 接受规则，G1 相对 G0 **≥3/4 库 F1 不下降、SYSU 严格提升**；另查看 CDD/WHU 不产生异常大下降。单种子小幅收益只可称“本协议下的受控观察”，不能称普适显著优势。
- **机理门**：G1 的 score 相对 G0 有非零、非仅数值噪声的排名变化；`β` 与 pooled context 修正可检查；对象级小目标/伪变化结果支持机制解释；G1 与 G2 差异清楚。**G1 不优于 G2 ⇒ 不支持双曲曲率必要**。
- **失败裁决**：若 G1≥G0 但 G1≈G2，结论改为“局部结构径向证据可能有益”；若 G1≤G0，否决 CRH 主线；若 G1 有增益但只来自某一库，不写跨数据集稳定提升。

### 8.4 每个正式 run 需要报告的字段

| 类别 | 必须归档 |
|---|---|
| 性能 | **Recall / Precision / OA / F1 / IoU / Kappa**（最后完整 TEST 区块） |
| 模型规模 | **总参数量、可训练参数量、部署有效参数量**；更换 scorer 前后差 |
| 计算 | **FLOPs**（列出不支持统计的算子）、真实 GPU latency、显存/吞吐 |
| CAACP 机制 | β、`||βΔC||/||c_avg||`、cell 权重熵、top1 cell 保留率、评分 rank 非退化度 |
| 对象/边界 | small pooled Recall、Hit@25、ObjP/ObjR、未匹配小 FP、边界带 PR-AUC；与主指标口径分开 |
| 一致性 | 预训练键/逐位误差、随机/真实 batch train↔deploy max_abs、**0.5 disagreement=0** |
| 复现 | Git commit、源文件 SHA256、seed、实际 optimizer step、GPU、checkpoint 路径、日志路径、配置 JSON |

---

## 9. P0/P1/P2 风险清单（先解决正确性，再谈论文）

### P0 正确性

1. **严禁半径固定导致等秩“假创新”**：将第 2.3 节退化定理写成单测。
2. **严禁把 `cp` 权重和 hyperbolic score 同时改动**：第 8.3 节最后锁定了**方案 II**，其余所有组一字不差。
3. **严禁 A/B score 误分配**：`torch.cat([s,s],dim=0)` 的 batch 对齐必须与 `x2b` 一致；score 不走 GT。
4. **严禁改变 2×2 的顺序/宽高**：不能引入全局聚类、TopK 重排、不同扫描方向不同 weights。
5. **严禁 FP16 双曲数值溢出**：半径有界、FP32 score、所有 score/weight finite；zero score 的 `rank` 并列行为应专门测试。
6. **严禁训练/部署不同评分**：部署 scorer 仍执行同一函数；不得“测试时偷偷退回 cosine”。
7. **严禁伪造参数/ FLOPs**：新 scorer 无可学习参数不代表无推理开销，须同时核对 `fvcore unsupported_ops` 与实际 GPU 延迟。
8. **严禁覆盖历史日志/checkpoint**：必须新建 RunH1，测试标签阈值 `gray≥128`，保留同期全部原始记录。

### P1 方法瓶颈

1. **β 修正通道弱**：先核验 Run7 E9 的 β ON/OFF；若仍≈0，score 更复杂可能没有端到端影响。
2. **尺度错位**：Stage3 1/16 的 2×2 score 不必然救活 SYSU 的 tiny 目标；某些目标在 16² 已不可定位。
3. **“层次”未经证明**：局部离散度也可以用欧氏比较；双曲增益必须超越 radial-E。
4. **跨库背景噪声**：CDD 强伪变化可能出现 CRH rank 高而 GT 无变化，Precision 下降。
5. **过大非线性评分开销**：`acosh`+排序尽管只在 16²，真实 latency 未测不能称 0 FLOPs。
6. **与 HFAE 的交互**：E9 的边缘辅助监督改变骨干高频表征；在 E9 上做 scorer 训练时必须维持 HFAE 的监督预算与 tap 行为不变，避免误把辅助监督收益归给曲率。

### P2 实验工程

- PyTorch `2.14.0+cu132` / sm_120 / RTX5090；SS2D 用 `selective_scan_cuda_oflex`，训练脚本保留 `LD_LIBRARY_PATH` 指向 torch/lib。
- Run7 已修复过 HFAE forward-hook 注册非幂等问题；新方案不要碰 hook 相关代码。
- `score_mode` 需要被 checkpoint sidecar、eval 构造参数、日志表头和检查器**一致记录**；否则权重恢复后可能错用评分。
- 若恢复训练，需检查 model/optimizer/scheduler/actual_steps/RNG 与源码身份；不承诺跨设备逐位相同，只披露准确恢复状态。

---

## 10. 最小 smoke / dry-run 验收（新方法未跑之前不允许宣布成功）

| 代号 | 测试 | 通过要求 |
|---|---|---|
| T0 | `beta=0` 下旧 E9 结构 vs G1/G2 随机输入，同权重 | **逐位一致**；必须比较真实数据 batch |
| T1 | `F_A=F_B` 且相同局部结构 | score=0（允许明确规定的数值容差）；每 cell `w=0.25` |
| T2 | 交换 A/B | 变化 score / cell weights 完全不变（浮点容差内） |
| T3 | 固定半径 H vs cosine | rank 相同；rank-weight `max_abs≈0` |
| T4 | 真正可变半径 H vs radial-E | 数值有限；可以看到几何距离的差异，但**不把不同本身当精度提升** |
| T5 | 每 cell 的权重和、形状、索引 | `w.sum(cell)=1`；`16²→8²`；无全局重排 |
| T6 | 前向+反向 | β、SS2D 预训练参数梯度有限；由于 scorer `no_grad`，不得伪称分数本身有梯度 |
| T7 | 预训练加载 | 继承键数量/数值对齐原 E9；新方法不增加额外 checkpoint 缺失键 |
| T8 | STR fold train↔deploy | max_abs 符合已有容差，**0.5 disagreement=0**（随机及实际 batch） |
| T9 | 真实 DataLoader 3-step dry-run | label `gray≥128`、A/B/label 同步几何增强、无 NaN/OOM |
| T10 | 复杂度 | deploy ≤5M；测量 scorer 真实 GPU 时延；报告 FLOPs 不支持项 |
| T11 | CLI/checkpoint 恢复 | `score_mode` 一致、80K exact step 完整、源身份锁定 |
| T12 | GT 改写抗泄漏 | 保持 A/B 不变，临时改变诊断用 GT，不应改变模型 score/forward；GT 仅进入 loss/评价 |

**启动顺序**：本地源码审计与单元测试 → RSML-3 oflex GPU smoke → 四库真实 batch dry-run → 冻结 `SELECTION_LOCK.json` 与 SOURCE_IDENTITY → 同期 G0 → G1 → 先对关键 SYSU/LEVIR 补 G2 → 按正式结论决定 G2 是否四库补齐 → 每个 run 末完整 TEST + JSON 审计 → 汇总表/失败解释。避免并行执行时互相覆盖目录、损坏 checkpoint。

**建议输出目录（新目录，无需清理旧产物）**：

```text
/home/yqwang/outputs/CASA-CD/CASA-TViM/RunH1/
  G0_COS_RANK/{CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256}/train_log.txt
  G1_CRH_RANK/{...}/train_log.txt
  G2_EUCLIDEAN_RADIAL_RANK/{...}/train_log.txt
  _RESULTS/summary.json
  _RESULTS/geometry_mechanism.json
  _RESULTS/verdict.json

/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/RunH1/
  G0_COS_RANK/{dataset}/last.pth / best_F1=xxx.pth
  G1_CRH_RANK/{dataset}/last.pth / best_F1=xxx.pth
  G2_EUCLIDEAN_RADIAL_RANK/{dataset}/last.pth / best_F1=xxx.pth
```

具体 `.pth` 命名应沿用训练脚本实际规则；上例 `best_F1=xxx.pth` 是**格式占位符**，不是已经生成的 checkpoint。

---

## 11. 如何从“模块”讲成 TGRS 审稿人能检查的论文机制？

建议的科学问题不是“欧氏空间不行，所以我用双曲空间”，而是：

> **在预训练紧凑型频率解耦状态空间模型中，扫描前规则格低频上下文压缩只利用两时相点特征的角向差异，可能忽略同一像素相对局部结构角色的变化。能否通过曲率感知的径向—角向几何证据，在不增加部署参数、不破坏规则扫描及预训练继承的前提下，使低频聚合更偏向真实变化位置？**

必须有三个层次的证据：

**层次 1：数学必要性。** 解释固定半径双曲距离与 cosine rank 等价，证明为什么选了局部非恒定半径，而不是“只增加一个 acosh”。

**层次 2：结构必要性。** 证明评分确实改变了 scan 前低频 cell 权重及 \(\beta\Delta C\)，并查看是否改善了特定场景：局部变化边缘、小目标或强伪变化；与在 decoder 上加普通双曲注意力的先例形成结构差异。**注意：不能仅因把模块放在 scan 前就宣称自己是首个在 Mamba 中用双曲的工作，除非进一步调研全部相关作品。**

**层次 3：曲率必要性。** G1 必须对照 G2：如果不同曲率下的测地距离无法在等径向条件下带来更优变化检测，就不以“负曲率是关键”写主张。单次四库 seed16 的比较只支持此协议下的观察，若未来期刊严审统计可信度，可在不违背“不得多 seed 挑优”原则下，**预先设计固定复现实验或 bootstrap 不确定性分析**，而不是挑最优 seed 报告。

**与 CASA-CD 主叙事的连接**：现有“CAACP（压缩）+ STR（部署折叠）”旨在说明变化证据如何在紧凑低频扫描和可部署时相重建中被保存。若 CRH 成立，它增强的是**压缩阶段证据选择的质量**，而不是贡献一条孤立的“双曲几何模块”。若 CRH 没有增益，不应迫使它成为第三创新点；应保留负结果，优先强化已有 CAACP/STR 的因果消融。

---

## 12. 文献与代码速查索引（方便直接打开）

### 12.1 与变化检测直接相关

1. **HES-Net · JSTARS 2024**：<https://doi.org/10.1109/JSTARS.2024.3390427>；官方代码：**未核验**。
2. **HyperCLIP-CD · TGRS 2026**：<https://doi.org/10.1109/TGRS.2026.3670127>；官方代码：**未核验**。
3. **HSN / Deep Change Monitoring · CVPR 2025**：<https://openaccess.thecvf.com/content/CVPR2025/html/Li_Deep_Change_Monitoring_A_Hyperbolic_Representative_Learning_Framework_and_a_CVPR_2025_paper.html>；GitHub <https://github.com/liyantett/Tree-Changes-Detection-with-Siamese-Hyperbolic-network>。
4. **PID-CD · JSTARS 2025**：<https://doi.org/10.1109/JSTARS.2025.3556096>；官方代码：**未核验**。
5. **ChangeCLIP · ISPRS JPRS 2024**：<https://www.sciencedirect.com/science/article/pii/S0924271624000042>；GitHub <https://github.com/dyzy41/ChangeCLIP>。

### 12.2 遥感中“流形用于异常/局部结构”的相邻工作

6. **HSAE · TGRS 2024**：<https://doi.org/10.1109/TGRS.2024.3419075>；官方代码：**未核验**。
7. **Manifold-Geometric Regularized GCN · TGRS 2025**：<https://doi.org/10.1109/TGRS.2025.3584561>；官方代码：**未核验**。

### 12.3 CVPR/ICCV 双曲几何学习、稠密预测及反例

8. **HypAD · CVPR 2024**：<https://openaccess.thecvf.com/content/CVPR2024/html/Li_Hyperbolic_Anomaly_Detection_CVPR_2024_paper.html>；官方代码：**未核验**。
9. **Flattening the Parent Bias · CVPR 2024**：<https://openaccess.thecvf.com/content/CVPR2024/html/Weber_Flattening_the_Parent_Bias_Hierarchical_Semantic_Segmentation_in_the_Poincare_CVPR_2024_paper.html>；GitHub <https://github.com/tum-vision/hierahyp>。
10. **Hi-Mapper · CVPR 2024**：<https://openaccess.thecvf.com/content/CVPR2024/html/Kwon_Improving_Visual_Recognition_with_Hyperbolical_Visual_Hierarchy_Mapping_CVPR_2024_paper.html>；GitHub <https://github.com/kwonjunn01/Hi-Mapper>（代码完整性待核）。
11. **HypCD · CVPR 2025**：<https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.html>；GitHub <https://github.com/Visual-AI/HypCD>。
12. **HyperCLIP（语义分割）· CVPR 2025**：<https://openaccess.thecvf.com/content/CVPR2025/html/Peng_Understanding_Fine-tuning_CLIP_for_Open-vocabulary_Semantic_Segmentation_in_Hyperbolic_Space_CVPR_2025_paper.html>；GitHub <https://github.com/SJTU-DeepVisionLab/HyperCLIP>。**与 TGRS HyperCLIP-CD 是不同论文**。
13. **Learning Visual Hierarchies · ICCV 2025**：<https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Learning_Visual_Hierarchies_in_Hyperbolic_Space_for_Image_Retrieval_ICCV_2025_paper.html>；官方代码：**未核验**。
14. **Accept the Modality Gap · CVPR 2024**：<https://openaccess.thecvf.com/content/CVPR2024/html/Ramasinghe_Accept_the_Modality_Gap_An_Exploration_in_the_Hyperbolic_Space_CVPR_2024_paper.html>；官方代码：**未核验**。
15. **Hyperbolic Learning with Synthetic Captions · CVPR 2024**：<https://openaccess.thecvf.com/content/CVPR2024/html/Kong_Hyperbolic_Learning_with_Synthetic_Captions_for_Open-World_Detection_CVPR_2024_paper.html>；官方代码：**未核验**。

### 12.4 CASA-CD 的结构压缩与骨干先例

16. **SAT · CVPR 2026 Findings / arXiv**：<https://arxiv.org/abs/2604.07994>；GitHub <https://github.com/PhuTran1005/SAT>。
17. **TinyViM · ICCV 2025**：GitHub <https://github.com/xwmaxwma/TinyViM>；论文入口请从官方仓库核对。
18. **CASA-CD 本仓库（本次审查基准）**：<https://github.com/YuqiWang-code/CASA-CD/tree/23eb250cd48bb012ffaafc908dd1daafa560aec2>。

---

## 13. 立即执行顺序与仍需补充证据

### 13.1 立即执行顺序（按研究价值排序）

1. **锁代码身份**：将本次 main SHA `23eb250c...`、当前 E9 的 train/eval sidecar、Run7 正式 TEST 最终区块归档；确认实际服务器代码是否在此提交之后又有变化。
2. **先做不训练的数学负对照**：cosine → fixed-radius hyperbolic → 原 rank，要求 rank 与 pool 结果等价。此关不通过即停查代码。
3. **在同一 Stage3 前缀实现 `crh_rank` 与 `radial_euclidean_rank`**，分数函数无参数，既有 cell 权重公式完全不变。**首轮正式协议锁定方案 II**。
4. **新增并通过 T0–T12 smoke**；真实四库 DataLoader 3-step dry-run；测真实 scorer overhead 与部署 Params。
5. **重测 E9 β 的局部利用**，对比旧 M1 的 0.19% 历史值，确定本轮机制是否可能产生有效变化。
6. **预注册 G0/G1/G2**：同步代码、loss、seed16、80K、best 选择协议，不从旧 Run7 直接拿指标充当同期 G0。
7. **训练 G0 与 G1 四库**，完成最后完整 TEST 后，再按曲率辨识需要训练 G2；不要根据某一次 test 行随意改半径。
8. **以 HES-Net、HyperCLIP-CD、HSN 为 Related Work 主对照**，论文中明确先例和自己聚合位置/约束/等秩反例差异；成功才考虑拓展为 TGRS 主张。

### 13.2 仍需补充证据（当前缺失）

- **[M1] RSML-3 的真实最新 checkpoint、完整 `train_log.txt`**：本次不能直接复算 Run7 或 H1 的正式结果；不要把本报告当作训练结论。
- **[M2] HyperCLIP-CD 出版商全文**：核实 HMSA 数学实现、是否有无参数相似性/局部池化设计及开源状态；其 DOI/摘要可核验，但全文细节尚不完整。
- **[M3] HES-Net 的完整可复制结构与 Params/FLOPs**：若需与 CASA 的实测模型做公平数值对照，应重新确认输入分辨率、数据划分、训练预算和评测阈值，不能直接拿它论文 F1 当成同协议结果。
- **[M4] HFAE_E9 最新 β、cell 变更量与小目标对象级表现**：Diag1 M1 的历史诊断不能自动外推。
- **[M5] 双曲半径是否真的捕捉到稳定的局部结构层次**：目前只是手工定义的结构离散度；需要变化/未变化、边界/纹理、大小目标分组证据。
- **[M6] 曲率必要性的实验**：缺少 G1 对 G2 的同期完整 80K 对比，此前不能声称负曲率本身改善 F1。
- **[M7] 实际 deploy latency / selective_scan/FLOPs 覆盖**：常规 FLOPs 统计对超越逐点乘加的算子可能漏计，不宜预先标称“零计算代价”。

---

## 最后裁决

**建议立项做一轮严格的“几何机制验证”，但暂不把双曲空间确认为 CASA-CD 第三个创新点。** 目前最有价值的研究路径不是“把余弦换成 Poincaré”，而是围绕**非恒定径向、局部上下文角色变化、规则 scan 前压缩**，给出一个零新增参数的可证伪设计；并以具有**相同局部结构信息的欧氏径向对照**证明双曲曲率是否真的必要。若无法超越欧氏对照，保留低成本欧氏机制与诚实负结果，比在 TGRS 中提出无法被证据支持的“双曲优越性”更有科研价值。

**本文档是调研与预注册草案，不是已实现、已训练或已发表的方法。**
