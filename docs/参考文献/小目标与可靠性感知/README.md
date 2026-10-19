# 小目标与可靠性感知 · 文献核验记录

> **核验日期**：2026-10-19

本目录是为「小目标变化检测 / 可靠性感知（伪变化、噪声、结构证据、不确定性）」这条线
单独开的核验批次。**六条来自一份第三方咨询文档，逐条回到官方源核实**，不采信二次转载。

## 核验方法（可复现）

| 来源 | 用途 |
|---|---|
| [IEEE Xplore](https://ieeexplore.ieee.org/) 正文页 | 期刊论文的题名 / 全部作者 / 卷期 / 在线日期 / DOI / 官方代码链接 |
| [CVF Open Access](https://openaccess.thecvf.com/) | CVPR 论文的题名 / 作者 / 页码 / 官方 PDF |
| [Crossref API](https://api.crossref.org/) | Elsevier 等期刊的 DOI、卷期、页码、作者（ScienceDirect 直连返回 403，改用 Crossref/OpenAlex 元数据） |
| [OpenAlex API](https://api.openalex.org/) | 交叉核对 Elsevier 论文的 venue / volume / pages / 作者 |
| [arXiv API](http://export.arxiv.org/api/query) | 判断是否存在预印本 |
| [Unpaywall API](https://api.unpaywall.org/) | 判定 `is_oa` / `oa_status`，确认 PDF 能否合法免费获取 |
| 官方 GitHub / HuggingFace | 确认官方代码仓库真实存在（只认论文正文或官方页给出的地址） |

**PDF 入库规则**：只从出版社官方（CVF Open Access）或 arXiv 下载；下载后校验文件头必须是
`%PDF-` 且大小 > 100 KB，否则删除并记为「PDF 不可得」。**不使用任何盗版/镜像源。**

---

## 一、核验总表

| 序号 | 自称题名 | 核验结果 | 实际 venue · 年份 | DOI / 官方链接 | 代码 | PDF 入库 |
|---|---|---|---|---|---|---|
| 1 | GCCANet: A Small-Object Sensitive and Environment-Robust Network for Remote Sensing Change Detection | **存在**，题名/venue/年份**全部准确** | IEEE TGRS, Vol. 64, 2026-08-06 | [10.1109/TGRS.2026.3720341](https://doi.org/10.1109/TGRS.2026.3720341) | ✅ [E1ison/GCCANet](https://github.com/E1ison/GCCANet) | ❌ 不可得 |
| 2 | HA2F: Dual-Module Collaboration-Guided Hierarchical Adaptive Aggregation Framework for Remote Sensing Change Detection | **存在**，题名准确；年份口径需注明（DOI 前缀 2025 / 正式卷期 2026） | IEEE TGRS, Vol. 64, 2026-01-05 | [10.1109/TGRS.2025.3650706](https://doi.org/10.1109/TGRS.2025.3650706) · [arXiv:2601.16573](https://arxiv.org/abs/2601.16573) | ✅ [HuggingFace 官方仓库](https://huggingface.co/InPeerReview/RemoteSensingChangeDetection-RSCD.HA2F) | ✅ `HA2F(TGRS2026,arXiv版).pdf` |
| 3 | EdgeRefNet: An Edge-Guided Refinement Network for Building Change Detection in Remote Sensing Images | **存在**，题名/venue/年份**全部准确** | IEEE TGRS, Vol. 64, 2026-04-30 | [10.1109/TGRS.2026.3689397](https://doi.org/10.1109/TGRS.2026.3689397) | ✅ [Wafaa-Hima/EdgeRefNet](https://github.com/Wafaa-Hima/EdgeRefNet) | ❌ 不可得 |
| 4 | Feature Spectrum Learning for Remote Sensing Change Detection（FeaSpect） | **存在**，题名/venue/年份**全部准确** | CVPR 2025, pp. 12647–12657 | [10.1109/CVPR52734.2025.01180](https://doi.org/10.1109/CVPR52734.2025.01180) · [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html) | ⚠️ 未找到官方仓库 | ✅ `FeaSpect(CVPR2025).pdf` |
| 5 | Small Object Change Detection in UAV Imagery via a Siamese Network Enhanced with Temporal Mutual Attention and Contextual Features（TMACNet） | **存在**，但**题名不符（被截断）** | ISPRS J. P&RS, Vol. 218, pp. 352–367, 2024-12 | [10.1016/j.isprsjprs.2024.09.027](https://doi.org/10.1016/j.isprsjprs.2024.09.027) | ⚠️ 未找到官方仓库 | ❌ 不可得 |
| 6 | UAFDcd: Uncertainty-Aware Feature Decoupling Network for Remote Sensing Change Detection | **存在**，题名/venue/年份**全部准确** | IEEE TGRS, Vol. 64, 2026-06-08 | [10.1109/TGRS.2026.3701174](https://doi.org/10.1109/TGRS.2026.3701174) | ⚠️ 未找到官方仓库 | ❌ 不可得 |

**统计**：6 篇**全部真实存在**（无编造文献）；5 篇题名/venue/年份与自称一致，1 篇（TMACNet）题名被截断。
可合法免费下载的 PDF 只有 2 篇（FeaSpect、HA2F），其余 4 篇均在付费墙后。

---

## 二、逐条核验详情

### 1. GCCANet（✅ 存在，信息全对）

- **确切题名**：`GCCANet: A Small-Object Sensitive and Environment-Robust Network for Remote Sensing Change Detection`
- **作者（5 人，全列）**：Jingqi Chen、Yonghong Song、Yifei Gao、You Su、Yong Yu
- **venue · 年份**：IEEE Transactions on Geoscience and Remote Sensing, **Vol. 64, 2026**；
  在线日期 2026-08-06；Article Sequence Number 4415313
- **DOI**：[10.1109/TGRS.2026.3720341](https://doi.org/10.1109/TGRS.2026.3720341)
- **官方页**：[IEEE Xplore 11643269](https://ieeexplore.ieee.org/abstract/document/11643269/)
- **官方代码**：[https://github.com/E1ison/GCCANet](https://github.com/E1ison/GCCANet)
  （仓库真实存在，README 明确写 "Official implementation of GCCANet"，含 `train.py` / `models/`；
  预训练权重走百度网盘。注意只有 1 star，属新发布仓库。）
- **PDF**：❌ **不可得**。Unpaywall 判定 `is_oa=False, oa_status=closed`；arXiv 无该文预印本；
  直连 IEEE `iel8` PDF 返回 `<html>`（51,996 字节，非 PDF），按规则删除。
- **与本课题的关系**：它的"小目标"卖点落在**解码端**——lightweight decoder 上挂 MS-SOE
  （multiscale small-object enhancement），按多尺度强化细弱变化、保住细长结构；可靠性那一半来自
  BSM（background suppression module）把 ALA 压出的**全局 Transformer 引导注入每个 CNN stage**，
  在差分之前先压背景噪声。**注意：它是"全局引导抑制背景"，不是"结构证据筛选"**——没有显式的
  edge/structure 分支，做我们 CASA-CD 的结构证据论述时只能当"尺度选择性 + 伪变化抑制"的旁证，
  不能当结构证据的正面证据。

### 2. HA2F（✅ 存在，年份口径需注明）

- **确切题名**：`HA2F: Dual-Module Collaboration-Guided Hierarchical Adaptive Aggregation Framework for Remote Sensing Change Detection`
  （IEEE 页 `Dual-Module`，arXiv 版 `Dual-module`，仅大小写差异）
- **作者（4 人）**：Shuying Li、Yuchen Wang、San Zhang、Chuang Yang
- **venue · 年份**：IEEE TGRS, **Vol. 64, 2026**；在线日期 2026-01-05；Article Sequence Number 5602313
  ⚠️ **但 DOI 是 `10.1109/TGRS.2025.3650706`——编号在 2025 年分配**（early access 2025，正式卷期 2026）。
  写 "TGRS 2026" 可接受；但 BibTeX 里 `year` 与 DOI 前缀年份不一致时，**建议按正式卷期写 2026 并在
  条目里注明 early access**，否则查重/引用校验工具会报 year 与 DOI 矛盾。
- **DOI**：[10.1109/TGRS.2025.3650706](https://doi.org/10.1109/TGRS.2025.3650706)
- **官方页**：[IEEE Xplore 11328896](https://ieeexplore.ieee.org/document/11328896/)
- **官方代码**：[https://huggingface.co/InPeerReview/RemoteSensingChangeDetection-RSCD.HA2F](https://huggingface.co/InPeerReview/RemoteSensingChangeDetection-RSCD.HA2F)
  （IEEE 正文与 arXiv 版均给出该地址，HTTP 200 可访问。属 HuggingFace repo 而非 GitHub；
  命名带 `InPeerReview`，说明是投稿期建的仓库，**代码成熟度需自行评估**。）
- **PDF**：✅ **已入库** `HA2F(TGRS2026,arXiv版).pdf`（13 页，14.9 MB，`%PDF-1.7`，
  来源 [arXiv:2601.16573](https://arxiv.org/abs/2601.16573)，2026-01-23 提交）。
  ⚠️ 入库的是**预印本**，与 IEEE 正式版可能有排版/内容差异；**官方发表版本未下载**。
- **与本课题的关系**：它把"可靠性"做在**特征聚合层**——DHFCM（dynamic hierarchical feature
  calibration）用感知特征选择动态融合相邻层特征，专门压制跨时相**特征匹配偏差**；NAFRM
  （noise-adaptive feature refinement）用双特征选择生成空间 mask，把无关区域和**阴影**挡掉。
  这正好是"伪变化由谁引起"的一个具体答案（配准偏差 + 光照/阴影），但**不涉及尺度选择，也不涉及结构证据**。

### 3. EdgeRefNet（✅ 存在，信息全对）

- **确切题名**：`EdgeRefNet: An Edge-Guided Refinement Network for Building Change Detection in Remote Sensing Images`
- **作者（5 人）**：Wafaa I. M. Hussin、Zhi Lu、Aysha Ashraf、Aji Mao、Zhenming Peng
- **venue · 年份**：IEEE TGRS, **Vol. 64, 2026**；在线日期 2026-04-30；Article Sequence Number 4409610
- **DOI**：[10.1109/TGRS.2026.3689397](https://doi.org/10.1109/TGRS.2026.3689397)
- **官方页**：[IEEE Xplore 11501963](https://ieeexplore.ieee.org/document/11501963/)
- **官方代码**：[https://github.com/Wafaa-Hima/EdgeRefNet](https://github.com/Wafaa-Hima/EdgeRefNet)
  （真实存在，README 给出完整训练/测试脚本与 `--net_G EdgeRefNet` 入口，基于 EGENet 代码库改；
  另有一个 0 star 的镜像 `wafohima/EdgeRefNet`。）
  ⚠️ **该 repo 的 README 写的题名与 IEEE 正式题名不一致**：README 是
  "EdgeRefNet: Edge-Guided **Enhancement and** Refinement Network for Building Change Detection **of** Remote Sensing Images"，
  而 IEEE 正式题名是 "An Edge-Guided Refinement Network ... **in** Remote Sensing Images"。
  **以 IEEE 为准**；引用时不要抄 README。
- **PDF**：❌ **不可得**。Unpaywall `is_oa=False, oa_status=closed`；arXiv 无预印本；
  IEEE `iel8` PDF 直链返回 `<html>`（55,435 字节），已删除。
- **与本课题的关系**：这是六篇里**唯一真正做"结构证据"的**——双路设计（context path 管语义 +
  edge path 管细粒度结构），并用 **context-guided cross-attention 让语义上下文反过来精炼边缘特征**
  （即"用语义判断这条边缘可不可信"），加上 EDB/EEM 从浅层抽取并强化边缘、change 与 edge **联合监督**。
  对我们的"结构证据可靠性"论述是**最贴的直接正面证据**。
  但要写清边界：① 只在 LEVIR-CD / WHU-CD 两个**建筑物**数据集上验证；② 是 building CD
  （建筑专用、边界精度导向），**不是小目标**，也不是通用二值 CD 的尺度选择。

### 4. FeaSpect（✅ 存在，信息全对）

- **确切题名**：`Feature Spectrum Learning for Remote Sensing Change Detection`
- **作者（5 人）**：Qi Zang、Dong Zhao、Shuang Wang、Dou Quan、Zhun Zhong
  （西电 + 合工大；通讯 shwang@mail.xidian.edu.cn）
- **venue · 年份**：**CVPR 2025**，Proceedings pp. **12647–12657**
- **DOI**：[10.1109/CVPR52734.2025.01180](https://doi.org/10.1109/CVPR52734.2025.01180)
- **官方页（CVF）**：[CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html)
  （另有 poster 页 <https://cvpr.thecvf.com/virtual/2025/poster/33143>、IEEE Xplore doc 11095146）
- **官方代码**：⚠️ **未找到**。全文文本检索无 `github` 字样；GitHub 仓库搜索 "FeaSpect"
  只命中一个无关 iOS 项目。**论文未给出官方 repo，记为"未核实/无官方代码"**。
- **PDF**：✅ **已入库** `FeaSpect(CVPR2025).pdf`（11 页，4.19 MB，`%PDF-1.5`，CVF 官方开放获取版）。
- **与本课题的关系**：它处理的是**伪变化的成因分类里"风格/辐射漂移"那一支**——不用 GAN 做像素级
  风格迁移，而是在**潜空间**做 feature spectrum transformation（FST），只让"extraction box"里
  带 learnable stride + adaptive attention 的频带变换可训练，把变换引向"风格对齐的判别性特征"，
  从而抑制伪变化。**这是"可靠性感知"的强证据，但和小目标、尺度选择性基本无关**——
  把它挂到"小目标"名下属于过度解读（见"发现的问题"）。

### 5. TMACNet（⚠️ 存在，但**题名不符**）

- **自称题名（第三方文档）**：`Small Object Change Detection in UAV Imagery via a Siamese Network Enhanced with Temporal Mutual Attention and Contextual Features`
- **实际**：官方题名带副标题，**被截掉了后半截**：
  `Small object change detection in UAV imagery via a Siamese network enhanced with temporal mutual attention and contextual features: **A case study concerning solar water heaters**`
- **作者（5 人）**：Shikang Tao、Mengyuan Yang、Min Wang、Rui Yang、Qian Shen
- **venue · 年份**：**ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 218, pp. 352–367, 2024-12**
  （Crossref / OpenAlex 双源一致；Bibcode `2024JPRS..218..352T`）
- **DOI**：[10.1016/j.isprsjprs.2024.09.027](https://doi.org/10.1016/j.isprsjprs.2024.09.027)
- **官方页**：[ScienceDirect S0924271624003654](https://www.sciencedirect.com/science/article/pii/S0924271624003654)
  （直连 403，元数据经 Crossref/OpenAlex 核实）
- **"TMACNet" 这个名字是真的**：ScienceDirect 摘要片段原文 "This study proposes an SOCD model
  called TMACNet"，即论文自身用这个缩写，**不是第三方瞎起的名字**。
- **官方代码**：⚠️ **未找到**。arXiv 无预印本、GitHub 搜索无对应仓库、Crossref 元数据无 code 链接。
- **PDF**：❌ **不可得**。Unpaywall `is_oa=False, oa_status=closed`，Elsevier 订阅墙，无 arXiv/OA 副本。
- **与本课题的关系**：**六篇里唯一直题"小目标"的**——Siamese 双分支 + 时序互注意力（temporal
  mutual attention）+ 上下文特征增强，任务设定是 UAV 影像里**屋顶太阳能热水器**这类极小目标的变化检测
  （作者自称 SOCD，small object change detection）。可直接支撑"小目标在双时相里纹理弱、易被背景吞掉，
  需要跨时相互注意力而非单纯差分"这一条。
  **但必须写清两条边界**：① 它是**单类场景案例研究**（solar water heaters），不是通用 CD 基准，
  指标不可与 LEVIR/SYSU 横向比较；② 它服务的是**无人机高分辨率、目标尺寸极小**的设定，
  与我们 256×256、骨干下采样到 1/32 的设定不同构，**不能直接拿它的数值当 SOTA 论据**。

### 6. UAFDcd（✅ 存在，信息全对）

- **确切题名**：`UAFDcd: Uncertainty-Aware Feature Decoupling Network for Remote Sensing Change Detection`
  （⚠️ 摘要正文里自述为 "an uncertainty-aware **cross-domain** feature decoupling network"，比标题多了
  一个 cross-domain；引用时**以标题为准**）
- **作者（5 人）**：Bin Wang、Sichao Qiao、Ziyv Zhao、Pinle Qin、Jianchao Zeng
- **venue · 年份**：IEEE TGRS, **Vol. 64, 2026**；在线日期 2026-06-08；Article Sequence Number 5625815
- **DOI**：[10.1109/TGRS.2026.3701174](https://doi.org/10.1109/TGRS.2026.3701174)
- **官方页**：[IEEE Xplore 11554085](https://ieeexplore.ieee.org/abstract/document/11554085/)
- **官方代码**：⚠️ **未找到**。IEEE 摘要页未给代码链接；GitHub 仓库搜索 "UAFDcd" 返回 **0 结果**；
  arXiv 无预印本。**若第三方文档声称它有代码，属未核实。**
- **PDF**：❌ **不可得**。Unpaywall `is_oa=False, oa_status=closed`；IEEE `iel8` PDF 直链返回
  `<html>`（54,145 字节），已删除。
- **与本课题的关系**：**六篇里"可靠性"做得最彻底的一篇**——用因果推断把双时相特征解耦成
  共享/独有部分，切断"不变特征 → CD 输出"的**后门路径**；再用解耦原型对比学习拉近正原型、
  推远负原型；最后用 **Dirichlet 证据分布**做不确定性估计（同时输出变化/不变概率与
  epistemic uncertainty，用 evidential loss 压认知不确定性）。对我们"置信度/可靠性加权"的设计
  是直接可引的方法论。**但与小目标、尺度选择无关**，也没有结构证据机制。

---

## 三、发现的问题（第三方文档的错 / 夸大 / 无法核验处）

1. **【明确的错】TMACNet 题名被截断。** 漏掉了副标题 `: A case study concerning solar water heaters`。
   这个副标题不是可有可无的装饰——它说明该文是**太阳能热水器单类场景的案例研究**，不是通用 CD 方法。
   按截断后的题名去引用，会让人误以为它是在通用基准上比的方法。

2. **【夸大/过度解读】把 FeaSpect 归到"小目标"名下。** FeaSpect 通篇讲的是
   bitemporal pseudo-change（风格/辐射漂移）的潜空间频谱抑制，**没有一个模块是为小目标设计的**，
   实验也不含小目标专项指标。它能支撑的是"**可靠性感知 / 伪变化抑制**"，不是"小目标"。
   同理，UAFDcd、HA2F 也不含小目标设计——**六篇里真正做小目标的只有 TMACNet 和 GCCANet（MS-SOE）两篇**。

3. **【口径模糊，易出错】HA2F 的年份。** 自称 "TGRS 2026" 不算错（正式卷期 Vol. 64 / 2026，
   在线 2026-01-05），但 **DOI 是 `10.1109/TGRS.2025.3650706`，编号 2025 年分配**。
   第三方文档只写 "2026"，没提这个 2025/2026 双年份，写 BibTeX 时极易出现 `year=2026` 与
   DOI 前缀 2025 互相矛盾，被查重/引用工具判为错引。

4. **【题名不一致，抄错源】EdgeRefNet 的 GitHub README 题名与 IEEE 正式题名不同。**
   README 写 "Edge-Guided **Enhancement and** Refinement Network ... **of** Remote Sensing Images"，
   IEEE 写 "**An** Edge-Guided Refinement Network ... **in** Remote Sensing Images"。
   幸好第三方文档抄的是 IEEE 版本（正确）；但**若有人照 README 补参考文献就会写错**。

5. **【无法核验】代码归属。** 只有 3 篇有可信官方代码：GCCANet（GitHub）、HA2F（HuggingFace）、
   EdgeRefNet（GitHub）。**FeaSpect（CVPR 2025 顶会）反而没有官方 repo**，TMACNet、UAFDcd
   也搜不到。第三方文档若声称这几篇"代码开源"，**均未核实（未在官方源找到）**。

6. **【无法核验】4 篇 PDF 在付费墙后。** GCCANet / EdgeRefNet / UAFDcd（IEEE）与 TMACNet（Elsevier）
   的 `oa_status` 均为 `closed`，无 arXiv 版本。本目录**只有 2 个 PDF**，写文献综述时这 4 篇
   只能引摘要级信息（模块名、数据集、结论），**不能引用具体数值/公式细节**——若要引用数值，
   需通过学校图书馆 IEEE/Elsevier 权限另行获取。

7. **【正面结论】没有编造文献。** 6 条全部在官方源找到实体记录，DOI 全部真实可解析，
   未发现"幽灵论文"或张冠李戴的 venue。第三方文档的主要问题是**题名截断 + 主题归类夸大**，
   不是伪造。

---

## 四、本目录文件清单

| 文件 | 大小 | 来源 | 说明 |
|---|---|---|---|
| `FeaSpect(CVPR2025).pdf` | 4.19 MB / 11 页 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.pdf) | 官方开放获取版，与正式版仅差水印 |
| `HA2F(TGRS2026,arXiv版).pdf` | 14.92 MB / 13 页 | [arXiv:2601.16573](https://arxiv.org/pdf/2601.16573v1) | **预印本**；IEEE 正式版未下载（付费墙） |

**未入库（PDF 不可得，原因见上）**：GCCANet、EdgeRefNet、TMACNet、UAFDcd。

---

## 五、给后续写作的三条提醒

1. **"小目标"这条线只能引 TMACNet + GCCANet(MS-SOE)**，且 TMACNet 是单类案例、
   GCCANet 的小目标增强只在解码端。**不要**把 FeaSpect / UAFDcd / HA2F 写成小目标工作。
2. **"结构证据可靠性"这条线只有 EdgeRefNet 是正面证据**（语义引导的边缘精炼 + change/edge 联合监督），
   且它只管建筑物边界，**没有小目标维度**——正好是我们"结构证据 + 小目标"交叉点上的空缺，
   可作为创新空白的一个引用支点。
3. **"可靠性感知"这条线材料最厚**：FeaSpect（潜空间频谱去伪变化）、HA2F（配准偏差 + 阴影抑制）、
   UAFDcd（因果解耦 + Dirichlet 不确定性）。三篇机制各不相同，可并列综述，
   但注意三者都**没有**与尺度/小目标耦合——这也是一个可写的空白。
