# LaViT

- **论文**: *You Only Need Less Attention at Each Stage in Vision Transformers* (LaViT), CVPR 2024.
- **状态**: **失败 — 未找到官方代码仓库**。作者（Shuoxi Zhang, Hanpeng Liu, Stephen Lin, Kun He）未公开官方 GitHub 实现：CVPR 开放论文页与 arXiv:2406.00427 均无代码链接；GitHub 搜索（repo/关键词/作者账号多种组合）未找到对应仓库（同名 LaVIT 仓库均属其它论文，如 jy0205/LaVIT 是 LLM 视觉 tokenizer 论文）。
- **本目录内容**: 仅此说明文件，无源码。
- **可参考信息**: 论文核心机制为 LaViT 的 stage-wise 少注意力设计——每个 stage 只计算少量 attention，后续层通过"注意力变换（attention transformation）"复用先前注意力分数完成特征对齐，可参考论文公式自行实现。
- **上游许可证**: 不适用（无官方仓库）。
- **检索日期**: 2026-10-01。
