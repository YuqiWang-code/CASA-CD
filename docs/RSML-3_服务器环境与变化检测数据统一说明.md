# RSML-3 服务器环境与变化检测数据统一说明

> 用途：该服务器上所有变化检测（CD）项目共享的环境与数据参考。写 DataLoader、配路径、加载预训练权重前先读本文件。
> 更新：2026-09-13（数据集与 Cache 已迁移到 `/share_datasets/`）。

## 1. 服务器环境

- 主机 `RSML-3`，Ubuntu 22.04，用户 `yqwang`。
- CPU：Threadripper 7960X（24 核 48 线程）；内存 64 GB；系统盘 4 TB NVMe。
- GPU：**2 × RTX 5090**（Blackwell，`sm_120`，每卡 32 GB），PCIe 无 NVLink，多卡用 DDP。
- CUDA：Driver 595.80，系统 Toolkit 13.2。

### Conda 环境

| 环境 | 定位 | 关键版本 |
|---|---|---|
| `tools` | 登录/监控工具 | Python 3.11；`nvitop` 在 `~/.local/bin/nvitop` |
| `cd_base` | 通用 CV / CNN / Transformer 模板 | Torch 2.14+cu132 |
| `mamba_base` | Mamba / SSM 模板（已冻结） | Torch 2.10+cu130，mamba-ssm 2.3.2 |
| `casacd` | **本项目**（clone 自 `cd_base`） | Torch 2.14+cu132；补 `einops opencv-python fvcore` |

- 论文项目应从模板 `conda create -n <env> --clone cd_base|mamba_base`，不要污染 base。

### 工作流

本地 Cursor 写代码 → SFTP（`.vscode/sftp.json`，手动上传）→ 服务器 GPU 训练/推理。大数据集与 checkpoint 不随代码反复上传下载。

## 2. 目录约定（CASA-CD）

| 用途 | 路径 |
|---|---|
| 项目代码 | `/home/yqwang/projects/CASA-CD` |
| 数据集 | `/share_datasets/CD` |
| Checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD` |
| 日志 / 实验输出 | `/home/yqwang/outputs/CASA-CD` |
| 预训练权重 | `/home/yqwang/projects/CASA-CD/pretrained_weight` |

当前预训练权重：
`/home/yqwang/projects/CASA-CD/pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth`（DeiT-Tiny，ChangeViT-T 用）。

## 3. 四个数据集

| 数据集 | 总样本 | Train / Val / Test | 图像 / Label | Train 变化像素比 |
|---|---:|---:|---|---:|
| CDD-CD-256 | 15,998 | 10,000 / 2,998 / 3,000 | JPG / JPG(L) | 11.9% |
| LEVIR-CD-256 | 10,192 | 7,120 / 1,024 / 2,048 | PNG / PNG(L) | 4.1% |
| SYSU-CD-256 | 20,000 | 12,000 / 4,000 / 4,000 | PNG / PNG(L) | 21.1% |
| WHU-CD-256 | 7,434 | 5,947 / 743 / 744 | PNG / PNG(L) | 3.4% |

- 四个 split 均已审计：完整、互斥、无缺失。
- 实验角色：CDD 抗伪变化、LEVIR/WHU 建筑小目标且类别极不平衡、SYSU 通用地表变化且变化像素最多。

## 4. DataLoader Contract（必读）

- 结构：`<dataset>/{A,B,label}/` + `<dataset>/list/{train,val,test}.txt`；A=T1、B=T2、label=二值 mask。
- **用 `list/*.txt` 里的 sample name（含扩展名）构造 A/B/label 路径**，禁止 `os.listdir` 自然顺序。
- Label 统一：`mask = (gray >= 128).astype(...)`——CDD 的 label 是 JPEG，有中间灰度，必须阈值化，不要 `==255`。
- A/B/label 必须共享同一几何增强；颜色/归一化不作用于 label。
- 归一化：有预训练 backbone 时用其官方 normalization（ChangeViT 用 ImageNet 统计 ×2，BGR 顺序），否则才用本地 RGB 统计。
- LEVIR 根目录另有 `list_*.txt`（替代 split），默认用 `list/` 下那套。WHU 有 5%~100% 半监督 split。

## 5. 其它项目资产（CASA-CD 不使用）

- `/share_datasets/CD_teacher_cache/`（OVCDistill / SAMStruct teacher cache）属于其它蒸馏类项目，CASA-CD 不读取、不修改。

## 6. 权威来源优先级

冲突时按：服务器实际文件 > `manifest.json` > `list/*.txt` > 本文件 > 项目代码/README > 原始数据集官网/论文。
