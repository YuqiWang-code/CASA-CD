# models/ — 主线代码（当前 Run9）

本目录**只放主线任务的代码**。探索性 / 额外任务在 `../models_v1/`，通用分析工具在 `../tools/`。

## 入口

| 文件 | 用途 |
|---|---|
| `train.py` | 训练入口。`--arch casa_tvim_str`（Run1–Run7）/ `casa_vmamba_str`（Run8 v1）/ `casa_vmamba_str_v2`（**当前主线**） |
| `eval.py` | 独立测试入口（读 `arch.json` 侧车做 fail-closed 结构校验） |
| `main.py` | 上游 ChangeViT 原版入口，仅作参考，未改动 |
| `smoke_test.py` | 冒烟测试（`T-CA-1..10`，无需真实数据） |

## 当前主线的回归门

| 文件 | 覆盖 |
|---|---|
| `test_hsam_cpu.py` | **Run9 回归门 32 项**：Poincaré 映射 / 双曲距离闭式一致 / 论文原式必然均匀（固定断言）/ **非退化证书 `rank_corr(HD,1−cos)`** / 零参数 / fail-closed |
| `test_hqkv_lattice_cpu.py` | Run8c 回归门 27 项：无几何、无半径分解、反自锁、参数量、fail-closed |
| `test_str_reparam_equivalence.py` | TAR/DCR 折叠原语等价性 |

## 当前主线的只读探针（服务器上跑）

| 文件 | 用途 |
|---|---|
| `run9_smoke.py` | 真实数据 smoke：`hsam=0/1` 对比（读数 / 吞吐 / 显存 / 预训练加载） |
| `run9_degeneracy_cert.py` | 非退化证书：`pre_norm` vs `post_norm` vs 加全局 RMS 的 `rank_corr` 对照 |
| `run9_radius_probe.py` | 半径诊断：`outnorm` 之前/之后的范数、半径、`1−‖p‖²` 分布 |

## 网络结构

| 路径 | 说明 |
|---|---|
| `model/vmamba_tiny.py` | VMamba-Tiny 自包含主干（`vssm_tiny_0230` 口径，官方权重逐位加载 `worst_diff=0`）。`forward_features()` 给**归一化后**的 stage 输出，`forward_pre_norm()` 给 **`outnorm` 之前**的激活（几何模块必须用后者） |
| `model/casa_vmamba_net_v2.py` | **当前主线网络**：`CAACP-HQKV(lattice)` + **HSAM**（在 `outnorm` 之前混合）+ TAR/DCR |
| `model/casa_vmamba_net.py` | Run8 v1 网络（历史复现） |
| `model/casa_tvim_str_net.py` | Run1–Run7 主架构（TinyViM-S-Slim + CAACP-SS2D） |
| `model/layers/hsam.py` | **Run9 核心**：Poincaré 球双曲相似度注意力，**零可学习参数** |
| `model/layers/caacp_hqkv_lattice.py` | Run8c：固定格点压缩的跨层 SAT-QKV（无几何） |
| `model/layers/caacp_hqkv.py` / `hyperbolic.py` | Run8 v1（含 `γ` 自锁缺陷，仅历史复现） |
| `model/str_tar.py` / `str_dcr.py` / `str_reparam.py` | TAR / DCR / 折叠原语 |
| `model/layers/vmamba_kernel.py` | selective-scan 后端适配（绕开 `casacd` 里 ABI 已坏的 `.so`） |

## 两条必须记住的约束

1. **几何模块必须取 `outnorm` 之前的激活**。`outnorm{i}=LayerNorm2d(dim)` 让每个 token 的
   范数**精确等于 `√C`**，Poincaré 半径因此恒定，双曲距离退化成余弦距离的单调变换
   （实测秩相关 **1.000000**）。现行硬门：`rank_corr(HD,1−cos) < 0.999`。
2. **`models/` 下的模型/训练文件在代码身份链里**（`train.py::SOURCE_IDENTITY_FILES`，
   哈希写进每个 run 的 `arch.json` / `run_manifest.json`）。**campaign 跑动期间不要改**，
   否则同一轮不同 run 的侧车会记录不同代码。

## `_legacy/`

past-run 的一次性脚本（Run4/Run7/Run8 的 dry-run 与预检），仅供备查，**不属当前主线**。
它们依赖的路径已随目录调整变化，直接跑可能需要改 `sys.path`。

## 上游与许可

基于 [ChangeViT](https://github.com/zhuduowang/ChangeViT)（Pattern Recognition 2025）改造；
其许可证文本保留在 `../docs/参考文献/baseline/ChangeViT_LICENSE.txt`。
