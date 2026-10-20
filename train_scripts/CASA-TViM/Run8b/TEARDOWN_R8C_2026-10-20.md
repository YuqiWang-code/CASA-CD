# Run8c：拆除双曲几何（2026-10-20）

> 用户指令（2026-10-20）：**停训**（理由：现在的设计没什么用）→ **把双曲几何结构拆除**
> → 拆除范围选 **B**（拆掉「流形映射 + 半径分解」，**保留**跨层 QKV 注入）
> → 拆完之后 **只做清理 + 归档，先不训练，等新方案**。

本文件记录这次拆除做了什么、凭什么、怎么验证的。

## 0. 一句话

把 `CAACP-HQKV` 里「Lorentz 双曲映射 → 半径分解 → detail/global 双分支」整条链删掉，
只留下**跨层 SAT 式 QKV 注入**（Q 全 `N`、K/V 用**固定格点**压到 `N/w²`），
并把反自锁的工程手段原样保留。模块参数 **0.62 M → 0.198 M**。

## 1. 停训（12:00:24 彻底停下）

| 项 | 结果 |
|---|---|
| 督导 `_r8b_pipeline_supervisor_single.sh` | TERM，剩余 0 |
| `_r8b_run_one_ds.sh` | TERM，剩余 0 |
| `python train.py` | TERM，剩余 0 |
| 复活检查 | `t+5/10/15/20 s` 三项计数**全为 0** |
| GPU0 | **166 MiB / 0%**（我们从 29 GB 让出） |
| GPU1 | 同门 `bqwang` 的 `673363/673364`，**全程未触碰** |
| 产物 | **一份未删**（只新增文件） |

**踩到的坑（值得写进纪律）**：11:48 第一次只杀了 `python train.py` 和督导，
结果 11:50 训练**自己回来了**（新 pid 720375，同数据集 CDD、user=yqwang、重新吃掉 GPU0）。
原因：`_r8b_run_one_ds.sh` 里有一个 **10 s 重试循环**，它是重试的宿主，父进程不死就会重启子进程。
⇒ 正确顺序：**先杀重试宿主 `run_one_ds` → 再杀 `train.py` → 再观察 20 s**。
`_r8b_stop.sh` 已按这个顺序重写，并把 20 s 复活检查做成脚本的一部分。

> 同一坑在探索线也出现过（他们的 commit `420a42f`：第一次暂停被 `run_one.sh` 的 10 s 重试循环复活）。
> 这是本仓库的**跨线共性缺陷**，不是偶发。

## 2. 拆了什么

### 2.1 归档（字节可复核）

`_SUPERSEDED_hqkv_v2_2026-10-20/` 保存 6 个文件的原字节 + `MANIFEST.sha256`：

| 文件 | SHA256（前 16） | 与 `SOURCE_SHA256.txt` |
|---|---|---|
| `hyperbolic.py` | `e28f07751fa41cce` | ✅ |
| `caacp_hqkv_v2.py` | `b6f30bec7221ad2e` | ✅ |
| `casa_vmamba_net_v2.py` | `c6b3de16a45ac96e` | ✅ |
| `test_run8b_antilock_precheck.py` | `35bbee476ba9b779` | ✅ |
| `run8b_dry_run.py` | `9f4a66c554d8c7cd` | ✅ |
| `train.py`（拆除前，取自 `94f9e69`） | `536b149d42667f1b` | ✅ |

服务器侧实测：**归档核对 OK=6 BAD=0** 才允许删除现行文件
（`_r8c_teardown.sh` 是**先验归档、不完整就 exit 3** 的硬门）。

### 2.2 删除 / 改写

| 动作 | 文件 |
|---|---|
| **删除** | `models/model/layers/caacp_hqkv_v2.py`、`models/test_run8b_antilock_precheck.py`、`models/run8b_dry_run.py` |
| **新增** | `models/model/layers/caacp_hqkv_lattice.py`、`models/test_hqkv_lattice_cpu.py` |
| **重写** | `models/model/casa_vmamba_net_v2.py`（改为 lattice；几何值 fail-closed） |
| **改动** | `models/train.py`（协议串、`--hqkv_geometry` choices、v2 分支 fail-closed、侧车字段、DIAG 打印、`--hqkv_window`） |
| **不动** | `models/model/layers/hyperbolic.py`、`models/model/layers/caacp_hqkv.py`、`models/model/casa_vmamba_net.py`（**v1 / Run8 血统仍可复现**） |

## 3. 拆完剩下什么（lattice 版）

```
T1/T2 各 stage 特征
  → Q:  T1 的 1×1 投影，保持全分辨率 N
  → KV: T2 经**固定格点均值池化**（window w=2）压到 n = N/w²  ← 无参数、无 TopK/聚类/随机
  → attn = softmax(Q·Kᵀ / √rank)        形状 (B, N, n)
  → out_i = RMS-normalize(proj(attn·V))  ⇒ 每支 RMS = 1
  → out = (1/√项数) · Σ_i γ_i · out_i    γ 每支独立、初值 0.5
  → 注入 TAR 输出的 feats[0]（DCR 1/4 输入）
```

- **不再有**：流形/双曲/欧氏几何项、变化分数、半径 `r0`/`a`、detail/global 双窗口与双分支、
  HGSL 的分数输入口；
- **仍然有**（反自锁资产，与几何无关）：每支独立 γ、`rms_normalize`、非零初值、
  `contrib_norm` 诊断口、step-1000 fail-fast；
- **参数量**：`198,340`（0.198 M），拆除前三支为 ≈0.62 M。

### 关键设计取舍（写下来免得以后又被改回去）

1. **压缩不再用「变化分数」加权**，改回**固定格点**。理由：分数来自几何，几何没了；
   而且这正是 README 给 CAACP 划的 novelty 边界（`structured context compression / 固定 cell lattice`），
   同时避免「分数没学好 ⇒ 压缩权重乱掉」这层间接失败模式。
2. **绝不能对最终 `out` 再做一次 `rms_normalize`**（Run8b 首版试过并撤回）：
   那会把 `γ` 的效果归一化掉、并截断 `∂L/∂γ`，等于把自锁换层皮。
   压量级只能用**固定结构常数** `1/√项数`。
3. **`hqkv=1 且 hgsl=1` 直接 `SystemExit`**，不静默跳过。几何拆了，HGSL 没有分数可喂；
   静默跳过会让人以为「加了 HGSL」而实际没加。

## 4. 验证（服务器实测，不是推断）

`bash train_scripts/CASA-TViM/Run8b/_r8c_verify.sh` ⇒ **回归门 27/27 PASS**（2026-10-20 12:02）：

| 组 | 项数 | 内容 |
|---|---:|---|
| A 拆除完整性 | 10 | 无 hyperbolic import、无 r0/radius 参数、`manifold/radius_decomposition/score_weighted_pooling` 全 `none`、`score_maps()` 恒空、form = `hqkv_v3_lattice` |
| B 反自锁不回退 | 4 | 4 支 γ 初值 0.5 且**梯度全非零**（{2.44e-1, 2.45e-1, 2.23e-1, 2.52e-1}）、`contrib_norm=615.7`、**负对照 γ=0 ⇒ contrib=0 且梯度全 0** |
| C 结构正确性 | 6 | Q=1024 / KV=256（N→N/4）、窗口不整除报错、输出 `(2,192,64,64)`、rank 不整除报错、**净网络 GPU 前向** `contrib=2.40e+02` |
| D 参数量 | 3 | report == real == 198,340；< 0.40 M；在结构性上界 240,004 内（0.826） |
| E fail-closed | 4 | `hyperbolic`/`euclidean` ⇒ `SystemExit`；`lattice` ⇒ 正常；`hqkv=1,hgsl=1` ⇒ `SystemExit` |

`train.py --help` rc=0，choices 已是 `{hyperbolic,euclidean,lattice,none}`（前两个对 v2 会 fail-closed）。
协议串 bump 为 **`run8c_vmamba_lattice_qkv_exact80k_v1`** ⇒ lattice 版 checkpoint
**不可能**被当成 Run8b 的 ckpt 复用。

## 5. 边界（不要拿这次拆除去支持过强的结论）

- 数据只支持「**双曲版** HQKV 不划算」（`B1_HYP` 四库 ΔF1 均值 −0.00085）；
- `B2_EUC` 只跑到 5.1%（`5,634/80,000`，停训时 KILL）⇒ **「欧氏版是否也不划算」没有答案**；
- lattice 版本**没有任何训练结果**。它的 27 项 PASS 只证明「代码对、能跑、有梯度、fail-closed」，
  **不证明**它比 CTRL 好。

## 6. 复现命令

```bash
# 归档核对 + 删除现行文件 + 回归门（幂等；归档不完整会 exit 3）
bash train_scripts/CASA-TViM/Run8b/_r8c_teardown.sh
# 只做总验收
bash train_scripts/CASA-TViM/Run8b/_r8c_verify.sh
# 单独跑回归门（本机无 GPU 时 C6 会 SKIP）
cd models && python test_hqkv_lattice_cpu.py
# 停训（含重试宿主；--kill-run 才动 train.py）
bash train_scripts/CASA-TViM/Run8b/_r8b_stop.sh --kill-run
```
