# 归档：CAACP-HQKV v2（双曲几何版）— 2026-10-20 拆除

> **为什么有这份归档**：2026-10-20 用户判定「加了双曲几何的 HQKV 不划算」，下令
> **停训并拆除双曲几何结构**。本目录原样保存拆除前的实现字节，供回溯、复核、以及论文中
> 引用"我们试过什么"。
>
> **本目录不是死代码**：它是产出 `Run8b` 那 8 个封口 run 的**确切源代码**，
> 与 `../SOURCE_SHA256.txt` 的记录**逐项相等**（见下表）。任何对 `Run8b` 结果的复核
> 都应当用这里的字节，而不是 `models/` 下的现行代码。

## 1. 字节一致性（这是本归档的核心证据）

| 文件 | 本归档 SHA256（前 16） | `SOURCE_SHA256.txt` 记录 | 一致 |
|---|---|---|---|
| `models/model/layers/hyperbolic.py` | `e28f07751fa41cce` | `e28f07751fa41cce…` | ✅ |
| `models/model/layers/caacp_hqkv_v2.py` | `b6f30bec7221ad2e` | `b6f30bec7221ad2e…` | ✅ |
| `models/model/casa_vmamba_net_v2.py` | `c6b3de16a45ac96e` | `c6b3de16a45ac96e…` | ✅ |
| `models/test_run8b_antilock_precheck.py` | `35bbee476ba9b779` | `35bbee476ba9b779…` | ✅ |
| `models/run8b_dry_run.py` | `9f4a66c554d8c7cd` | `9f4a66c554d8c7cd…` | ✅ |
| `models/train.py`（拆除前，取自 commit `94f9e69`） | `536b149d42667f1b` | `536b149d42667f1b…` | ✅ |

> `train.py` 也归档在这里，因为它同样被写进了那 8 个 run 的
> `arch.json.source_code_sha256`（`536b149d…`），而且拆除**改了它**（协议串、
> `--hqkv_geometry` 的 choices、v2 分支的 fail-closed、侧车字段）。
> 归档用的是 `git show 94f9e69:models/train.py` 的**原始字节**（用 Python 以 `wb` 写出，
> 避免 PowerShell 重定向把 LF 转成 CRLF —— 第一次就是这么写错的，哈希当场对不上，已修正）。

完整哈希见 `MANIFEST.sha256`。校验：

```bash
# 在仓库根目录执行：MANIFEST 里记录的是**原始仓库路径**
cd /home/yqwang/projects/CASA-CD
while read -r h p; do [ -n "$p" ] && echo "$h  $(basename $p)"; done \
  < train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/MANIFEST.sha256 > /tmp/arc.sha
cd train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20 && sha256sum -c /tmp/arc.sha
```

> 服务器侧一键校验 + 拆除脚本：`train_scripts/CASA-TViM/Run8b/_r8c_teardown.sh`
> （**先验归档，不完整就拒绝删除**；实测输出 `归档核对：OK=5 BAD=0` 才动手）。

## 2. 被拆掉的是什么（精确到机制）

拆除前的 `caacp_hqkv_v2.py` 做的是这条链：

```
T1/T2 各 stage 特征
  → 映射到双曲空间（Lorentz 模型）
  → 双时相差异感知的双曲变化分数 r = r0 + a·q（半径非常数）
  → 按「局部离散度 vs 全局半径」把分数拆成 detail / global 两条分支
  → 两分支各自一套特化 SAT 式 QKV（Q 全 N，K/V 压到 k·N）
  → 三支独立 γ 门控（detail/global/lite）+ RMS 归一化 + 非零初值 → 注入 DCR
  → 同时把 stage 分数喂给 HGSL 几何监督损失
```

拆除后**被完全移除**的部分：

1. `hyperbolic.py` 的 Lorentz 实现：`lorentz_change_score` / `fixed_radius_change_score` /
   `euclidean_change_score` / `cosine_change_score` / 半径与 ``r0`/`a` 参数 /
   `branch_rank_disagreement` 及退化诊断工具；
2. **半径分解**：`detail`（局部离散度）与 `global`（全局半径）两分支及其
   `detail_window=2` / `global_window=4` 两套压缩窗口；
3. 由 (2) 派生的**双分支特化 QKV**（每 stage 由 2 套 QKV 降为 1 套）；
4. **HGSL 的输入口**：`hgsl_score_maps()` 不再有分数可喂 ⇒ `hqkv=1 且 hgsl=1` 现在**直接报错**，
   不再静默跳过（见 `models/model/casa_vmamba_net_v2.py`）。

## 3. 为什么拆（数据，不是感觉）

`B1_HYP`（双曲 HQKV）vs `CTRL_R8B`（同位同协议同 seed，唯一变量 = HQKV 开关）：

| dataset | Δ F1 | Δ IoU |
|---|---:|---:|
| CDD | +0.0001 | +0.0003 |
| LEVIR | −0.0009 | −0.0015 |
| SYSU | −0.0012 | −0.0017 |
| WHU | −0.0014 | −0.0025 |
| **均值** | **−0.00085** | −0.00135 |

代价：**+0.304 M 参数 / +0.46 G FLOPs / 约 +9% 训练时间**。
⇒ 1 平 3 负、均值负、还更贵。详见 `../STOPPED_2026-10-20.md`。

## 4. 结论边界（不要用这份归档去支持过强的结论）

- 只支持「**双曲版** HQKV 不划算」；
- `B2_EUC`（欧氏反事实）只跑到 5.1% 被停训，**无 TEST 区块** ⇒
  「欧氏版是否也不划算」「整个跨层 QKV 注入是否无效」**都没有答案**；
- 「抗自锁」是**独立成立**的正面工程结论（`contrib_norm` 全程非零、三支 `gamma_grad` 非零、
  fail-fast 通过），它与几何无关，拆除几何**不应**把这条一起丢掉。

## 5. 拆除后新代码在哪

- `models/model/layers/caacp_hqkv_lattice.py` —— 无几何、无半径分解的**格点压缩跨层 QKV**；
- `models/model/casa_vmamba_net_v2.py` —— 改为调用 lattice 版；对 `hyperbolic`/`euclidean`
  **fail-closed 报错**；
- `models/test_hqkv_lattice_cpu.py` —— 新的 CPU smoke（前向/反向/γ 梯度/参数量/ token 压缩比）。
