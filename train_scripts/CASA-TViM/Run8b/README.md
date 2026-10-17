# Run8b — CASA-VMamba-STRNet v2（**抗自锁**）+ CAACP-HQKV v2 + HGSL

> 生成 2026-10-17。**这是 Run8 的修正重跑**：Run8 的 `γ` 零初始化导致模块**从未进入网络**
> （四库 `contrib_norm == 0`、`gamma ≈ ±5e-41`），H1/H2 与 CTRL 的 F1 逐位同级
> ⇒ Run8 的对比是 **`NO-OP`（无效对照）**，不是机制否证。
> 本轮唯一的科学目标是：**先让机制真的生效，再谈它有没有用。**

---

## 0. 冻结输入（开跑前不得更改）

| 项 | 值 |
|---|---|
| 骨干 | VMamba-Tiny（`vssm_tiny_0230` 口径，**29.94M**，`worst_diff=0` 逐位加载） |
| 新模块 | **CAACP-HQKV v2**（`hqkv_v2_antilock`）：三支独立门 + 每支 RMS 归一化 + **γ 非零初值** |
| 新损失 | HGSL（λ=**0.2 冻结**） |
| 几何 | Lorentz 双曲，`r0=0.5`、`a=0.5`（冻结，禁止搜索） |
| **抗自锁开关** | `--hqkv_gamma_init` 默认 **0.5**（**必须 > 0**；=0 只用于复现 v1 自锁作负对照） |
| 判据 | `SELECTION_LOCK.json::acceptance_criteria_frozen_numeric`（**开跑前冻结**） |
| 协议 | `run8b_vmamba_hqkv_antilock_exact80k_v2` |

## 1. 为什么必须做这一轮（Run8 的实测诊断）

Run8 的 `H1_HYP` 与 `H2_EUC` 在**四库全部**出现：

```
[HQKV-GAMMA]          detail ≈ ±5e-41   global = 0.0   lite = 0.0
[HQKV-DIAG-*]         contrib_norm = 0.0（4 个 stage）   branch_out_ratio_db = 0.0
```

F1 与 CTRL 逐位同级（CDD 0.9776 / 0.9777 / 0.9777；LEVIR 0.9121 / 0.9116；
SYSU 0.8188 / 0.8173；WHU 0.9434 / 0.9418）。

**根因（三条，全部是实现级，不是"双曲没用"）**

| # | 缺陷 | 后果 |
|---|---|---|
| 1 | **`γ` 零初始化 + 乘性门 ⇒ 自锁**：`r = γ·Δ`，`γ=0 ⇒ r=0 ⇒ ∂L/∂γ = ⟨∂L/∂r, Δ⟩ = 0` 恒成立 | `γ` 永远停在 0 |
| 2 | `gamma_lite` **从未进入前向**（stage0 输出来自 `lite_fuse`，却被乘 `gamma_detail`） | 该门永远收不到梯度 |
| 3 | `gamma_global` **从未进入前向**（detail/global 先 `o=a+b` 求和，再乘单一 `gamma_detail`） | 该门永远收不到梯度 |

**把 1 修好却没有修 2/3 ⇒ 仍然只有一支在起作用**，这就是为什么本轮**同时**修三处。

## 2. 修法（v2）

```
Δ_detail, Δ_global, Δ_lite  ──rms_normalize──▶  各支 RMS = 1
        │                                            │
        └── r = γ_detail·Δ_detail + γ_global·Δ_global + γ_lite·Δ_lite
                       （三个门**各自**进入前向，γ 初值 0.5）
        │
        └── × 1/√(n_branch · n_stage)   ← 固定结构常数（不参与梯度）
                                          初值注入 RMS ≈ 0.5（主干是单位方差）
```

**明确没做的事（踩过的坑，必须记）**：**不能在最后对 `out` 再做一次 `rms_normalize`**。
那会把 `γ` 的效果归一化掉、并再次截断 `∂L/∂γ` —— 等于把自锁换了层皮。首版犯过这个错。

**CPU 预检的负对照**（`models/test_run8b_antilock_precheck.py` C5）：
`γ=0` ⇒ `contrib_norm == 0` **且三个门的梯度全为 0** ⇒ 这就是 v1 的症状，被精确复现。
`γ=0.5` ⇒ 三支梯度 `0.34 / 0.34 / 0.33`，三支 share 各 1/3，注入 RMS **0.5054**（期望 0.5）。

## 3. 预训练纪律的取舍（**必须在论文披露**）

v1 用「`γ=0` ⇒ epoch-0 逐位等同无该模块」作为预训练兼容的证明。**v2 放弃这条**，
改用**更强、更直接**的证明：`[PRETRAIN-LOAD] worst_diff == 0`（主干 212/222 键逐位相等）。

理由：**零初始化 + 乘性门在数学上就是自锁的**（见 §1 #1），不是一个可接受的初始化方式。
代价：**epoch-0 不再与 CTRL 逐位一致**；四个门初值非零 ⇒ 模块从第一步就贡献真实扰动。

## 4. 四个变体（每阶段唯一变量 + 每阶段同期 CTRL）

| 波 | 变体 | hqkv | geometry | hgsl | inject | 作用 |
|---|---|---|---|---|---|---|
| 1 | **`CTRL_R8B`** | 0 | — | 0 | none | **同期对照** |
| 2 | **`B1_HYP`** | 1 | hyperbolic | 0 | dcr1 | **主实验**（抗自锁） |
| 3 | **`B2_EUC`** | 1 | euclidean | 0 | dcr1 | **曲率必要性对照** |
| 4 | **`B3_HYP_GS`** | 1 | hyperbolic | 1 | dcr1 | HGSL 增量 |

## 5. fail-fast（本轮新增的硬闸）

`train.py` 在 **step 1000** 检查 `contrib_norm`：

- `< 1e-3` ⇒ `raise SystemExit(3)`；编排器**不重试**，督导**立即停整条链**；
- 日志留下 `[FAILFAST]` / `[FAILFAST-FAIL]` 行，收口链据此判 `INVALID`。

**为什么需要它**：Run8 白烧了 4 个完整 80K run（每个 ~7 h）才发现 `contrib_norm` 恒为 0。

## 6. 执行顺序

```bash
cd /home/yqwang/projects/CASA-CD

# 0) 身份对拍 + 变体一致性（必须全 OK）
sha256sum -c train_scripts/CASA-TViM/Run8b/SOURCE_SHA256.txt | grep -c ': OK'
python train_scripts/CASA-TViM/Run8b/check_mapping.py

# 1) 只读预检（CPU，26 项，含 γ=0 自锁负对照）
cd models && python test_run8b_antilock_precheck.py

# 2) 服务器 smoke + 真实数据 dry-run（四库 × 2 变体）
bash train_scripts/CASA-TViM/Run8b/_r8b_smoke.sh
bash train_scripts/CASA-TViM/Run8b/_r8b_dryrun.sh

# 3) 开波（单卡串行，四波自动接续）
GPU_ID=0 setsid nohup bash train_scripts/CASA-TViM/Run8b/_r8b_pipeline_supervisor_single.sh \
    > /tmp/run8b/pipe1.stdout 2>&1 < /dev/null &
```

## 7. 禁止项

1. **禁止**改 `gamma_init`（0.5 冻结）、`r0`/`a`（0.5）、`λ_HGSL`（0.2）、lr、增广、阈值、seed；
2. **禁止**用 Run8 的数字当同期对照（Run8 是无效对照）；
3. **禁止**修改 Run1–Run8 的代码与产物；
4. **禁止**在一次 campaign 中途改任何受哈希约束的文件；
5. **禁止**在 `γ` 仍然 ≈0（即 fail-fast 未过）时解释任何 F1；
6. 产物只改名封存（`_SUPERSEDED_*` / `_INVALID_*`），**绝不删除**。

## 8. 已知风险（开跑前冻结）

| # | 风险 |
|---|---|
| R1 | `γ` 非零 ⇒ epoch-0 不再逐位等同 CTRL；预训练证明改用 `worst_diff==0` |
| R2 | **`Kendall τ(双曲, 同式欧氏) = 0.9983`**（CPU 预检）⇒ 曲率增量可能极小，`B2_EUC` 就是为抓这个 |
| R3 | Run7 的 `aux_inject`（E10）不优于 `aux_only`；注入先例失败，机制不同≠有效 |
| R4 | **`γ` 可能反向跑大**（单位 RMS 分量 + 非零初值）⇒ 每 epoch 的 `[HQKV-DIAG] gammas/contrib_norm` 必须盯着；固定缩放只约束**初始** RMS |
| R5 | GPU1 上是另一会话的探索线；`GPU_ID` 固定 0，督导**只观测** GPU1 |
| R6 | CASAA 的终止结论「routing ranking ↑ ≠ F1 ↑」仍未证伪 |

## 9. 结果归档

| 用途 | 路径 |
|---|---|
| 日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8b/<variant>/<dataset>/` |
| dry-run | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8b_DRY/<dataset>/<variant>/` |
| 正式指标 | 每个 `train_log.txt` 的**最后一个完整** `=== TEST RESULTS === … === END TEST RESULTS ===` |

## 10. 与 Run8 的对照关系（写论文时必须说清）

| 项 | Run8（v1） | Run8b（v2） |
|---|---|---|
| 模块结构 | 双分支，先求和后乘单一 `γ_detail` | **三支独立残差 + 三个独立门** |
| `γ` 初值 | **0**（自锁） | **0.5** |
| 分支归一化 | 无 | **每支 RMS = 1** |
| 实测 `contrib_norm` | **0.0（四库）** | 预期 O(1)（由 fail-fast 守门） |
| 结论有效性 | **`NO-OP`（无效对照）** | 本轮要回答的就是这个 |

> 只有 Run8b 的结果才能用来判断「双曲双分支 SAT-QKV 有没有用」；
> Run8 的结果只能用来证明「零初始化门 + 无归一化 = 自锁」这条工程结论。
