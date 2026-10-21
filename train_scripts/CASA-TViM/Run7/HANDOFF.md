# Run7 HANDOFF — HFAE 消融**已完成**（12 run / 3 波；`STATUS = MECHANISM-PASS`）

> 生成时间：2026-10-11 10:29（首版）；**最后更新 2026-10-12 12:40（已完成）**。
> 本文件是**状态与恢复索引**，不是结果报告；**结果**见根 `README.md`「# Run7 结果」与
> `README.md`（本目录）§10。
>
> ## ✅ 2026-10-12 12:36 完成
>
> | 项 | 值 |
> |---|---|
> | 12 run | **12/12 VALID**（各严格 80,000 `optimizer.step`，协议 `run7_hf_edge_exact80k_v1`） |
> | 收口链 | `FINISH_RC=0`（`gate_rc=0 / check_run_fail=0 / object_rc=0 / verdict_rc=0 / audit_rc=0`） |
> | 判定 | **`MECHANISM-PASS`**，`winner = E9_HFAE_A`，`reasons = []` |
> | 独立审计 | **10 checks / 0 FAILED**（A8 单一代码身份 `dc6d9f6e55a1fabd`，12 行） |
> | 四库 F1 | CTRL 0.9717/0.9115/0.8268/0.9484 · E9 0.9705/**0.9134**/**0.8331**/**0.9507** · E10 0.9702/0.9123/0.8319/0.9518 |
> | ΔE9 vs CTRL | −0.0012 / +0.0019 / **+0.0063** / +0.0023（3/4 库 ≥ CTRL 且 SYSU 提升） |
> | 归档 | xlsx 78→90 行（0 丢失）· 快照 `docs/temporary/models_and_metrics_CASA-TViM_Run7.txt` |
>
> **接手者无需再等或再跑训练**：若只是要复核，直接按本目录 `README.md` §9 跑一遍交付状态核对，
> 再看根 `README.md`「# Run7 结果」。督导进程已正常退出（波三因 LEVIR 的 OOM 判失败而
> `[WAVE-ABORT]`，尾巴由人工恢复接管并已跑满 80K）。
>
> ## ⚠️ 2026-10-11 23:27：campaign 曾**整轮重跑**（前一轮全部作废封存）
>
> **原因**：`_ensure_hf_hooks()` 的 hook 幂等标记打错对象（打在 `RemovableHandle` 上、却到
> `mod._forward_hooks` 的 hook 函数里查），导致 `hf_edge=1` 的 run **每次 forward 重复注册
> 1 个 hook/模块**，单步耗时随累计 forward 数线性上升 ⇒ 波二 E9 四库劣化约 **4×**
> （CDD 65→282 s/epoch，CTRL 恒 56~63 s）。根因、定量定律（`2cF²`，用独立实测 c 无自由参数
> 预测四库斜率，误差 10~15%）、以及"纯性能缺陷、数值逐位中性"的证明见**预注册 §7.7** 与
> `README.md` **§7.16**。
>
> **处置（用户裁定）**：修复该缺陷后**整轮 12 runs 从零重跑**，以保持 `independent_audit.py`
> **A8「12 run 代码身份唯一」**（A8 = `len(unique_code_identity)==1`，跨版本会被它正确地判 FAIL）。
> 判据/阈值/λ/seed/80K/协议版本**一字未改**。
>
> | 项 | 值 |
> |---|---|
> | 新身份 | `models/model/casa_tvim_str_net.py = e125e7ac622bff0b…`（`SOURCE_SHA256.txt` 27/27 与服务器逐字节一致） |
> | 修复后验收 | **T0–T12 = 13 checks / 0 FAIL / 0 SKIP**；`check_mapping`/`check_imports` PASS |
> | 新 campaign 起点 | **2026-10-11 23:27**（波一启动）；supervisor pid 见 `/tmp/run7_supervisor.pid` |
> | 旧产物（**禁止删除**，仅供追溯） | 日志 `…/CASA-TViM/_SUPERSEDED_Run7_20261011_2324_prehookfix/`（含封存说明）；ckpt `…/checkpoints/…/Run7/*_SUPERSEDED_20261011_2324_prehookfix/` |
> | 作废步数 | CTRL 4×80000；E9 CDD 65,730 / LEVIR 66,677 / SYSU 63,375 / WHU 80,000 |
> | ETA（新，按 CTRL 实测增速） | 波一 ≈ 4.1–4.4 h（≈ 10-12 03:50）；波二 ≈ +4.4–4.7 h（≈ 08:30）；波三 ≈ +4.5–4.8 h（≈ **13:10**）；含收口归档 ≈ **13:30** |
>
> **旧 campaign 的任何中途数字（含"E9 同阶段四库 ≥ CTRL"的观察）一律不进结论、不进表格。**

> ⚠️ **本文件里的"进度/剩余时长"都是带时间戳的快照，必然随时间过期。**
> 请**以命令实测为准**，不要把下面任何进度数字当作当前值：
>
> ```bash
> # 当前进度（三个变体 × 四库的累计 optimizer.step / 80000）
> for v in M1_R7CTRL E9_HFAE_A E10_HFAE_B; do for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
>   f=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/$v/$d/train_log.txt
>   [ -f "$f" ] && printf '%-11s %-14s %s\n' "$v" "$d" \
>     "$(grep -E '^.ACTUAL-OPT-STEPS' "$f" | tail -1)"
> done; done
> # 督导是否还活着：**只能**用 kill -0（日志在波二/波三期间静默，别拿它判死活）
> kill -0 "$(cat /tmp/run7_supervisor.pid)" && echo ALIVE || echo DEAD
> ps -o pid,stat,etime,cmd --ppid "$(cat /tmp/run7_supervisor.pid)" | head -3
> ```
> 变体目录 `…/Run7/E9_HFAE_A/`、`…/E10_HFAE_B/` **出现** = 对应波次已启动。
> ⚠️ `Run7/` 下**不应**再有任何 `*SUPERSEDED*`（已移出 outputs 通配根，避免混进指标表）。

## 0. 新接手的 agent：按这个顺序做（约 5 分钟就能接手）

1. **读本文件 §1 / §1.1 / §1.2 / §1.3，以及文首 §7.7 红框**（整轮重跑的原因与封存位置）——
   现状、ETA、健康读数、解读锚。
   ⚠️ §1.3/§1.4 里 14:43–14:45 的中途读数属于**已作废的旧 campaign**，只作历史记录，不是结果。
2. **跑一遍交付状态核对**：`train_scripts/CASA-TViM/Run7/README.md` **§9**（三条命令）。
   全 PASS 才继续；有 FAIL 先查那一条。
3. **看训练到哪了**：**读训练日志**，不要读督导日志（波二/波三期间督导日志静默）：
   ```bash
   for v in M1_R7CTRL E9_HFAE_A E10_HFAE_B; do for d in CDD-CD-256 LEVIR-CD-256 SYSU-CD-256 WHU-CD-256; do
     f=/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/$v/$d/train_log.txt
     [ -f "$f" ] && printf '%-11s %-14s %s\n' "$v" "$d" "$(grep -E '^.ACTUAL-OPT-STEPS' "$f" | tail -1)"
   done; done
   ```
   变体目录**出现** = 该波次已启动；`nvidia-smi --query-compute-apps` 应恰 **4** 个。
4. **若 12 个 run 都齐了**（每变体 4/4 VALID）：
   ```bash
   cd /home/yqwang/projects/CASA-CD
   bash train_scripts/CASA-TViM/Run7/finish_run7.sh; echo "FINISH_RC=$?"
   ```
   `rc` 含义见 `README.md` §8.2（四个码都已端到端实测）。**不要**为了"更快"去改任何参数。
5. **写结论**：按 `README.md` **§8.4**（五种 verdict 写法）+ **§8.5b**（根 README 结果节骨架）
   + **§8.5**（三张表，`verdict.md` 里已直接产出）。**负结果照写**；硬目标未达≠失败，
   但**绝不可**回调 λ/阈值/seed。
6. **若中途出事**：督导死了 ≠ 训练死了。按 `README.md` **§8.8** 恢复
   （注意 `RESUME_SAME_RUN` 陷阱）；产物只改名不删除。

> **不要重验** §3 表里已验过的东西——重验只会花时间并引入 churn（每次改收口工具都要重跑
> 12-run 全量排练）。


## 1. 一句话现状

`M1_R7CTRL` / `E9_HFAE_A` / `E10_HFAE_B` × 四库（CDD/LEVIR/SYSU/WHU）各**严格 80,000** `optimizer.step`，
共 12 个 run，正在 2×RTX 5090 上按**三波串行**跑；监督进程自动接续下一波。**无错误**。

| 项 | 值 |
|---|---|
| 监督进程 | `/tmp/run7_supervisor.pid`（**重跑后 pid `2448174`，以 pidfile 为准**），日志 `…/CASA-TViM/Run7_supervisor.log` |
| 波次 → 变体 | 1 → `M1_R7CTRL`，2 → `E9_HFAE_A`，3 → `E10_HFAE_B` |
| GPU 分配 | GPU0 = CDD + LEVIR；GPU1 = SYSU + WHU（每卡 2 个 run） |
| 并发 | 4 个 run 同时跑（每卡 2 个）；batch 32；`--exact_max_steps 1`；波一实测 ~5.5 step/s/run |
| 进度 | **随时实测**（见文首命令）。⚠️ 前一轮进度与曲线已全部作废（见文首 §7.7 红框）。重跑 23:27 起波一；23:30 实测 0.9–1.1k/80k、epoch 耗时 CDD 56 s / LEVIR 40 s / SYSU 70 s / WHU 31 s |
| ETA（**实测**，2026-10-11 23:35 重算） | 波一 ≈ 10-12 **03:50**；波二 ≈ 08:30；波三 ≈ **13:10**；含收口归档 ≈ **13:30**（波次门 = 最慢那个 run） |
| 资源 | ckpt 每 variant ≈ 635 MB（4 run）；日志 ≈0.1 MB @80K；磁盘 2.6 T 空闲 |

### 1.1 ETA 的实测修正

波次门要求**四个 run 全部 VALID**，所以波次完成时间 = **最慢**那个 run 的完成时间。

| 波 | 实测 | 结果 |
|---|---|---|
| **波一** `M1_R7CTRL` | 09:27 → **13:35:17 GATE PASS (4/4 VALID)**，用时 **4.14 h**；聚合 **21.59 步/秒**（每 run 5.40） | ✅ 已完成 |
| **波二** `E9_HFAE_A` | 13:35:17 起；14:42 实测每 run 平均 **3.74–4.55 步/秒** ⇒ 波二总时长 ≈ **5.94 h** | ETA ≈ **19:31** |
| **波三** `E10_HFAE_B` | 按同速推算 | ETA ≈ **01:27** |

⇒ 三波全部结束 ≈ **01:27**；加上收口（`finish_run7.sh` ~3.5 min）+ 归档（下载/xlsx/快照/README）≈ **01:45**。

**⚠️ 实验组比对照臂慢约 30%（重要，需在论文里说明）**：
波一（`hf_edge=0`）每 run **5.40 步/秒**，波二（`hf_edge=1, aux_only`）只有 **3.74–4.55**。
这是**训练期**开销，来源是每步额外做的：256² 标签上的 dilate/erode（4 次 conv2d）生成边缘 GT、
64² 上的边缘头前向、以及经只读 tap 回流的辅助梯度与 HF hook 开销。
**它不影响推理**（`aux_only` 的 deploy 参数与 FLOPs 与 CTRL 逐位同值，见 README §7.13）。
⇒ 论文里"零推理代价"的说法成立，但要如实说明**训练时间增加约 30%**。

**复测命令**：`_r7_eta47.sh`（按波次起始时刻与当前累计步算平均速率）。
注意：**波次内督导日志静默**（见 §7），所以 ETA/进度必须从 train_log 或本脚本取。

### 1.3 冻结机制门的中途预检（14:43，**不是结果**）

> ⚠️ **本节属于 2026-10-11 23:24 被封存作废的旧 campaign**（§7.7 重跑），仅作历史记录；
> 新 campaign 需重新预检。

预注册的四条机制自检里，有两条最可能在收尾时"意外不过"，先做中途预检：

**⑤ `[HFAE-LOSS]` 逐 epoch 下降（要求收尾 `aux[-1] ≤ aux[0]`）** —— E9 四库趋势：

| dataset | n epochs | first → last | 斜率/epoch | `last ≤ first` |
|---|---|---|---|---|
| CDD | 51 | 0.308879 → **0.150981** | −1.47e-03 | ✅ |
| LEVIR | 72 | 0.320944 → **0.068898** | −1.80e-03 | ✅ |
| SYSU | 41 | 0.306743 → **0.146364** | −1.43e-03 | ✅ |
| WHU | 99 | 0.331062 → **0.071213** | −1.25e-03 | ✅ |

⇒ 四库**全部明显单调下降**，收尾时该门极可能通过（此前我把它列为"单点失效风险"，现已被测量去风险）。

**① `[HFAE-GRAD]` 首步梯度进高频分支** —— E9 四库首步行均已写出
（`aux_loss` 0.327–0.338），证据齐备 ✅

**③ 真实 batch 折叠 disagreement = 0** —— E9 目前 `[REPARAM-REAL-*]` 行数为 **0**（该行只在
80,000 步收尾时写出），届时若出现非零，`finish_run7.sh` 的 `[1b/6]` 步会**自动**生成刀锋定位证据
（见预注册 §7.6 与 README §7.15）；CDD-CTRL 已实测走过这条自动路径。

### 1.4 同阶段趋势：E9 相对 CTRL 在**拉开**（14:45，**不是结果**）

> ⚠️ **本节整节属于 2026-10-11 23:24 被封存作废的旧 campaign，仅作历史记录。**
> 旧 campaign 的 E9 曲线受 §7.7 hook 缺陷影响（性能，不是数值），该缺陷已证明数值中性；
> 但按纪律，**旧 campaign 的任何中途数字一律不进入结论、表格与裁决**。新 campaign 从零产生数字。

把 E9 与 CTRL 在**相同 cumulative 步数**处对照（取 CTRL 中 ≤ 该步数的最后一个 epoch F1）：

| dataset | E9 步数 | E9 bestF1 | CTRL@同阶段 | Δ | (14:41 时的 Δ) |
|---|---|---|---|---|---|
| CDD | 16,276 | 0.9487 | 0.9484 | **+0.0003** | −0.0003 |
| LEVIR | 16,279 | 0.9087 | 0.9048 | **+0.0039** | +0.0009 |
| SYSU | 15,375 | 0.8286 | 0.8106 | **+0.0180** | +0.0083 |
| WHU | 18,786 | 0.9469 | 0.9431 | **+0.0038** | +0.0028 |

- 同阶段下 E9 **4/4 库 ≥ CTRL**（14:41 时还是 3/4），且四个库的 Δ **全部在扩大**；
- **SYSU（小目标瓶颈库）增益最大（+0.0180）** —— 与预注册 §5「收益应在小目标召回/定位
  （SYSU、LEVIR 的建筑小目标）」的预期**方向一致**；
- 若该趋势保持，按预注册接受标准（≥3/4 库 F1 ≥ CTRL 且 SYSU 严格提升）E9 有望 **MECHANISM-PASS**
  —— 但**晚期收敛可能收窄差距**，最终一律以 80,000 步收尾的 TEST 区块为准；
- ⚠️ 本表只是中途健康/趋势观测，**不得**作为结论引用。




### 1.2 波一中期健康读数（2026-10-11 10:32，**仅供健康核对，不是结果**）

| dataset | best F1 @~26% | 当前 epoch | 该库总 epoch | epoch×iters/epoch | ≈80,000? |
|---|---|---|---|---|---|
| CDD | **0.9538** | 64/256 | 256 | 256×313 = 80,128 | ✓ |
| LEVIR | **0.9089** | 92/359 | 359 | 359×223 = 80,057 | ✓ |
| SYSU | **0.8319** | 54/214 | 214 | 214×375 = 80,250 | ✓ |
| WHU | **0.9475** | 120/431 | 431 | 431×186 = 80,166 | ✓ |

**这些数字的用途**：① 训练健康（F1 仍在涨、Loss 在降、无 `[RESUME]`/异常行）；
② **独立佐证步数预算**：四库 `总 epoch × iters/epoch` 全部 ≈80,000；
③ 给后来者一个**解读锚**：CTRL 已接近 Run1 `M1_FULL` 的**终值**
（Run1：CDD 0.9718 / LEVIR 0.9107 / SYSU 0.8347 / WHU 0.9507），
符合"同构对照"的预期 ⇒ 之后看 E9/E10 就是与这条线比。
④ 每个 ckpt 目录**恰好 1 个** `best_F1=*.pth`（收口 `pick_best_ckpt` 依赖此不变量）。

### 1.3 解读锚（10:51 更新）：CTRL 会落在哪，以及"硬目标"实际要求多少提升

| dataset | CTRL bestF1 @~1/3 训练 | vs Run1 终值 | **距硬目标** |
|---|---|---|---|
| CDD | 0.9578 | −0.0140 | **−0.0222** |
| LEVIR | 0.9092 | −0.0015 | −0.0158 |
| SYSU | 0.8319 | −0.0028 | −0.0181 |
| WHU | 0.9487 | −0.0020 | −0.0013 |

**怎么用这张表**（写结论时必须分清两件事）：

1. **预注册的接受标准**只看**相对 CTRL**：`≥3/4 库 F1 ≥ CTRL` 且 `SYSU F1 > CTRL`。
2. **硬目标（98.00/92.50/85.00/95.00）是努力方向，不是必达门**（新研究政策）。
   由上表可见，若 CTRL 最终落在 Run1 终值附近，则"达到硬目标"意味着 E9/E10 相对 CTRL 需要
   **CDD +2.2pp / LEVIR +1.6pp / SYSU +1.8pp / WHU +0.1pp** —— 对一个**辅助监督**模块而言这是
   相当高的要求。因此：
   - 若 verdict 为 `MECHANISM-PASS` 但未达硬目标 ⇒ 照实写"机制有效、未达硬目标"，
     **不得**因此回调任何参数；
   - 若 verdict 为 `GUARDRAIL-FAIL` / `MECHANISM-FAIL` ⇒ 记为负结果 + 归因（同样有价值）。
3. 上表是**训练中期读数，不是结果**；CTRL 的终值以 `finish_run7.sh` 产出的四库 F1 表为准。



## 2. 收口只需一条命令（已 12-run 全量排练，`FINISH_RC=0`，耗时 207 s）

> ⚠️ **别依赖那个 bounded watcher**：`.claude/_r7_watch.ps1` 是 **11 小时**上限
> （2026-10-11 09:28:37 启动 ⇒ 约 **20:28–20:31 自行退出**），而按 §1.1 的实测修正
> 三波全部结束约 **22:15** ⇒ **watcher 会在完成前约 1 小时 45 分钟就到期**。
> 也就是说**最后约 1.75 小时完全没有 watcher 覆盖**。
>
> ⚠️ **更要紧**：**督导日志在波二/波三期间是静默的**（督导同步阻塞在 `run_all.sh --wave N`
> 里，整波约 4–5 小时一行都不写）。所以"日志没更新"**不能**用来判断督导死活 ——
> 那会让人误以为督导挂了而去执行恢复流程。
> - **判断死活**：`kill -0 "$(cat /tmp/run7_supervisor.pid)"`
>   且 `ps --ppid $(cat /tmp/run7_supervisor.pid)` 能看到 `run_all.sh --wave N` ⇒ 正常运行。
> - **看进度**：读 `Run7/<variant>/<dataset>/train_log.txt`
>   （或 `nvidia-smi --query-compute-apps`，应恰 4 个）。


```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/CASA-TViM/Run7/finish_run7.sh; echo "FINISH_RC=$?"
```

**rc 含义、五种 verdict 的写法、必写的三张表、禁做清单** ⇒ 见 `train_scripts/CASA-TViM/Run7/README.md`
**§8 收口 runbook**（务必按它写结论，尤其是 `GUARDRAIL-FAIL`/`MECHANISM-FAIL` 也要记负结果）。

## 3. 已经验证过的东西（**不要重验**，重验只会浪费时间并引入 churn）

| 项 | 证据 | 出处 |
|---|---|---|
| 收口链 6 步整条 | 12-run 全量 DRY 排练 `FINISH_RC=0`（207 s） | README §7.9 |
| 逐 run 硬门 `check_run.py` | 排练 12/12 PASS；真实 CTRL 清单 9 条协议断言全过 | README §7.5 / §7.8 |
| 波次衔接（唯一自动交接点） | 逐字提取督导逻辑实测 → `4 4 320000` GATE PASS；3/4 正确拦截 | README §7.12 |
| 对象级复算 | 真实 E9/E10 权重三变体 `missing=0 unexpected=0` | README §7.6 |
| `object_metrics` → `verdict` 接口 | 真实 JSON 驱动，8 个必需字段全命中 | README §7.7 |
| deploy 参数门（三形态） | eval.py 实测 4.880/4.880/**4.890** M，FLOPs 2.6747/2.6747/**2.7170** G | README §7.13 |
| 注入点 | 读码核验：注到 **DCR 的 1/4 输入**（非 refine 之后） | README §7.14 |
| 预注册 ↔ 代码 | 逐条对拍一致；§6「不改既有算子」经 git 取证 | 预注册文档 §7 + README §7.10 |
| 崩溃恢复 | `RESUME_SAME_RUN` 陷阱实测确认 | README §8.8 |
| 单元/集成测试 | T0–T11 12 checks 0 FAIL；analyse 142 tests OK | 本会话 round 18 |

## 4. 如果会话在 run 跑完前结束

1. **先判断督导是否还活着**（pidfile + `kill -0`；日志 >5 min 无新行即为死）——
   注意**督导死了不等于训练死了**，4 个 `train.py` 是独立进程。
2. **按 `Run7/README.md` §8.8 恢复**。关键陷阱：督导调 `run_all.sh --wave N` 时**不带**
   `RESUME_SAME_RUN`，而**非空 ckpt 目录会被拒绝启动** ⇒
   - 波一未完成：直接重启督导即可（安全）；
   - 波二/波三之后：**必须先 `export RESUME_SAME_RUN=1`** 再启动（否则会在波二处 ABORT）。
3. 恢复后逐条核对：每变体 4/4 VALID、`check_run` 无 FAIL、日志无 `budget already reached`、
   `run_manifest.json` 的 `resume_source` 不得指向别的变体/库。
4. **绝不**删除任何 ckpt/日志（只 `_INVALID_*` / `_SUPERSEDED_*` 改名）。

## 5. 冻结项（见结果之前已固定，**任何情况下不得事后修改**）

- `λ = 0.2` **禁止搜索**；两部署形态（`aux_only` / `aux_inject`）固定；
- 判据与阈值以 `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json` 的
  `acceptance_criteria_frozen_numeric` 为**唯一规范形式**（冻结于 2026-10-11T09:35:00+08:00，
  当时 **12 个 run 都还没有任何 TEST 区块**，故无从观察结果）；
- **硬目标（98.00/92.50/85.00/95.00）是努力方向而非必达门** —— 未达不等于失败，
  **但绝不可因此回调 λ/阈值/seed**；
- 既有算子（`ss2d.py` / `caacp_ss2d.py` / `str_dcr.py` / `str_reparam.py` 等）Run7 期间未改动（git 已取证）。

## 6. 关键路径

| 用途 | 路径 |
|---|---|
| 训练日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/` |
| 收口产物 | `…/Run7/_RESULTS/{summary,verdict,INDEPENDENT_AUDIT}.json` + `check_run_*.json` |
| 对象级指标 | `…/Run7/_OBJECT_METRICS/object_metrics_*.json` |
| 收口工具 | `train_scripts/CASA-TViM/Run7/{finish_run7.sh,check_run.py,collect_results.py,run7_verdict.py,independent_audit.py}` |
| 对象级工具 | `tools/archive/run7_object_metrics.py` |
| 预注册文档 | `docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md` |
| **收口 runbook** | `train_scripts/CASA-TViM/Run7/README.md` §8（rc→动作、五种 verdict 写法、三张表、禁做清单） |
| 督导 | `train_scripts/CASA-TViM/Run7/supervise_waves.sh` + `…/Run7_supervisor.log` |

## 7. 关于 `Run7/` 目录里的 `_r7_*.sh`（一次性诊断探针，**不是流水线的一部分**）

服务器上 `train_scripts/CASA-TViM/Run7/` 除了被跟踪的正式文件外，还有约 **34 个** `_r7_*.sh`
（我本会话做验证/监测时上传的一次性探针，如 `_r7_survey.sh`、`_r7_eta.sh`、`_r7_rerun.sh`、
`_r7_rc4.sh` …）。它们**不影响任何流程**，已验证：

| 断言 | 实测 |
|---|---|
| 收口链是否依赖目录列表 | `finish_run7.sh` 里 `glob`/`listdir` 出现 **0** 次 ⇒ 全部显式路径，探针无法影响 |
| `check_mapping.py` 是否被探针干扰 | 在 34 个探针在场时实跑 = **PASS（14/14）** |
| `check_imports.py` | 实跑 = **PASS** |
| 审计的 `glob` | 只用于 `object_metrics_*.json` 与 `SOURCE_FILES.txt` 清单，与探针无关 |

⇒ 看到这些文件**不必清理**（清理反而有改错路径的风险）；正式流水线只认
`SOURCE_FILES.txt` 里登记的 **26 个条目**（其中包含 `HANDOFF.md` 自身）。

