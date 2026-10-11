# Run7 — HFAE（前两层频率分解高频 → 边缘检测辅助监督）执行索引

> **当前状态：预注册完成，代码已实现并通过本地可跑的 T0/T2/T8/T9/T10/T11；待服务器跑全量 T0–T11
> + SYSU dry-run，通过后按波次开训。** `training_runs_executed = 0`。
> 依据 README「研究目标与实验纪律（2026-10-11 修订）」：硬目标是**努力方向**；**每轮改进都上服务器
> 跑完整 80K**；训练前诊断从宽，**有效性由服务器 80K 消融判定**。

## 0. 冻结输入

| 项 | 值 |
|---|---|
| 预注册文档 | `docs/temporary/CASA-CD_Run7_高频边缘辅助监督HFAE_预注册_2026-10-11.md` |
| 决策锁 | `train_scripts/CASA-TViM/Run7/SELECTION_LOCK.json` |
| 新模块 | `models/model/str_hf_edge.py`（唯一的架构新增） |
| 接线文件 | `models/model/casa_tvim_str_net.py`、`models/train.py`、`models/eval.py` |
| 验收测试 | `models/test_run7_hf_edge.py`（T0–T11） |
| 真实数据 dry-run | `models/run7_dry_run.py` |
| 编排 | `train_scripts/CASA-TViM/Run7/run_all.sh --wave {1,2,3,all}` |
| `protocol_version` | `run7_hf_edge_exact80k_v1`（**由 run4 升级**，因训练目标改变） |
| 预训练 | `pretrained_weight/tinyvim_s_1000e.pth`（sha 短 `f63e58da8109dbb1`） |

## 1. 机制一句话

从 **stage0（1/4，HF 36ch，HF 占比 75%）** 与 **stage1（1/8，HF 32ch）** 的频率分解高频分支
（`SS2D.forward_core` 里的 `x_high = self.local_conv(x_high)`，**只读 hook，不改 `ss2d.py`**）
接出证据 → 中高频增强 → **变化边缘检测辅助监督** → 梯度回流高频分支以**重塑表征**。

**为什么是新机制**：Run5 否证的是**无监督读出**、F14 否证的是**手工证据通道**、Run2/Run4 否证的是
**末端旁路**、Run6 否证的是**读出层门控**。本机制改的是**主干算出来的东西**，且 tap 在 Run5 从未
接过的 **1/4**。去掉边缘监督即退化为已被否决的路线 ⇒ **监督是承重部件**（已获批准，λ=0.2 固定）。

## 2. 三个变体（deploy 参数实测）

| variant | `--hf_edge` | `--hf_edge_form` | train 图参数 | **deploy 参数** | 增量 |
|---|---|---:|---:|---:|---:|
| `M1_R7CTRL` | 0 | `none` | 5,084,017 | **4,880,190** | — |
| `E9_HFAE_A` | 1 | `aux_only` | 5,091,058 | **4,880,190** | **+0（零推理代价）** |
| `E10_HFAE_B` | 1 | `aux_inject` | 5,094,226 | **4,890,366** | +10,176 |

（硬门 ≤5,000,000 全部满足；`switch_to_deploy()` 对 `aux_only` 删除整个边缘头，
对 `aux_inject` 只丢监督专用 `edge_out`。）

## 3. 执行顺序

| 步 | 内容 | 阻断条件 |
|---|---|---|
| S0 | 上传 + 远端 `sha256sum -c` 对拍（`_r7_stage.py`） | 任一不符 ⇒ STOP |
| S1 | **服务器 T0–T11 全量**（`python models/test_run7_hf_edge.py --device cuda:0`） | 任一 FAIL ⇒ 不训练 |
| S2 | **SYSU dry-run**（`models/run7_dry_run.py`，3 步，三个变体） | 任一 gate FAIL ⇒ 不训练 |
| S3 | **波一 `M1_R7CTRL` × 4 库** 80K | 波内有 run 失败 ⇒ 不启动下一波 |
| S4 | **波二 `E9_HFAE_A` × 4 库** 80K | 同上 |
| S5 | **波三 `E10_HFAE_B` × 4 库** 80K | 同上 |
| S6 | 收口：8 库测试块门 + 对象级只读复算 + 与 CTRL 对比 | — |

GPU 布局：GPU0 = CDD + LEVIR，GPU1 = SYSU + WHU，每卡 2 并发、batch 32。
每个 run 自带 `last.pth` 崩溃恢复（仅限**本 run**，上限 3 次），**绝不复用 Run1–Run6 目录**。

## 4. 判据（预注册，见决策锁 `acceptance_criteria`）

- **主**：四库 F1 vs 同期 `M1_R7CTRL`；
- **次**：SYSU small(1–255px) pooled Recall / Hit@25（冻结参考 **0.1974 / 0.1896**）、
  未匹配小预测 CC（冻结 **2430**）、ObjP/ObjR（冻结 **0.5320 / 0.7674**）；
- **守门**：中/大对象 pooled Recall 不回退；CDD/WHU 不明显恶化；
- **机制自检**：`[HFAE-GRAD]` 首步 `hf_tap_grad_norm > 0`、epoch-0 `torch.equal(CTRL,EXP)`、
  真实 batch 折叠 disagreement = 0、deploy ≤5M、`[HFAE-LOSS]` 逐 epoch 下降；
- **接受**：`E9`/`E10` 至少一个在 **≥3/4 库 F1 ≥ CTRL** 且 **SYSU F1 > CTRL** ⇒ 记为机制有效；
  否则记**负结果 + 归因**（照 Run6 惯例，不回调阈值）。

## 5. 禁止项

1. **不得搜索 λ**（=0.2 预注册固定）；不得改动机制/两形态/预算/判据后重跑当"同一实验"；
2. 不得用 Run1–Run6 权重 warm start；不得多 seed；不得改 batch/步数/阈值/增强；
3. **不得改 `ss2d.py` / `caacp_ss2d.py` / `str_dcr.py` / `str_reparam.py` 任何现有算子**；
   高频 tap 只允许通过**只读 forward hook**；
4. 产物只做 `_INVALID_*` / `_SUPERSEDED_*` **改名隔离，绝不删除**；
5. `edge_out` 是**监督专用**，不得在推理期参与主输出（T7 折叠门守此线）；
6. 论文定位：按 **STR 内部增强/消融** 写；R8（EdgeRefNet, TGRS 2026）已占"边缘引导细化+边缘监督"，
   **不得**把边缘监督抬成独立第三贡献；投稿前须重新核验 R8 方法细节；
7. **不得在脚本仍在运行时上传对同一脚本的修改**：bash 对脚本文件按**字节偏移惰性读取**，
   插入/删除行会让运行中的实例重新读到错位区域，可能中途损坏编排。
   （本轮就险些把"开波护栏"加进**正在跑**的 `run_all.sh` —— `_r7_stage.py` 会上传清单里的
   **每一个**文件，所以一次常规 stage 就会覆盖活文件；已在**上传前**发现并回滚，并用
   `local == server` 的 SHA256 证明运行实例未受影响。要改脚本：要么等它退出，要么写成新文件。）

### 5.1 开波护栏的执行方式（因上述限制而固定为手动）

`check_mapping.py`（变体/波次映射 + 协议常量）与 `check_imports.py`（常量 import 完整性）
本应"每波前自动跑"，但把它们加进 `run_all.sh` 必须改**正在运行**的脚本（见 §5 第 7 条），
故改为**在每波收口门核验时手动执行并留痕**。这一取舍是可接受的，因为：
- 波二/三的代码路径**已由真实的 40 步 `train.py` 冒烟逐条验证**（两种形态 rc=0，
  含 `[HFAE-GRAD] hf_tap_grad_norm=4.90e-02`、sidecar 三处齐全）——比静态护栏更强的证据；
- 代码一次性上传后由 `SOURCE_SHA256.txt` 锁定，波间不会漂移。

## 6. 产物索引

```text
/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/best_F1=*.pth
/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/<variant>/<dataset>/train_log.txt
/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7_DRY/<variant>/<dataset>/{dry_run_log.txt,dry_run_summary.json}
```

（运行完成后在此登记 sha256；当前为空。）
