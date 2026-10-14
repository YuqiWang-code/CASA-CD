# Run8 HANDOFF — CASA-VMamba-STRNet（VMamba-Tiny + CAACP-HQKV + HGSL）

> 生成 2026-10-15 02:xx（本地）。本文件是**状态与恢复索引**，不是结果报告。
> 结果见根 `README.md`「# Run8 结果」与 `README.md`（本目录）§8。

## 0. 接手者按这个顺序做

1. 读 `README.md`（本目录）§0/§1/§2/§3/§7 —— 冻结输入、机制、四变体、参数预算、已知风险；
2. 读 `SELECTION_LOCK.json` 的 `user_rulings_2026_10_15` 与 `acceptance_criteria_frozen_numeric`
   （**判据在见结果之前已冻结，任何情况下不得事后修改**）；
3. 跑交付状态核对：
   ```bash
   cd /home/yqwang/projects/CASA-CD
   sha256sum -c train_scripts/CASA-TViM/Run8/SOURCE_SHA256.txt | grep -c ': OK'
   python train_scripts/CASA-TViM/Run8/check_mapping.py
   ```
   两条都 PASS 才继续；
4. 服务器 GPU 预检（**只读**）：`bash train_scripts/CASA-TViM/Run8/_r8_smoke.sh`；
5. 真实数据 dry-run：`bash train_scripts/CASA-TViM/Run8/_r8_dryrun.sh`；
6. 两关全绿才开波一：`bash train_scripts/CASA-TViM/Run8/run_all.sh --wave 1`。

## 1. 一句话现状（生成时）

**代码已写完、本地 CPU 预检已过（24/26，2 个 FAIL 是待对账的容差与口径），
服务器尚未跑任何 Run8 训练。** 等待：① SOURCE 身份账本生成；② smoke；③ dry-run；④ 波一。

| 项 | 值 |
|---|---|
| 协议版本 | `run8_vmamba_hqkv_exact80k_v1` |
| 骨干 | VMamba-Tiny v2，`vssm_tiny_0230_ckpt_epoch_262.pth`（已复制到 `pretrained_weight/`） |
| scan 内核 | `selective_scan_cuda_oflex`（casacd 自带；`casacd` 里的 `selective_scan_cuda.so` ABI 已坏，**不要用**） |
| 环境要求 | `PYTHONPATH` 前置 `mamba_base` 的 site-packages（见 `models/model/layers/vmamba_kernel.py`） |
| 波次 | 1=CTRL_R8 · 2=H1_HYP · 3=H2_EUC · 4=H3_HYP_GS；**四库串行**（GPU1 被占） |

## 2. 关键路径

| 用途 | 路径 |
|---|---|
| 训练日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/` |
| dry-run | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8_DRY/<dataset>/<variant>/` |
| 变体定义三处 | `run_all.sh::variant_args()` · `models/run8_dry_run.py::VARIANTS` · `SELECTION_LOCK.json::variants` |

## 3. 已做完的验证（**不要重验**）

| 项 | 证据 |
|---|---|
| VMamba 官方权重可被 `casacd` 进程使用 | 服务器探针：`sys.path` 前置 mamba_base 后 `selective_scan_fn` fwd+bwd 通过 |
| 退化定理在真实张量上复现 | `test_run8_cpu_precheck.py` C2：`frac_rank_diff=0.0, tau=1.000000` |
| 非退化性 | C3：`detail vs 全局分支` `frac_rank_diff=0.9746`；`detail vs 余弦` `0.9785` |
| γ 零初始化 ⇒ epoch-0 恒等（CPU） | C4.4：`out.abs().max() == 0.0` |
| 参数预算 | C6：非骨干 2.34M（CTRL 2.04M），HQKV 304,131 |
| HGSL 对小目标不塌 | C5.1：4px 小目标经 `adaptive_max_pool2d` 到 1/32 仍保留；avg 池化会抹掉 |
| 变体定义三处同步 | `check_mapping.py` PASS（4 variants / 4 waves / 39 flags） |

## 4. 尚未验证 / 待办（接手者要注意）

1. ~~`models/eval.py` 的 `casa_vmamba_str` 支持~~ —— 由子任务补齐，接手时先确认
   `python models/eval.py --help | grep hqkv` 有输出；
2. **VMamba-T 参数量口径**：文档写「≈30M」，本地按官方 v2 配置实测 **42,954,720**。
   已派任务与权重文件对账；**在对账完成前，论文不得写任何 VMamba 参数数字**；
3. 服务器 smoke（GPU 前向 + batch32 显存）——本地无 CUDA 跑不了；
4. dry-run 的四库结果；
5. `SOURCE_SHA256.txt` 需在文件冻结后由 `_r7_stage.py` 生成；
6. `finish_run8.sh` / `check_run.py` / `run8_verdict.py` / `independent_audit.py`
   尚未移植（照 Run7 的六步链改；**在四波跑完前完成即可**）。

## 5. 冻结项（见结果之前已固定，**任何情况下不得事后修改**）

- `λ_HGSL = 0.2`、`r0 = 0.5`、`a = 0.5` —— **禁止搜索**；
- 四个变体的开关组合（`hqkv` / `geometry` / `hgsl` / `inject_mode`）；
- `acceptance_criteria_frozen_numeric` 的全部阈值（含**曲率门**
  `curvature_min_datasets_h_gt_e = 3`、**非退化门** `non_degeneracy_min_rank_diff = 0.05`）；
- Run1–Run7 的文件不得修改；
- **不得**用 Run7 的数字当 Run8 的同期对照。

## 6. 明确不允许做的事

1. 见 §5 的冻结项；
2. 在一次 campaign 中途改任何受哈希约束的文件；
3. 声称 HQKV「零推理代价」（它保留在部署图里）；
4. 在 `H1 ≈ H2`（双曲 ≈ 欧氏）时主张「负曲率必要」；
5. 删除任何产物（只允许 `_SUPERSEDED_*` / `_INVALID_*` 改名）。
