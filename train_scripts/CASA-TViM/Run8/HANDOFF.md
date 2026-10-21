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

## 1. 一句话现状（2026-10-15 01:55 实测）

**代码已定稿并提交（commit `efba3d1`）；服务器 smoke 与 dry-run 全绿；波一 `CTRL_R8` 正在跑。**

| 项 | 值 |
|---|---|
| 协议版本 | `run8_vmamba_hqkv_exact80k_v1` |
| 骨干 | VMamba-Tiny（`vssm_tiny_0230` 口径）**29,938,176 = 29.94 M**；权重 worst_diff=**0.0** 逐位加载 |
| 非骨干参数 | CTRL **2,038,868**；+HQKV **2,342,999**（HQKV 本身 304,131） |
| FLOPs | CTRL **14.6841 G**（input 2×3×256×256，unsupported_ops=11） |
| scan 内核 | `selective_scan_cuda_oflex`（casacd 自带）；`mamba_base` 的 site-packages 前置在 `PYTHONPATH` |
| 排卡 | GPU0 单卡**四库串行**（batch32 peak 26.9–27.7 GB，余量仅 ~5 GB，**不可同卡 2 run**） |
| 波一实测 | CDD ~**104 s/epoch**（t=104s @ epoch1，F1=0.7730），313 iters/epoch ⇒ 256 epoch |
| ETA（粗估） | 单库 ≈ 15 iters/s ≈ **1.5 h**；单波 ≈ **6 h**；四波 ≈ **24 h** |

### 开跑时踩过的两个坑（已修，写在这里避免重犯）

1. **`run_all.sh` 由 `setsid nohup` 拉起时不继承 conda 激活** ⇒ `python: command not found`
   （rc=127，四库各失败 3 次）。已在脚本顶部显式 `source conda.sh` + `conda activate casacd`
   + `export PATH/PYTHONPATH/LD_LIBRARY_PATH/PYTORCH_CUDA_ALLOC_CONF`，并加 `command -v python` 断言。
2. **`train.py` 的 `--arch` choices 漏了 `casa_vmamba_str`** ⇒ argparse rc=2。
   为此新增 `_r8_preflight.sh`：用**真实 train.py + 真实数据**跑 3 步（`--max_steps 3`，
   ckpt 落 `/tmp`），把这两类错误压缩成 30 秒的检查。**开波前必须先跑它。**

## 2. 关键路径

| 用途 | 路径 |
|---|---|
| 训练日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/train_log.txt` |
| 权重 | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run8/<variant>/<dataset>/` |
| dry-run | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Run8_DRY/<dataset>/<variant>/` |
| 变体定义三处 | `run_all.sh::variant_args()` · `models/_legacy/run8_dry_run.py::VARIANTS` · `SELECTION_LOCK.json::variants` |

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

1. **服务器 smoke 与 dry-run 已全绿**（2026-10-15）：
   - smoke：VMamba worst_diff=0.0（212/222）、CPU 预检 26/26、HQKV 非退化性 0.73–0.996
     （门 0.05）、batch32 peak 26.9–27.7 GB；
   - dry-run：四库 × {CTRL_R8, H1_HYP} = **8/8 全绿**；epoch-0 逐位恒等；折叠门非零
     disagreement 全部判为 **KNIFE_EDGE_ARTIFACT**（1–3 像素、margin ≤ 3e-7）。
2. **VMamba-T 参数量口径已对账**：29,938,176（29.94 M），官方 ckpt 30,705,832（含 1000 类头）。
   论文引用必须连同 `depths=[2,2,5,2]`/`d_state=1`/`noz`/patch v2/downsample v3 一起标注。
3. **`SOURCE_SHA256.txt` 尚未生成** —— 文件已冻结（commit `efba3d1`），可在开波前用
   `_r7_stage.py` 生成；**一次 campaign 中途不得再改任何受哈希约束的文件**。
4. **收口工具尚未移植**：`finish_run8.sh` / `check_run.py` / `run8_verdict.py` /
   `independent_audit.py` / `collect_results.py`。照 Run7 六步链改；
   **在四波跑完前完成即可**（Run7 是在波三进行中就把收口链排练完的）。
5. 波二/三/四尚未开跑（逐个开：`bash _r8_launch.sh --wave N`）。

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
