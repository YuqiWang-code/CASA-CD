# Run-Diag / Diag1 复现说明（REPRODUCE）

> 诊断根目录（服务器）：`/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1`
> 执行日期：2026-10-10；本地仓库 HEAD（执行期间）：`fd8e0e3`
> 服务器工作副本**不是 git 仓库** → 代码身份以逐文件 SHA256 记录在 `RUN_MANIFEST.json:code_identity`。

## 0. 环境前缀（每个命令都需要）

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate casacd
cd /home/yqwang/projects/CASA-CD
export LD_LIBRARY_PATH=/home/yqwang/miniforge3/envs/casacd/lib/python3.10/site-packages/torch/lib:/home/yqwang/miniforge3/envs/casacd/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0          # 单卡；D1 与 D2 曾分别用 GPU0 / GPU1 并行
DIAG=/home/yqwang/outputs/CASA-CD/diagnostics/TViM-TinyLoss-Diag1
mkdir -p "$DIAG/logs"
```

环境事实（`RUN_MANIFEST.json:env`）：Python 3.10.21、torch 2.14.0+cu132、CUDA 13.2、2× RTX 5090（32 GB）。
`models/eval.py`/`train.py` 的 casa_tvim_str 口径 = **TF32 off + cudnn deterministic**（本诊断的 `set_eval_numerics()` 同款）。

## 1. P0：指标口径单元测试与协议落盘（无需数据）

```bash
python -m unittest discover -s analyse/tests -p 'test_tvim_*.py' -v      # 15 tests（含 torch 一致性）
python analyse/tvim_diag_common.py protocol --out-dir "$DIAG/P0"          # metrics_protocol.json
```
实测：本地（CPU-only）15 tests OK（1 skip=torch）；服务器 13/13 OK（torch 用例执行）。
`analyse/tests/test_tvim_log_parser.py` 还会把诊断侧日志解析器与
`analyse/extract_metrics_to_excel.py` 的官方实现**逐字对拍**（openpyxl 可用时）。

## 2. D0：M1 / A2 全 4000 张复算 + 折叠等价性 + manifest

```bash
python analyse/tvim_diag_common.py audit --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D0/M1_FULL"
python analyse/tvim_diag_common.py audit --dataset SYSU-CD-256 --run Run1 --variant A2_STR \
  --device cuda:0 --out-dir "$DIAG/D0/A2_STR"
```
实测耗时：M1 约 11 s（`time` 实测，含 250 个 batch 前向 + fvcore FLOPs + 折叠对拍）；A2 约 15 s。
结果：M1 `max_abs_delta=4.85e-5`（容差 1e-4）、折叠 disagreement=0 → **PASS**；
A2 `max_abs_delta=4.82e-5`、但固定真实 batch 上有 **1 个像素**（1/1,048,576 = 9.54e-7）刀锋翻转 → **FAIL（按文档硬门如实保留）**。

## 3. D1：小目标错误画像 + M1/A2 same-object 配对 + 预注册样例图

```bash
python analyse/tvim_small_error_audit.py --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --compare-run Run1 --compare-variant A2_STR --device cuda:0 \
  --out-dir "$DIAG/D1/M1_FULL" --examples
```
实测耗时：约 60–90 s（两个变体各一次全 test + 32 张样例图）。
产物：`object_level.csv`（5762 个 GT 对象逐行）、`image_level.csv`（4000 行）、`summary.json`、
`paired_delta_M1_FULL_vs_A2_STR.csv`、`paired_summary_M1_FULL_vs_A2_STR.json`、
`examples/`（**按 §5.3 预注册规则**抽样的 32 张 A/B/GT/pred/误差叠加图，仅保留在服务器）。

## 4. D2：分阶段 hook 诊断（本轮核心）

```bash
python analyse/tvim_stage_recoverability.py --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D2/M1_FULL" --verbose
```
实测耗时：约 2–3 min（4000 张 × 11 个编码器节点 + 10 个单路节点 + CAACP 内部统计）。
产物：`stage_raw.csv`（每图×每层）、`stage_summary.json`、`caacp_internal.json`、`stage_profile.png`、`gate.json`。
Gate：重建校验 `max_abs=0.0`、`disagreement=0.0` → **PASS**。

## 5. D3：冻结线性 probe（可选，本轮执行）

```bash
python analyse/tvim_linear_probe.py --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D3/M1_FULL" --max-train 3000 --steps 1500
```
实测耗时：约 2–3 min（特征提取 3011 张 + 7 个 probe × 1500 步 + test 一次性评估）。
预算偏差：拟合样本 3011（<全部 12,000），已写入 `probe_protocol.json:budget_deviation`。
产物：`probe_protocol.json`、`probe_results.json`、`probe_<layer>_PROBE_ONLY.pth`（诊断权重，**不入正式 checkpoint 目录**）、`gate.json`。

## 5b. D3b：可学习时相口径补证（本轮追加，§11）

```bash
python analyse/tvim_linear_probe.py --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D3_concat/M1_FULL" --encoder-input concat \
  --layers L01b_norm0,L03b_norm2,L05_stage3_last_prefix,L06b_norm4,L07_network5,L08b_norm6,P00_head_logits \
  --max-train 3000 --steps 1500
```
实测耗时：约 2.5–3 min（含 7 层 test 一次性评估）。
唯一变量 = 编码器 probe 输入口径（`absdiff` → `concat(F_A,F_B,|F_A−F_B|)`）；其余（checkpoint / 3011 训练样本 /
seed / steps / lr / batch）与 §5 完全一致；输出写入**新目录**，不覆盖 `D3/M1_FULL`。
一致性锚点：`P00_head_logits`（与口径无关）两次运行 pooled AP 在小数点后 9 位一致。

## 6. D4：CAACP β 受控反事实

```bash
python analyse/tvim_caacp_counterfactual.py --dataset SYSU-CD-256 --run Run1 --variant M1_FULL \
  --device cuda:0 --out-dir "$DIAG/D4/M1_FULL"
```
实测耗时：约 60 s（ON/OFF 各一次全 test）。
产物：`summary.json`、`paired_counterfactual.csv`（5762 对象）、`gate.json`。
完整性：`only_beta_changed=True`（β 在 state_dict 中有 3 个别名键，指向同一张量）、BN 校验和不变、ON 复现 max_abs=0。

## 7. 报告与 manifest

```bash
python analyse/tvim_diag_report.py --root "$DIAG" --out "$DIAG/DIAGNOSIS_REPORT.md" --write-manifest
```
产出 `DIAGNOSIS_REPORT.md`（含 §8 偏差、§9 结论、§10 Run4 候选）与 `RUN_MANIFEST.json`
（代码 SHA256、环境、数据 list SHA256、checkpoint SHA256、各 Gate 状态、两份协议）。

## 8. 本轮未执行（明确记录，避免误读）

- **D5 跨数据集**（LEVIR/WHU/CDD）：按文档要求，仅在 SYSU 结论稳健后收窄执行；本轮未做。
- **8 连通敏感性分析**：模块已支持（`connectivity=8`），未在本报告展开。
- **编码器 probe 的 `concat(F_A,F_B,|Δ|)` 口径**：本轮固定使用紧凑口径 `abs(F_A−F_B)`，作为下一轮首要补证项。
- 未做任何新的 80K 训练；未修改/删除历史 checkpoint、原始 train_log、既有 `docs/temporary/*.json`。
