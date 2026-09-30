# Run4 — 主线二：<3M 极轻量结构（逐组件单变量推进）

> 决策依据：`docs/temporary/过去的想法/CASA-CD_Run4_极轻量结构主线_可执行方案.md`
> （原方案）+ `docs/temporary/CASA-CD_Run4_R4-2失败后_下一步决策与PSD-Detail方案.md`
> （R4-2/2b 失败后的修订方案，当前执行依据）。
> CASAA change-aware router 已按预注册停止规则终止；Run4 目标：~2.18M 的
> UL-V4（TinyViT4-192 + PSD-Detail 0.078M + SABI + DFPD），四数据集超轻量 SOTA。
> **每个阶段有 gate，前一个不过不进入下一个；不做自动连跑队列。**

## 阶段 1（已完成）：ViT 深度可行性 + corrected pretrain loader

- **R4-0 A0_FULL12_FROZEN / SYSU**：depth=12、vanilla attention、frozen、
  corrected DeiT loader、原 ResNet18/FI/Decoder → F1 **83.14**（健康 full12 参考）。
- **R4-1 VIT4_OLDHEAD / SYSU**：depth=4 → F1 **82.77**（ΔF1 −0.37 ≥ −0.50，**通过**）。

## 阶段 2（已完成，失败）：LightDetail 直接换代

- R4-2 VIT4_LIGHTDETAIL（32/64/128 + 1×1 adapters）：F1 82.30，ΔF1 −0.47 > −0.30 **未过**；
- R4-2b LIGHTDETAIL48（预注册容量 fallback 48/96/160）：F1 82.32，ΔF1 −0.45 **未过**
  （容量 bump 只 +0.02，噪声内 → 不像纯容量问题）。

## 阶段 2 修订（已完成，失败终止）

R4-2/2b 同时改变了两个因素（pretrained ResNet → random DSConv + 拓扑改变），
且预注册 fallback 已耗尽。修订方案执行结果：

1. **R4-D0 Detail Interface Audit（无训练，已完成）**：完整 SYSU test 对比
   ResNet direct vs Light raw vs Light adapted —— 深层（1/4、1/8）raw PR-AUC
   0.32 量级 vs ResNet 0.63/0.65，adapter 前后变化 ≤0.03。预注册判定
   A=0/3、B=1/3 → **C** → 跳过 adapter 修补，直接 PSD。报告：
   `outputs/UltraLight/Run4/R4_D0_DETAIL_AUDIT/audit_report.txt`。
2. **R4-2c ADAPTER_ALIGN**：audit 未触发情况 A → **永久停用**（脚本留存备查）。
3. **R4-2d PSD_DETAIL（已执行，FAIL）**：F1 82.02 / IoU 69.52（Gate 82.47/70.11），
   ΔF1 −0.75 vs R4-1；best@epoch64、末段无上升 → 非欠训练。**按 Stop-3 终止
   轻量 detail 路线：不启动 SABI/DFPD**。结果记录：
   `docs/temporary/CASA-CD_Run4_R4-2d_PSD结果与轻量detail路线终止.md`。

## 阶段 3/4（R4-2c 或 R4-2d 通过后才启动）

- R4-3：FeatureInjector → SABI（三尺度先 pool 到 16×16、投影 48d、一次低秩
  cross-attention），新判据（决策文档 §17）：相对 detail-pass 模型 F1 drop ≤0.20、
  IoU drop ≤0.35，且绝对 F1 ≥82.40；
- R4-4：原 difference/up 解码器 → DFPD（difference-first pyramid），
  最终必须同时满足：effective params ≤2.20M、SYSU F1 ≥82.30、SYSU IoU ≥69.92
  → 再跑 LEVIR（≥91.50）→ CDD/WHU。

## 训练协议（沿用，决策文档 §26 确认不改）

BCE+Dice、Adam(lr 2e-4)、poly + 200 warmup、max_steps=80000、batch 16、
256×256、seed 16、test-as-val；全部 **frozen ViT**（解冻需先过 2K health gate，
且只允许一次 vit_lr=2e-5 的 80K）。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/train_log.txt`
- Audit 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run4/R4_D0_DETAIL_AUDIT/`

## 启动顺序（GPU1 专属；GPU0 让给其它项目）

```bash
cd /home/yqwang/projects/CASA-CD
# 1) R4-D0 audit（无训练，~30-60 分钟；跑完人工读 [DECISION] 行）
nohup bash train_scripts/UltraLight/Run4/audit_R4_D0_detail_interface_SYSU.sh > /dev/null 2>&1 &
# 2) 按 audit 判定二选一（绝不同时启动两个）：
nohup bash train_scripts/UltraLight/Run4/train_R4_2c_ADAPTER_ALIGN_SYSU.sh > /dev/null 2>&1 &   # 仅情况 A
nohup bash train_scripts/UltraLight/Run4/train_R4_2d_PSD_DETAIL_SYSU.sh > /dev/null 2>&1 &      # 情况 B/C
# 3) 正式训练前先 dry run（R4-2d）：
bash train_scripts/UltraLight/Run4/dryrun_R4_PSD_SYSU.sh
```

代码入口：
- `models/model/psd_detail.py`（PSD-Detail，78,464 参数）
- `analyse/run4_detail_interface_audit.py`（R4-D0）
- `--detail_mode resnet|light|light48|light_bnrelu|psd`
- smoke：`python smoke_test.py --mode all`（含 T-PSD0..T-PSD5）
