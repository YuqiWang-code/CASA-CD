# Run4 — 主线二：<3M 极轻量结构（逐组件单变量推进）

> 决策依据：`docs/temporary/CASA-CD_Run4_极轻量结构主线_可执行方案.md`。
> CASAA change-aware router 已按预注册停止规则终止；Run4 目标：~2.11M 的
> UL-V4（TinyViT4-192 + LightDetail 32/64/128 + SABI + DFPD），四数据集超轻量 SOTA。
> **每个阶段有 gate，前一个不过不进入下一个；不做自动连跑队列。**

## 阶段 1（当前）：ViT 深度可行性 + corrected pretrain loader

- **R4-0 A0_FULL12_FROZEN / SYSU**：depth=12、vanilla attention、frozen、
  **corrected DeiT loader**（patch_embed 原位 + pos_embed 14×14→16×16 bicubic）、
  原 ResNet18/FI/Decoder——健康 full12 参考点。
- **R4-1 VIT4_OLDHEAD / SYSU**：depth=4，其余完全一致——回答
  「shallow pretrained prefix 是否提供足够的 16×16 semantic prior」。
- **Gate**：R4-1 vs R4-0 `ΔF1 >= -0.50` 通过；`(-1.0, -0.5]` 只允许 prefix-5；
  `< -1.0` 停止 prefix 策略。

## 阶段 2（R4-1 通过后）

- R4-2：ResNet18 → LightDetail 32/64/128（+临时 32→64/64→128/128→256 adapters），
  判据相对 R4-1 F1 drop ≤0.30；
- R4-3：FeatureInjector → SABI（三尺度先 pool 到 16×16、投影 48d、一次低秩
  cross-attention），判据相对 R4-2 F1 drop ≤0.30；
- R4-4：原 difference/up 解码器 → DFPD（difference-first pyramid），
  判据相对 R4-3 F1 drop ≤0.60；最终参数 **≤2.20M**、SYSU Gate-A F1 ≥81.50、
  论文候选 F1 ≥82.30 → 再跑 LEVIR（≥91.20）→ CDD/WHU。

## 训练协议（沿用）

BCE+Dice、Adam(lr 2e-4)、poly + 200 warmup、max_steps=80000、batch 16、
256×256、seed 16、test-as-val；Run4 第一轮全部 **frozen ViT**（解冻需先过
2K health gate，且只允许一次 vit_lr=2e-5 的 80K）。

## 路径

- Checkpoint：`/share_datasets/yqwang/checkpoints/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/`
- 日志：`/home/yqwang/outputs/CASA-CD/UltraLight/Run4/<VARIANT>/<DATASET>/train_log.txt`

## 启动顺序（GPU1 专属；GPU0 让给其它项目）

```bash
cd /home/yqwang/projects/CASA-CD
# U1/U2 审计
CUDA_VISIBLE_DEVICES=1 python analyse/param_breakdown.py --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth
CUDA_VISIBLE_DEVICES=1 python analyse/vit_pretrain_audit.py --pretrained_weight_path pretrained_weight/deit_tiny_patch16_224-a1311bcf.pth --vit_depth 4
# smoke + dry run
bash train_scripts/UltraLight/Run4/smoke.sh
bash train_scripts/UltraLight/Run4/dryrun_SYSU.sh
# R4-0 → R4-1 串行（各自 80K，~4-5h）
nohup bash train_scripts/UltraLight/Run4/train_R4_0_A0_FULL12_FROZEN_SYSU.sh > /dev/null 2>&1 &
# R4-0 完成并核对后：
nohup bash train_scripts/UltraLight/Run4/train_R4_1_VIT4_OLDHEAD_SYSU.sh > /dev/null 2>&1 &
```
