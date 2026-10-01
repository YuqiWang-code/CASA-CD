# Run6 — ViT 语义预算重分配（PTPR + SGDP）

> 决策依据：`docs/temporary/CASA-CD_Run6_ViT语义预算重分配_可执行预注册方案.md`。
> Run5 已按 §12 收束（Mobile raw gate FAIL）；Run6 主推：删除独立 detail encoder，
> 从 ViT4 PatchEmbed token 做 PTPR 金字塔重建 + 复用 SGDP（≈2.100M / ≤2.0G）。

## 状态（已执行 → FAIL 停止）

**R6-D0 零训练 gate FAIL（2026-10-01，完整 SYSU test 4000 对）**：

```text
G1 P0   : PR-AUC 0.3711 < 0.44  FAIL
G2 B4   : PASS（0.6127 / 0.5535 / +0.4255）
G3 Fuse : PR-AUC 0.4753 < 0.50  FAIL（Spearman 0.2427 < 0.25）
```

审计有效（冻结 ViT checksum ✓；CTRL 精确复现 ResNet 1/8 = 0.6535/0.5948 ✓）。
**按 §4.6/§9：永久停止「ViT4 token-only + token reconstruction」路线；
不实现 PTPR、不启动 R6-1/2/3/4。** 记录：
`docs/temporary/CASA-CD_Run6_R6-D0结果与Run6停止.md`。

下一轮唯一出口：**semantic source replacement（候选 3）**，重新预注册。
可复用：`analyse/run6_semantic_token_audit.py`（换 semantic 源后照跑同一 gate）。

## 启动命令（历史留存）

```bash
cd /home/yqwang/projects/CASA-CD
bash train_scripts/UltraLight/Run6/audit_R6_D0_SEMANTIC_SYSU.sh
# 日志：/home/yqwang/outputs/CASA-CD/UltraLight/Run6/R6_D0_SEMANTIC_TOKEN_AUDIT/audit_report.txt
```
