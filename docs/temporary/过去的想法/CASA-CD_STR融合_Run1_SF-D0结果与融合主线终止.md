# CASA-CD STR-Fusion Run1：SF-D0 审计结果与融合主线终止记录

> **日期**：2026-10
> **方案**：`docs/temporary/CASA-CD_STR融合_Run1_设计与预注册方案.md`（按 §6.1 / §7 执行）
> **执行事实**：完成了全部融合实现与训练前审计——`models/model/str_{reparam,tar,dcr,encoder,fusion}.py`、
> `models/test_str_reparam_equivalence.py`（T0/T1/T2/T2b）、`models/smoke_test.py` 的 T-SF-1..8、
> `analyse/run1_strfusion_budget.py`（G4）、`analyse/run1_strfusion_interface_audit.py`（SF-D0），
> 全部在 GPU1 运行。**0 个 80K**（SF-D0 FAIL → 未进入 dry run / 正式训练）。
> **结果纪律**：无正式 TEST RESULTS；gate 数字来自四数据集完整 test 集确定性零训练前向。

---

## 0. 结论先行

```text
[SF-D0-GATE] FAIL
  G0 (审计有效性)                          : PASS（R4-1 对照 0.6535/0.5948 精确复现；
                                              B4 边界带 PR-AUC 四数据集与 R8-D0 4 位小数逐位一致）
  G1 (PRbnd(fuse4) >= PRbnd(B4)+0.02,
      SYSU 必过且 >=2/4)                    : 0/4，SYSU +0.0089 < +0.02 必过项未过
  G2 (PRbnd(fuse2) >= PRbnd(B4)-0.02, >=3/4): 3/4（仅 WHU −0.0273 未过）
```

按预注册 §6.1：

> **不启动任何 80K，融合主线按预注册终止。** 不降 G1 阈值、不换 O-PRE/训练 stem 补救、
> 不训练「看看端到端会不会救回来」——补救方向需重新预注册。

**H（「深度即尺度」多深度 token 金字塔携带严格多于 B4-only 的边界带变化证据）被四数据集证伪。**

## 1. 审计有效性（G0 全过）

- 冻结 ViT4 与 corrected DeiT 初始化逐位一致（checksum ✓，depth 0–3 + norm + pos_embed）；
- SYSU R4-1 ResNet 1/8 对照：PR-AUC **0.6535** / Top32 prec **0.5948**（±0.005 精确命中）；
- B4-only 边界带 PR-AUC 与 R8-D0 记录四数据集 4 位小数逐位一致
  （CDD 0.5216 / LEVIR 0.5369 / SYSU 0.6113 / WHU 0.5718）。

## 2. 四数据集结果（rank 分数重采样 256×256 → 边界带 PR-AUC）

| 数据集 | B4-only 像素 / 边界 | fuse4（B1–B4 mean）像素 / 边界 | fuse2（B1+B2 mean）像素 / 边界 | fuse4 边界 lift | fuse2 边界 lift |
|---|---|---|---|---|---|
| CDD | 0.2173 / 0.5216 | 0.2317 / 0.5316 | 0.2227 / 0.5322 | **+0.0100** | **+0.0106** |
| LEVIR | 0.0908 / 0.5369 | 0.0934 / 0.5308 | 0.0843 / 0.5226 | **−0.0061** | −0.0143 |
| SYSU | 0.4691 / 0.6113 | 0.4865 / 0.6202 | 0.4744 / 0.6213 | **+0.0089** | **+0.0100** |
| WHU | 0.0837 / 0.5718 | 0.0751 / 0.5628 | 0.0649 / 0.5445 | **−0.0090** | **−0.0273** |

- G1（≥ +0.02）：**0/4**（CDD/SYSU 为正但 ≤ +0.010，LEVIR/WHU 为负）；
- G2（≥ −0.02）：3/4（WHU 的浅层证据显著稀释语义，−0.0273）。

## 3. 解读：深度维度救不回 inner-patch 局限——第八轮负结果

- **G1=0/4 是干净证伪**：四个冻结深度状态的任何参数自由融合，在真实变化边界 ±4px
  邻域都不比 B4-only 多出可辨识的证据；正向 lift 最好的两个数据集（CDD/SYSU）也
  只有 +0.010 量级，低于预注册的 +0.02 门槛，LEVIR/WHU 则为负。
- **与 R8-D0 结论完全同源**：R8-D0 证伪「B4 单深度无训练重建」，本审计证伪「多深度
  融合无训练重建」——**冻结 plain ViT 的 token 流无论取多少深度，都被 inner-patch
  局限约束**（ViT-CoMer 预警的实证再+1）。「ranking ≠ dense 重建」从单深度推广到
  多深度融合。
- **对 R8/R9 postmortem 的回应被数据驳回**：postmortem 说「补 dense evidence 需要
  训练过的任务适配空间表征」；本方案让 TAR/DCR 承担该角色，但接口证据 gate 表明
  **该表征无法从 16×16 token 网格中凭空重建 sub-patch 证据**——解码器再可训练，
  其输入的边界带信息量也不足。
- **融合假设的证伪点**：设计文档 §2.3 的诚实风险声明命中——「如果 SF-D0 gate FAIL，
  多深度接口同样继承 inner-patch 局限」，现在得到四数据集实测确认。

## 4. 停止执行内容（严格遵守）

- ❌ 不启动 SF-1（SYSU C0/M1）任何 80K；不跑 dry run；
- ❌ 不改 G1/G2 阈值、不换融合方式（max/min/加权）重刷 gate；
- ❌ 不用 O-PRE 作 fine-scale 侧向源、不加训练 stem、不解冻 ViT 救场——均需重新预注册；
- ❌ 不删本实现：全套代码作为「被 gate 否决的候选 + 可复用审计工具链」存档。

## 5. 留存资产（正面，可复用于任何后续预注册）

| 资产 | 数值 / 状态 | 用途 |
|---|---|---|
| STRFusion 全套实现 | `str_reparam/str_tar/str_dcr/str_encoder/str_fusion.py` + train/eval/smoke 加性分支 | 折叠原语 + 接口的完整可训练实现 |
| 折叠等价性 | T0 1e-13~1e-15；T1 9.5e-7~6.4e-6；T2 4.8e-7~1.2e-5；**T2b 活分支 5.1e-7/5.4e-7、disagree=0** | 折叠纪律在 CASA 工作区的移植验证 |
| smoke | T-SF-1..8 全过（epoch-0 逐位一致、双梯度家族、冻结 checksum、分支删除、C0/M1 deploy 相等） | 新机制 smoke 范式 |
| 预算（G4 PASS） | deploy TOTAL=EFFECTIVE **2,596,353（2.596M < 3M）**、FLOPs **2.2136G**（vs baseline 26.32G）；M1 训练图 3.123M/1.146M 可训练 | 机器实测预算事实 |
| RNG 纪律落地 | `skip_init` + 本地 Generator 的 aux 初始化（C0/M1 共享初始化逐位一致，0/267 键差异） | 解决 STR 交接文档 7.1-3 的坑 |
| SF-D0 审计工具 | `analyse/run1_strfusion_interface_audit.py`（四数据集边界带 gate，G0 自检可复用） | 未来任何「ViT token 作 dense 证据」候选的免费闸门 |

## 6. 课题状态更新（八轮负结果后的证据总结）

```text
Run3  CASAA deployable router       ：ranking↑ 未转 F1（证伪）
Run4  自定义轻量 detail ×3          ：端到端 gate 三连败（证伪）
Run5  MobileNet 预训练 prefix       ：raw gate FAIL（证伪）
Run6  canonical P0 重建             ：raw gate FAIL（证伪）
Run7  B2 深度截断                    ：四数据集 gate FAIL（证伪）
Run8  B4-only 无 detail 重建        ：dense boundary gate FAIL（证伪）
Run9  O-PRE 采样格变密              ：G0/G1 全 0/4（证伪）
Run10(融合) 多深度 token 金字塔      ：SF-D0 G1 0/4（证伪，本记录）
```

- 八轮全部在 **0–1 个正式 80K** 内被预注册 gate 拦截（正式训练总消耗仍为 1 个 R4-2d）。
- 证据链现在覆盖：token routing、detail 结构、patch 重建、深度截断、单/多深度 dense
  重建、采样格、可训练折叠解码器接口——**冻结 plain ViT 框架内「零/极小参数补回
  dense 边界」的路径已被系统排除**。
- 下一步若继续冲击四数据集硬目标，必须放弃「冻结 ViT 内取 dense 证据」这一整类假设，
  在以下方向重新预注册（**本轮不自动执行，用户按轮次驱动**）：
  1. **允许任务适配的空间表征源**：tiny 可训练 stem/浅层卷积旁路作 fine-scale 输入
     （承认「encoder=纯 ChangeViT」边界的代价，重新过 <3M 预算与零训练 gate）；
  2. **解冻受限 ViT**：block 0–1 低 lr 解冻（需先过 2K-step health gate；打破冻结
     纪律，风险高）；
  3. **放弃融合线**：回到 CASA-CD 分析型收尾（R4-1 82.77 为最强已验证轻量结构，
     R8/R9 与本文的负结果链作为 budget-allocation / negative-evidence study）。

## 7. 结果记录与审计证据

- 审计报告：`/home/yqwang/outputs/CASA-CD/diagnostics/STR-Fusion/Run1/SF_D0_INTERFACE_AUDIT/audit_report.txt`
- 预算审计：G4 PASS（deploy 2.5964M / 2.2136G；C0/M1 deploy 参数逐位相等）
- 等价性：`test_str_reparam_equivalence.py` ALL PASSED（含 T2b）
- 代码与文档：本地 `models/`、`train_scripts/STR-Fusion/Run1/`、本记录；已提交 GitHub
  （`update code`）。
