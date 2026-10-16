# Exp1 — SPD(2)/H² 结构张量测地变化证据（探索线执行索引）

> 生成 2026-10-16。**代码在 `models_v1/`**（`models/` 是 Run8 campaign 的冻结身份，未改动）。
> 设计与预注册：`docs/temporary/CASA-CD_Exp1_SPD2结构张量测地证据_设计与预注册_2026-10-16.md`
> （§5/§6 已按用户 2026-10-16 指令收窄为**只训 LEVIR + SYSU**）。

## 0. 一句话

把每个像素的局部结构张量 J 当作 **SPD(2)** 流形上的点（det=1 切片 ≅ 曲率 −1/2 的双曲平面），
用仿射不变黎曼度量的**闭式测地距离**度量双时相「结构变化」，作为**参数自由**（0 参数）的
几何证据注入 DCR 最细尺度（1/4）。该算子在数学上对**任意仿射辐射变化严格不变**。

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `models_v1/model/layers/st_geodesic.py` | 数学核心（结构张量 / 双角度坐标 / H² 测地距离 / 证据图 / 可部署模块），含完整推导与数值纪律 |
| `models_v1/test_st_geodesic.py` | 20 项数学验收（T1 对拍 eigh 参考、T1b 反例守卫、T2/T2c/T11 辐射不变性、T4 分解恒等式、T12 真实变化响应…） |
| `models_v1/model/casa_vmamba_stgeo_net.py` | 网络（VMamba-Tiny + TAR/DCR + ST-GEO，与 Run8 `hqkv=0` 分支同序构造） |
| `models_v1/train_stgeo.py` | 训练/测试入口（协议与 Run8 逐项一致；日志格式与 Run8 相同，可直接被 `analyse/` 解析） |
| `models_v1/epoch0_check.py` | epoch-0 恒等性核对（三模式：dump Run8 CTRL / dump v1 / cmp） |
| `.claude/_deploy_v1.py` | 上传器（**只传 `models_v1/` 与 `train_scripts/Exploration/`**，绝不触碰服务器 `models/`） |
| `train_scripts/Exploration/Exp1/run_one.sh` | 单 run 启动（环境导出 / 非空目录守卫 / 3 次续训重试 / 严格 TEST 区块门） |
| `train_scripts/Exploration/Exp1/run_all.sh` | 逐库串行（默认 **LEVIR + SYSU**，可传库名列表覆盖） |
| `train_scripts/Exploration/Exp1/_exp1_preflight.sh` | 服务器预检（单测 + epoch-0 三方对拍 + 3 步真实 dry-run） |
| `train_scripts/Exploration/Exp1/_exp1_launch.sh` | `setsid nohup` 脱离会话拉起 |
| `train_scripts/Exploration/Exp1/_exp1_status.sh` | 只读状态快照 |
| `train_scripts/Exploration/Exp1/_exp1_stop_cdd.sh` | 范围收窄时的停机与**改名封存**脚本（不删除产物） |

## 2. 已实测证据（服务器 RSML-3，2026-10-16）

| 项 | 读数 |
|---|---|
| 数学单测 | **20/20 PASS**（本地 CPU 与服务器均跑） |
| 预训练 | `retained=212/222, worst_diff=0.0`（VMamba 官方 `vssm_tiny_0230` 逐位加载） |
| **epoch-0 逐位恒等** | `torch.equal(Run8 CTRL_R8, v1 stgeo=none) = True, max|d| = 0` |
| **γ 零初始化门** | `torch.equal(v1 none, v1 inject) = True, max|d| = 0` |
| 参数量 | ST-GEO **6,977**；非主干 2,038,868 → **2,045,845** |
| 3 步 dry-run | rc=0，`[ACTUAL-OPT-STEPS] 3/3`，batch32 峰值 ≈27 GB |
| ⇒ 结论 | **已跑完的 `CTRL_R8` 是本实验的合法同期对照**（同协议、同 seed、同共享路径、epoch-0 逐位一致） |

## 3. 执行

```bash
# 预检（只读 + 3 步 dry-run，GPU1）
bash train_scripts/Exploration/Exp1/_exp1_preflight.sh 1
# 启动（LEVIR → SYSU，严格 80K）
bash train_scripts/Exploration/Exp1/_exp1_launch.sh
# 状态
bash train_scripts/Exploration/Exp1/_exp1_status.sh
```

| 路径 | 位置 |
|---|---|
| 日志 | `/home/yqwang/outputs/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ/<dataset>/train_log.txt` |
| 权重 | `/share/yqwang/checkpoints/CASA-CD/CASA-TViM/Exploration/Exp1/STG_INJ/<dataset>/`（服务器：`/share_datasets/...`） |
| 代码 | `/home/yqwang/projects/CASA-CD/models_v1/` |

## 4. 判据（**预注册冻结**，详见设计文档 §6）

* **接受**：LEVIR 与 SYSU 的 F1 **均 ≥ CTRL_R8**（91.21 / 81.88）且 **SYSU 严格提升**，机制自检全过；
* **守门**：F1 降幅 ≤0.002；中大对象 pooled Recall 降幅 ≤0.02；
* **机制门**：`[PRETRAIN-LOAD].worst_diff==0`、epoch-0 逐位、折叠 disagreement=0（或刀锋像素带证据）、
  `[ACTUAL-OPT-STEPS] 80000`、**`[STGEO-GAMMA]` 不得休眠**（|γ|<1e-4 ⇒ `MECHANISM-DORMANT`）；
* **只作归因**：small pooled Recall / Hit@25 / 未匹配 <256px 预测数 / ObjP / ObjR。

## 5. 禁止项

1. 不得修改 `models/`（Run8 campaign 正在用它，且其 SHA256 身份链已冻结）；
2. 不得删除任何产物（只允许 `_SUPERSEDED_*` / `_INVALID_*` 改名封存）；
3. 不得在 campaign 中途改动受 `SOURCE_SHA256` 约束的文件；
4. 不得抢占 GPU0（Run8）或同门进程；只在 GPU1 上跑；
5. 不得事后放宽 §6 的判据或阈值；不得用 val-best 行代替正式 TEST 区块。
