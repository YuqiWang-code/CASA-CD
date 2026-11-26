# S7b 真实特征诊断报告（Run10 ManifoldContext，4 个 checkpoint）

审查基准：`main@1078bef`；本轮按 `docs/temporary/CASA-CD_V3仿真结论与下一轮实验执行方案.md` §4 执行 S7b。
**只做推理**（`eval()` + `no_grad` + FP32 + 关闭 TF32），不训练、不改冻结 runtime、不覆盖 checkpoint、不读 test。

## 1. 运行对象与环境

| 项 | 值 |
|---|---|
| 冻结源码 | `/home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/runtime` |
| checkpoint | `/share_datasets/yqwang/checkpoints/CASA-CD/CASA-MinorGoal/Run10/{arm}/{dataset}/best_model.pth` |
| 本次 4 个 checkpoint | `R_FULL-2334/SYSU-CD-256`、`R_FULL-2335/SYSU-CD-256`、`R_FULL-2334/LEVIR-CD-256`、`R_FULL-2335/LEVIR-CD-256` |
| manifest 核对 | `selection_split=val`、`selection=best val F1 at threshold 0.5 each epoch`、`independent_test=True`、`protocol=casa_pyramid_40k_testselect_heads_detail_v12r3` |
| 数据 | `/share_datasets/CD/{SYSU-CD-256,LEVIR-CD-256}`，仅 `list/{val,train}.txt`，**test 未加载** |
| 样本 | val 120 张、train 48 张，抽样清单先落盘（`s7b_samples.json`，含 filename / scene_id / change_px / stratum） |
| 设备 | CPU，`S7B_THREADS=8`（首次 4 进程各 48 线程把 load 打到 92，已改） |
| 输出 | `outputs/.../analysis/simulation/S7b/{arm}_{dataset}/`（CSV + `s7b_summary.json` + 8 张 `dump_*.npz`） |
| 截距温度 τ_s | R_FULL-2334/SYSU = [0.4403, 0.4880, 0.4444, 0.3167] |

## 2. 本轮对采集脚本的七项修复（方案 §4 要求）

1. 抽样清单先写盘，再跑模型；带 `scene_id`（去掉尾部 `_k` 切片编号）用于按原始场景分组。
2. train 与 val 使用**同一套确定性评估预处理**：`Scale(256,256) → Normalize(MEAN,STD) → ToTensor`；不再走 `make_loader("train")`（它含 RandomCropResize / Flip / Rotate90 / Exchange、shuffle、drop_last）。
3. `load_state_dict(..., strict=True)`（输出 `<All keys matched successfully>`），并核对 manifest 的 `selection_split=val`。
4. train / val 探针像素**分开采集**，禁止合并后随机划分。
5. pool 统计行拆成 `source_scale` 与 `class_id` 两个字段（旧版把类编号写进 `scale_index`）。
6. Recorder 可开关，且始终把原方法的返回值原样传回；不再改动 `collect_diag`。
7. parity：每个 checkpoint 前 4 张图比较四级输出、aux、二值差、原型 K/V、support。

**parity 实测：`max|Δlogit| = 0.0e+00`、aux 0.0、二值差 0、K/V 0.0、support 0.0**（逐位相同）。

> 过程中修掉的两个真实实现错：`manifold_pool` 的输入布局是 `points (B,h,R,P,D)`、`weights (B,R,2,P)`、输出 `(B,h,2R,D+1)`（原型顺序 `[r0c0,r0c1,r1c0,...]`），旧版统计按 `(B,h,R,P,D)+(B,h,R,D)` 广播，遇 `R=16,P=64` 直接维度错；`desc()` 的键在三个半径描述符上重名，写 CSV 时互相覆盖，已加 `mem_/memraw_/proto_` 前缀。

## 3. S7b-A：半径是否提供 guide 之外的判别信息

拟合在 train（48 图）上、评价在 val（120 图）上，逐尺度、逐像素；P1=guide logit，P2=模块半径，P3=logit+半径，P4=logit+原始投影范数（诊断），P5=方向特征（诊断）。

| checkpoint | 尺度 | P1 AUC | P2 AUC | P3 AUC | P4 AUC | P1 AP | P3 AP |
|---|---|---|---|---|---|---|---|
| 2334 LEVIR | 0 | 0.9944 | 0.4999 | 0.9944 | 0.9944 | 0.9513 | 0.9513 |
| 2334 LEVIR | 1 | 0.9951 | 0.6256 | 0.9951 | 0.9951 | 0.9577 | 0.9572 |
| 2334 LEVIR | 2 | 0.9954 | 0.5960 | 0.9953 | 0.9953 | 0.9594 | 0.9595 |
| 2334 LEVIR | 3 | 0.9945 | 0.9301 | 0.9925 | 0.9925 | 0.9516 | 0.9319 |
| 2334 SYSU | 0 | 0.9402 | 0.7227 | 0.9415 | 0.9415 | 0.8313 | 0.8331 |
| 2334 SYSU | 1 | 0.9395 | 0.7041 | 0.9396 | 0.9396 | 0.8321 | 0.8332 |
| 2334 SYSU | 2 | 0.9357 | 0.7691 | 0.9342 | 0.9342 | 0.8253 | 0.8234 |
| 2334 SYSU | 3 | 0.9268 | 0.8404 | 0.9285 | 0.9285 | 0.8164 | 0.8155 |
| 2335 LEVIR | 0 | 0.9934 | 0.5019 | 0.9934 | 0.9934 | 0.9449 | 0.9450 |
| 2335 LEVIR | 1 | 0.9941 | 0.6177 | 0.9941 | 0.9941 | 0.9526 | 0.9521 |
| 2335 LEVIR | 2 | 0.9948 | 0.9080 | 0.9945 | 0.9945 | 0.9543 | 0.9456 |
| 2335 LEVIR | 3 | 0.9889 | 0.8494 | 0.9868 | 0.9867 | 0.9168 | 0.9026 |
| 2335 SYSU | 0 | 0.9359 | 0.6502 | 0.9363 | 0.9363 | 0.8164 | 0.8180 |
| 2335 SYSU | 1 | 0.9353 | 0.5781 | 0.9354 | 0.9354 | 0.8152 | 0.8157 |
| 2335 SYSU | 2 | 0.9300 | 0.6184 | 0.9301 | 0.9301 | 0.8043 | 0.8043 |
| 2335 SYSU | 3 | 0.9177 | 0.5994 | 0.9182 | 0.9182 | 0.7942 | 0.7953 |

读法：
- 半径单独有 0.50–0.93 的 AUC，但**一旦与 guide logit 联合，增量落在 ±0.001 AUC 内**；16 个单元里 3 个 LEVIR 深层为**负**（−0.002 / −0.003 / −0.007 AUC，AP −0.02 / −0.006 / −0.018）。
- P4（原始投影范数）与 P3 逐位相同 ⇒ 模块半径确实只是投影范数的单调函数，两者提供同一信息。
- P5（方向）显著低于 logit 但非平凡（0.71–0.98）⇒ 方向携带信息，说明"压缩损失"若存在更可能出在方向/值聚合，而不是半径。

**判定（按方案 §4 S7b-A）**：P3 相对 P1 在两个数据集、两个 seed 上都没有稳定正向 AUC/AP 增益，且深层为负 ⇒ **继续暂停 OEM / 径向重锚 / 扩半径**；表述限定为"当前特征与当前探针未发现半径的增量证据"，不写成"半径无信息"。

## 4. S7b-B：压缩前后到底丢在哪

4 个 checkpoint 一致的趋势（比值 = 原型半径 / 成员加权半径；A = 方向一致性；ESS/px = 有效样本数 / 该区域像素数）：

| arm | kind | 半径保留比 | A | ESS/px |
|---|---|---|---|---|
| 2334 LEVIR | K | 0.797 | 0.916 | 0.63–0.82 |
| 2334 LEVIR | V | 0.772 | 0.973 | 0.63–0.82 |
| 2334 SYSU | K | 0.855 | 0.932 | 0.69–0.82 |
| 2334 SYSU | V | **0.546** | 0.918 | 0.69–0.82 |
| 2335 LEVIR | K | 0.808 | 0.919 | 0.63–0.85 |
| 2335 LEVIR | V | 0.784 | 0.970 | 0.63–0.85 |
| 2335 SYSU | K | 0.801 | 0.887 | 0.74–0.85 |
| 2335 SYSU | V | **0.668** | 0.916 | 0.74–0.85 |

逐尺度（2334 SYSU）：

| kind | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| K 成员半径 → 原型半径 | 0.241 → 0.218（0.900） | 0.233 → 0.195（0.828） | 0.231 → 0.195（0.836） | 0.251 → 0.217（0.857） |
| V 成员半径 → 原型半径 | 1.480 → 1.018（0.634） | 1.413 → 0.727（0.467） | 1.391 → 0.676（0.444） | 1.353 → 0.924（0.638） |
| V 池内半径 CV | 0.146 | 0.200 | 0.214 | 0.170 |

结论：
1. **K 内容没有被压到 0.6–0.7**：保留 0.72–0.90（均值 0.80–0.86），A 0.84–0.96。旧 S7 的"K 保留 0.86–0.92、V2 的 0.6–0.7"两个口径都不准。
2. **真正的损失在 V 侧**：SYSU 的 V 保留比只有 0.44–0.74（均值 0.55–0.67），中两层最差；LEVIR 的 V 保留 0.67–0.94 明显更好。V 是唯一进入 `readout` 的压缩量，这一不对称比"半径"更值得追。
3. ESS/px 0.63–0.85 ⇒ 一个原型只相当于区域内 63–85% 的独立像素；在 8×8 grid 的粗尺度（每区域 64 像素）EPS 约 40–55。
4. V 侧欧氏 SSE（20–150）远大于 K（0.15–0.7），因为 V 走的是未除 √d 的路径（`z=a`），量纲本来就大；跨臂比较只看比值、几何代价与余弦，不比较绝对 SSE。

## 5. 对上一轮结论的影响

| 旧说法 | 本轮结果 | 状态 |
|---|---|---|
| "K 被压到 Q 的 0.6–0.7" | K 半径保留 0.72–0.90 | **失效** |
| "K 原型半径 0.86–0.92、压缩温和" | 同上，且 V 只有 0.44–0.74 | 需改口：K 温和、V 严重 |
| "半径对变化没有独立信息" | 联合探针无增量，但单变量 0.50–0.93 | 改为"未发现增量证据" |
| "confidence support 恒 1e-6、梯度为 0" | 本轮 4 个 mass 臂不涉及；S13a-0 结论不变 | 保持 |
| "224 原型池化把半径压掉 30%" | 实测 K 10–28%、V 24–55%（按库不同） | 改写 |

## 6. 未完成（明确记录）

- 同 checkpoint 的 32 网格重建（需要另一次带 `kv_tokens=32` 配置的前向，本轮未跑）。
- `R_LEAN` / `R_COMPACT` / `R_RES` 的同口径采集（本轮只做 4 个 `R_FULL`）。
- S7b-C（support/温度/回写的离线反事实）、S13a-1（冻结先验）、S8b（真实四臂压缩）、S9（软注意力与下游反事实）。
- 逐图 bootstrap CI 的配对检验（本轮只报了逐尺度点估计）。

## 7. 复现

```bash
# 本地
python outputs/remote_cmds/push_file.py tools/simulation/s7b_capture.py   # REMOTE_TARGET=.../S7b/s7b_capture.py
# 服务器
cd /home/yqwang/outputs/CASA-CD/CASA-MinorGoal/Run10/analysis/simulation/S7b
S7B_THREADS=8 /home/yqwang/miniforge3/envs/casacd/bin/python s7b_capture.py \
    --arm R_FULL-2334 --dataset SYSU-CD-256 --val-limit 120 --train-limit 48 --dump-images 8
```

单个 checkpoint 约 90 s（CPU，8 线程）。
