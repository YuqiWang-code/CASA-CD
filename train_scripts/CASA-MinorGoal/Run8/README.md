# Run8 — CD²-TAR：先解耦变化，再解耦尺寸

协议 `casa_minor_run8_cd2_v1`。SYSU 先跑、LEVIR 后跑；seed 2334；val 选模、test 只在最后用一次、阈值 0.5；
边缘监督关闭（`edge_weight=0`），让两个因子成为唯一差异。

## 四臂（请求的 2×2）

| arm | module | tar | 读的是什么 | 训练 / 部署参数 |
|---|---|---|---|---|
| `CTRL` | none | reparam | 无桥改动、无检索 | 2.0471 / 1.9358 M |
| `TAR_ONLY` | none | **cd2** | 只换桥：变化解耦 + 尺寸解耦 | 2.1222 / 2.0726 M |
| `HRA_ONLY` | manifold | diff | 只换检索：流形上压缩原型 | 1.9618 / 1.9234 M |
| `TAR_HRA` | manifold | **cd2** | 两者都有 | 2.1400 / 2.0904 M |

## 桥里做了什么（`--tar-variant cd2`）

1. **变化解耦**：`a = P/μ₇|P|`、`b = Q/μ₇|Q|` → `d⁺=ReLU(a−b)`、`d⁻=ReLU(b−a)` → `Δ = BN(1×1[d⁺,d⁻])`；
   场景内容走旁路 `M = SiLU(BN(1×1[P,Q]))`，两者只在融合点相遇。
2. **尺寸解耦**（无损三分，`S+Mid+L ≡ Δ`）：`S = Δ−P₃`、`Mid = P₃−P₉`、`L = P₉`；各接 `1×1+BN`；
   逐像素路由 `r = softmax(1×1[S,Mid,L])`（初值均匀，故 ×3 后 `Δ* = Δ`）。
3. **小目标容量只给 S**：`Fˢᵗᵉᵐ`（128², 16ch）与 `DW 3×3, d=2` 只接在 S 通道上；其余尺度没有这两项。
4. 融合沿用门控：`G = σ(1×1(Δ))`，`T = RepLocal(SiLU(M)·(1+G) + Δ*)`；跨尺度把细尺度证据 `area` 下采样后加给粗尺度。

## 裁决口径

- `TAR_ONLY − CTRL` ≥ +0.36 pp（噪声 0.32）→ 桥的收益成立；
- `HRA_ONLY − CTRL` ≥ +0.36 pp → 检索的收益成立；
- `TAR_HRA − CTRL` ≥ +0.47 pp → 两者叠加；
- 若 `TAR_ONLY + HRA_ONLY − 2×CTRL` 明显大于 `TAR_HRA − CTRL`，说明两模块不可加（Run7 已见 1.744 vs 1.625）。

## 运行

```bash
python bootstrap.py                      # 打包 models/ 与 Run8 脚本到服务器并写 fingerprint
ssh … 'cd …/Run8/runtime/train_scripts/CASA-MinorGoal/Run8 && source environment.sh &&         export CUDA_VISIBLE_DEVICES=1 && python prepare.py'
ssh … 'cd …/Run8/runtime/train_scripts/CASA-MinorGoal/Run8 && source environment.sh &&         export CUDA_VISIBLE_DEVICES=1 && MINOR_GPUS=1 MINOR_SLOTS=2 python pipeline.py'
```

跑完：`collect_results.py` → `compare.py` → `tools/extract_metrics_to_excel.py` → `tools/models_to_txt.py --tag CASA-MinorGoal --run Run8`。

## 要额外量三件事（与总 F1 一起看）

1. 路由 `r` 的三通道均值质量分布；
2. 按 GT 面积分层召回（`<32² / 32²–96² / >96²`）；
3. `‖S+Mid+L−Δ‖ ≤ 1e-6` 的恒等式校验（`models/test_run8_modules.py` 已覆盖）。
