"""ST-GEO：局部结构张量的 SPD(2)/双曲平面测地变化证据（探索线 Exp1 的数学核心）。

**参数自由**（parameter-free）算子：不引入可学习权重，只把「双时相局部结构张量」当作
SPD(2) 流形上的两个点，用仿射不变黎曼度量（affine-invariant metric, AIM）的**测地距离**
度量变化。全部闭式、可微、无特征值分解。

================================================================================
1. 为什么是「结构张量」而不是「特征向量」
================================================================================
Diag1/Run5 的实测把 SYSU 小目标瓶颈定性为**判别失败**而非证据缺失：未匹配小预测（FP）的
原始变化能量 change_abs = 0.9730 **高于**真实漏检 small GT 的 0.8626，也高于背景 0.6063。
⇒ 任何「幅度型」证据（|ΔI|、高频能量、边缘强度）都会优先放大已经在过 firing 的假变化。

辐射型伪变化（光照/阴影/大气/季节）几乎不改变**局部结构**（梯度方向与各向异性），
真正的地物变化（新建、拆除、地表类型改变）才会改变局部结构。
⇒ 正确的判别量不是「梯度有多大」，而是「局部结构怎么变」。

Di Zenzo 矢量结构张量（2×2 对称半正定，逐像素）：

    J_t(i) = Σ_c ( G_ρ * [ I_{c,x}²      I_{c,x}I_{c,y} ] )
                  (       [ I_{c,x}I_{c,y}  I_{c,y}²    ] )

I_{c,x} = ∂x(G_σ * I_c)；特征值 λ₁≥λ₂≥0 与主方向完整刻画该点的
「梯度能量 + 主方向 + 各向异性」。

================================================================================
2. 双角度坐标：J ↦ 单位圆盘（= H² 的 Poincaré 盘模型）
================================================================================
记 tr = Jxx+Jyy，Δ = √((Jxx−Jyy)² + 4Jxy²) = λ₁−λ₂。定义**复结构张量**

    z = ( (Jxx − Jyy) + 2i·Jxy ) / (Jxx + Jyy)   ∈ ℂ,   |z| = Δ/tr =: κ ∈ [0,1]

* 模 κ = 相干性/各向异性（0=各向同性，→1=理想阶跃边），幅角 = 2θ（主方向的双角度）。
  这正是数字图像处理里经典的「双角度朝向表示」，此处它同时是**双曲平面的坐标**。
* z 的**构造天然有界**（|z| ≤ 1），且对 J 的任意正标量缩放**严格不变**：
  J ↦ c²J ⇒ z ↦ z（精确，且是逐元素的比值，数值良态）。

**命题 1（det=1 切片 ≅ 曲率 −1/2 的双曲平面）**
{J ≻ 0, det J = 1} 在 AIM 下是常曲率 −1/2 的 H²。取 J = R(θ)diag(e^u, e^{−u})R(θ)ᵀ，
直接代入 ⟨V,W⟩_J = tr(J⁻¹VJ⁻¹W) 得拉回度量

    ds²_AIM = 2·du² + 8·sinh²u·dθ²                                  (◇)

⚠️ **必须记住的陷阱**：常见的 z = tanh(u)·e^{2iθ}（本模块 `shape_disk_coords` 的输出）
**不是** (◇) 的等距盘坐标——它诱导的度量为 4du² + 16tanh²u dθ²，只在 cosh u = 2 时才与
(◇) 成比例；直接对 z 套 Poincaré 盘距离公式会得到与真实 AIM 距离**系统性偏离**的结果
（实测最大偏差 1.25，随各向异性单调增大）。z 的正确用途是提供**尺度不变的 (κ, θ) 坐标**
（κ = |z| 各向异性、arg z = 2θ 双角度朝向），距离必须用 (◇) 的测地公式：

    cosh(d/√2) = cosh(u₁)cosh(u₂) − sinh(u₁)sinh(u₂)·cos(2Δθ)
               = 1 + 2W,   W = sinh²(Δu/2) + sinh(u₁)sinh(u₂)·sin²(Δθ)
    d          = √2·arccosh(1+2W) = √2·log1p( 2W + 2√(W(W+1)) )

*与 τ 写法的等价性*：z = tanh(u)e^{2iθ} ⇒ 2cosh(2·artanh|z|) = tr(Ĵ₁⁻¹Ĵ₂) = τ，
两种写法给出同一个数（T1 用 eigh 参考实现对拍验证）。

**为什么用 (★) 而不是 τ = tr(Ĵ₁⁻¹Ĵ₂) 的 arccosh 写法（数值纪律，必须披露）**：
τ 写法在 τ→2 处按 √ 变化，FP32 下「形状完全相同」的两点也会有 ~1e-3 的假距离；
更糟的是 Ĵ 的元素可达 O(√(λ₁/λ₂))，误差被各向异性比放大到 O(ε·R²)。
(★) 用**有界坐标** z（|z|≤1）且在恒等附近**线性**（artanh x ≈ x），
误差既不被 √ 放大也不被 R² 放大 ⇒ 纯辐射变化下 d_shape 在 FP32 中**恒为 0**（T2/T2c/T11 验证）。

**命题 2（辐射不变性，可数值证伪）**
  (i)  I_B = c·I_A（纯增益/对比度）⇒ J_B = c²J_A ⇒ z_B = z_A ⇒ **d_shape = 0（精确）**；
  (ii) I_B = I_A + d（纯偏置）⇒ J_B = J_A ⇒ **d_shape = 0 且 d_scale = 0（精确）**。
⇒ 形状分量对**任意仿射辐射变化**严格不变（这是「抗伪变化」的数学依据）。

**命题 3（尺度-形状正交分解，用于对拍）** 记 σ = √(det J)，则
    d²_AIM(J₁,J₂) = d²_shape(Ĵ₁,Ĵ₂) + 2·(log(σ₁/σ₂))²
（SPD(n) 在 AIM 下等距同构于 R × SL(n)/SO(n) 的乘积结构；T4 数值验证 <1e-4。）

================================================================================
3. 输出证据（每尺度 5 通道，全部闭式）
================================================================================
    [0] d_shape = √2·artanh|(z_A−z_B)/(1−z̄_A z_B)|   结构/方向变化（抗辐射伪变化）
    [1] d_str   = |log(tr_A/tr_B)|                    梯度能量变化（良态的尺度项）
    [2] κ_A     = |z_A| ∈[0,1]                        各向异性（边缘/角点指示）
    [3] κ_B     = |z_B|
    [4] ori     = |sin(θ_A−θ_B)|·min(κ_A,κ_B)         朝向变化（相干性加权，抗噪）
（另提供 det 版尺度 `det_scale_distance`，仅用于命题 3 的对拍。）

**注意 d_str 与 d_shape 的分工**：d_str 对真实变化（新增/移除结构）敏感但**不抗辐射**；
d_shape 抗辐射但只对结构变化敏感。两者一起交给网络融合 ⇒ 单变量、不预设哪个更重要。

================================================================================
4. 与 Run8「特征空间 Lorentz 距离」的本质区别（novelty 边界）
================================================================================
| | Run8 CAACP-HQKV / HGSL | 本算子 ST-GEO |
|---|---|---|
| 流形上的点 | **网络特征** x ∈ R^C（人为映射到 Lorentz 双曲面） | **图像结构张量** J ∈ SPD(2)（天然的 SPD 点） |
| 度量来源 | 人造非恒定半径 r = r0 + a·q（r0,a 冻结） | J 自身的 tr/Δ（无自由参数） |
| 退化风险 | 固定半径 ⇒ 与 1−cos 同序（靠可变半径逃逸） | 不存在：J 是梯度的二阶统计，与特征余弦无关 |
| 抗辐射伪变化 | 无 | **命题 2：精确不变**（可数值证伪） |
| 分辨率 | 1/8、1/16、1/32（1–15px 已是 sub-token） | **全分辨率 256²** |
| 参数 | 304,131 | **0** |

与 2024–2026 已有双曲 CD 工作（Hyperboloid-Embedded Siamese Network, JSTARS 2024；
Deep Change Monitoring, CVPR 2025）的区别：它们把 CNN/ViT **特征嵌入**双曲空间再算距离；
本算子不做任何嵌入学习，作用在**图像自身的二阶结构统计**上，闭式、参数自由、逐像素、
全分辨率，并带一条可证伪的不变性定理。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TR_FLOOR", "D_SHAPE_FLOOR",
    "gaussian_kernel1d", "structure_tensor", "shape_disk_coords",
    "poincare_shape_distance", "tr_based_shape_distance", "trace_scale_distance",
    "det_scale_distance", "orientation_change", "coherence",
    "spd_geodesic_evidence", "STGeoEvidence",
]

TR_FLOOR = 1e-10          # tr(J) ≤ TR_FLOOR 的 cell 视为无数值意义的平坦 cell
D_SHAPE_FLOOR = 1e-3      # 形状距离死区：d < 1e-3 记 0（量程 0–8 的 0.0125%）
KAPPA_MAX = 1.0 - 1e-3    # κ 上限：κ→1 处 det=JxxJyy−Jxy² 出现灾难性抵消，且
                          # artanh′(κ)=1/(1−κ²) 把 κ 的 FP32 误差放大到 1e-3 量级
                          # （实测纯对比度变化下产生 2.6e-3 假距离）。
                          # 截断对应各向异性比 λ₁/λ₂ ≲ 2000，都是「极锐利边缘」，
                          # 该处距离饱和是可接受且必须披露的行为。
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. 结构张量
# ---------------------------------------------------------------------------
def gaussian_kernel1d(sigma, truncate=3.0, dtype=torch.float32, device=None):
    """归一化 1D 高斯核（确定性、无 RNG）。"""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    r = int(max(1, math.floor(truncate * sigma + 0.5)))
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur(x, sigma, truncate=3.0):
    """可分离高斯模糊（replicate padding，通道独立）。x: (B,C,H,W)。"""
    B, C, H, W = x.shape
    k = gaussian_kernel1d(sigma, truncate=truncate, dtype=x.dtype, device=x.device)
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, k.numel())
    ky = k.view(1, 1, -1, 1).expand(C, 1, k.numel(), 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=C)
    return x


def _central_diff(x, dim):
    """中心差分（replicate 边界），与 x 同尺寸。dim=3 → ∂x，dim=4 → ∂y。"""
    if dim == 3:
        xp = F.pad(x, (1, 0, 0, 0), mode="replicate")[..., :, :-1]
        xm = F.pad(x, (0, 1, 0, 0), mode="replicate")[..., :, 1:]
    else:
        xp = F.pad(x, (0, 0, 1, 0), mode="replicate")[..., :-1, :]
        xm = F.pad(x, (0, 0, 0, 1), mode="replicate")[..., 1:, :]
    return 0.5 * (xp - xm)


def structure_tensor(img, sigma_d=1.0, sigma_i=None, truncate=3.0):
    """Di Zenzo 矢量结构张量。img: (B,C,H,W)（同一时相的 C 个通道）。

    返回 (Jxx, Jxy, Jyy)，各 (B,1,H,W)、FP32、逐像素半正定。
    sigma_i 默认 = 2·sigma_d（微分尺度 σ / 积分尺度 ρ 的标准取法）。
    """
    if sigma_i is None:
        sigma_i = 2.0 * sigma_d
    s = _gaussian_blur(img.float(), sigma_d, truncate=truncate)
    ix = _central_diff(s, 3)
    iy = _central_diff(s, 4)
    jxx = _gaussian_blur((ix * ix).sum(dim=1, keepdim=True), sigma_i, truncate=truncate)
    jxy = _gaussian_blur((ix * iy).sum(dim=1, keepdim=True), sigma_i, truncate=truncate)
    jyy = _gaussian_blur((iy * iy).sum(dim=1, keepdim=True), sigma_i, truncate=truncate)
    return jxx, jxy, jyy


# ---------------------------------------------------------------------------
# 2. 双角度（Poincaré 盘）坐标与闭式测地距离
# ---------------------------------------------------------------------------
def shape_disk_coords(jxx, jxy, jyy, tr_floor=TR_FLOOR):
    """(Jxx,Jxy,Jyy) → (zx, zy, kappa, valid)。

    z = ((Jxx−Jyy) + 2i·Jxy) / (Jxx+Jyy)，|z| = κ ∈ [0,1]。
    对任意 c>0：J ↦ c²J ⇒ z 不变（**精确**，逐元素比值）。tr ≤ tr_floor 的平坦 cell
    置 valid=0、z=(0,0)（形状无定义；此时 ±∞ 的朝向不可能稳定估计）。
    """
    tr = (jxx + jyy)
    safe = tr.clamp_min(tr_floor)
    valid = (tr > tr_floor).float()
    zx = (jxx - jyy) / safe
    zy = (2.0 * jxy) / safe
    kappa_raw = torch.sqrt((zx * zx + zy * zy).clamp(0.0, 1.0))
    # κ 截断（并同步缩放 z，保持 |z| == κ 的自洽性）
    scale = (KAPPA_MAX / kappa_raw.clamp_min(_EPS)).clamp(max=1.0)
    zx = zx * scale
    zy = zy * scale
    kappa = kappa_raw * scale
    zx = torch.where(valid > 0.5, zx, torch.zeros_like(zx))
    zy = torch.where(valid > 0.5, zy, torch.zeros_like(zy))
    kappa = torch.where(valid > 0.5, kappa, torch.zeros_like(kappa))
    return zx, zy, kappa, valid


def poincare_shape_distance(zx1, zy1, k1, zx2, zy2, k2, d0=D_SHAPE_FLOOR):
    """AIM 测地距离（闭式、无条件数放大）。

    **推导（必须写进论文）**：取 J = R(θ)diag(e^u, e^{−u})R(θ)ᵀ（det=1 切片），
    直接代入仿射不变度量 ⟨V,W⟩_J = tr(J⁻¹VJ⁻¹W) 可得拉回度量

        ds²_AIM = 2·du² + 8·sinh²u·dθ²                                (◇)

    即该切片是**曲率 −1/2 的双曲平面**（H²），u 为测地半径方向、2θ 为双角度方向。
    （注意：常见的 z = tanh(u)e^{2iθ} 并不是这个度量的等距盘坐标——那会得到
    4du²+16tanh²u dθ²，只在 u 取特殊值时才与 (◇) 成比例，T1 会把它判 FAIL。）

    (◇) 的常曲率 −1/2 测地距离满足
        cosh(d/√2) = cosh(u₁)cosh(u₂) − sinh(u₁)sinh(u₂)·cos(2Δθ)
    用 cosh(u₁−u₂) = cosh u₁cosh u₂ − sinh u₁sinh u₂ 与 1−cos2Δθ = 2sin²Δθ 改写：

        cosh(d/√2) = 1 + 2W,   W = sinh²(Δu/2) + sinh(u₁)sinh(u₂)·sin²(Δθ)  ≥ 0
        d = √2·arccosh(1+2W) = √2·log1p( 2W + 2√(W(W+1)) )

    **数值纪律（本实现的关键）**：W 是**平方量之和**，不做「两个大数相减」，
    故在恒等附近 d ≈ 2√2·√W 对扰动是**线性**的（τ = tr(Ĵ₁⁻¹Ĵ₂) 的 arccosh 写法在此处
    按 √ 放大、且误差随各向异性比 R² 增长，实测纯对比度变化下有 ~1e-3 假距离）。
    逐项用恒等式避免抵消：
        t   = (κ₁−κ₂)/(1−κ₁κ₂)                   （artanh 差角公式）
        sinh²(Δu/2) = t²/(2√(1−t²)(1+√(1−t²)))   （有理化，t→0 时 ≈ t²/4）
        sinh(u₁)sinh(u₂) = κ₁κ₂/√((1−κ₁²)(1−κ₂²))
        sin²(Δθ) = (1 − cosΔφ)/2,  cosΔφ = (z₁·z₂)/(κ₁κ₂)
    d < d0（默认 1e-3，量程 0–8 的 0.0125%）记 0：纯辐射变化下形状通道**恒为 0**。
    """
    k1c = k1.clamp(0.0, 1.0 - 1e-6)
    k2c = k2.clamp(0.0, 1.0 - 1e-6)
    t = ((k1c - k2c) / (1.0 - k1c * k2c).clamp_min(1e-6)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    one_minus_t2 = (1.0 - t * t).clamp_min(1e-12)
    sq = one_minus_t2.sqrt()
    term_u = t * t / (2.0 * sq * (1.0 + sq))                    # sinh²(Δu/2)
    denom_k = ((1.0 - k1c * k1c).clamp_min(1e-12)
               * (1.0 - k2c * k2c).clamp_min(1e-12)).sqrt()
    # 朝向项（双角度）。**必须用有理化写法**：直接算 sinh u₁sinh u₂·(1−cosΔφ)/2 在
    # Δφ→0 时被灾难性抵消吃掉（cos ≈ 1−6e-8 已落在 FP32 ulp 内），实测假距离 2.6e-3。
    #   sin²(Δφ/2) = (1−cosΔφ)/2 = s²/(2(1+√(1−s²))),  s² = sin²Δφ = cross²/(κ₁κ₂)²
    #   1+√(1−s²) = 1+|cosΔφ| = 1+|dot|/(κ₁κ₂)
    # ⇒ sinh u₁sinh u₂·sin²(Δθ) = cross²/(2·√((1−κ₁²)(1−κ₂²))·(κ₁κ₂+|dot|))
    dot = zx1 * zx2 + zy1 * zy2
    cross = zy1 * zx2 - zx1 * zy2
    kk = (k1c * k2c).clamp_min(1e-12)
    cos_dphi = (dot / kk).clamp(-1.0, 1.0)
    sin2_dphi = (cross * cross / (kk * kk)).clamp(0.0, 1.0)       # sin²Δφ（无相减）
    sin2_dtheta = torch.where(
        cos_dphi >= 0.0,
        sin2_dphi / (2.0 * (1.0 + cos_dphi).clamp_min(1e-12)),   # Δφ→0：防抵消分支
        0.5 * (1.0 - cos_dphi))                                   # Δφ→π：良态分支
    prod_sinh = kk / denom_k                                      # sinh u₁ sinh u₂
    term_th = prod_sinh * sin2_dtheta.clamp(0.0, 1.0)
    W = (term_u + term_th).clamp_min(0.0)
    d = math.sqrt(2.0) * torch.log1p(2.0 * W + 2.0 * torch.sqrt(W * (W + 1.0)))
    return torch.where(d < d0, torch.zeros_like(d), d)


def tr_based_shape_distance(a1, b1, c1, a2, b2, c2):
    """**参考实现**（只用于测试对拍）：τ = tr(Ĵ₁⁻¹Ĵ₂)，d = √2·arccosh(τ/2)。

    数学上与 `poincare_shape_distance` 恒等，但在 τ→2 与强各向异性下数值病态，
    故不作为生产实现（T1 用它与 eigh 参考双重对拍生产实现）。"""
    tau = c1 * a2 + a1 * c2 - 2.0 * b1 * b2
    u = (0.5 * tau - 1.0).clamp_min(0.0)
    return math.sqrt(2.0) * torch.log1p(u + torch.sqrt(u * (u + 2.0)))


def trace_scale_distance(jxx1, jxy1, jyy1, jxx2, jxy2, jyy2, log0=1e-6):
    """尺度分量 d_str = |log(tr_A / tr_B)|（良态；对辐射变化敏感，不抗辐射）。"""
    t1 = (jxx1 + jyy1).clamp_min(_EPS)
    t2 = (jxx2 + jyy2).clamp_min(_EPS)
    lr = (t1.log() - t2.log()).abs()
    return torch.where(lr < log0, torch.zeros_like(lr), lr)


def det_scale_distance(jxx1, jxy1, jyy1, jxx2, jxy2, jyy2):
    """命题 3 的尺度项 √2·|log(σ_A/σ_B)|，σ = √(det J)。**仅用于对拍/证明**：
    各向异性 κ→1 时 det 的相对误差会放大，故生产证据用 trace 版尺度。"""
    d1 = (jxx1 * jyy1 - jxy1 * jxy1).clamp_min(0.0).sqrt().clamp_min(_EPS)
    d2 = (jxx2 * jyy2 - jxy2 * jxy2).clamp_min(0.0).sqrt().clamp_min(_EPS)
    return math.sqrt(2.0) * (d1.log() - d2.log()).abs()


def coherence(jxx, jxy, jyy):
    """各向异性 κ = (λ₁−λ₂)/(λ₁+λ₂) = |z| ∈ [0,1]。"""
    return shape_disk_coords(jxx, jxy, jyy)[2]


def orientation_change(zx1, zy1, zx2, zy2, k1, k2):
    """朝向变化（相干性加权）：|sin(θ_A−θ_B)|·min(κ_A,κ_B)。

    sin(θ_A−θ_B) 由双角度坐标给出：sinΔθ = (y₁x₂ − x₁y₂)/(κ₁κ₂)（κ>0 时）。
    权重 min(κ) 抑制各向同性（朝向不可靠）处的噪声。
    """
    cross = (zy1 * zx2 - zx1 * zy2)
    denom = (k1 * k2).clamp_min(_EPS)
    sin_dtheta = (cross / denom).abs().clamp(0.0, 1.0)
    return sin_dtheta * torch.minimum(k1, k2)


# ---------------------------------------------------------------------------
# 3. 证据图
# ---------------------------------------------------------------------------
def spd_geodesic_evidence(pre, post, sigma_d=1.0, sigma_i=None, tr_floor=TR_FLOOR):
    """双时相图像 → 每尺度 5 通道几何变化证据（全分辨率）。

    pre/post: (B,C,H,W)。返回 (evidence (B,5,H,W), parts dict)
    parts 键：d_shape / d_str / kappa_a / kappa_b / ori / valid_a / valid_b / valid。
    """
    ja = structure_tensor(pre, sigma_d=sigma_d, sigma_i=sigma_i)
    jb = structure_tensor(post, sigma_d=sigma_d, sigma_i=sigma_i)
    za_x, za_y, ka, va = shape_disk_coords(*ja, tr_floor=tr_floor)
    zb_x, zb_y, kb, vb = shape_disk_coords(*jb, tr_floor=tr_floor)

    d_shape = poincare_shape_distance(za_x, za_y, ka, zb_x, zb_y, kb)
    valid = va * vb
    d_shape = d_shape * valid                    # 任一时相平坦 ⇒ 形状无定义 ⇒ 0
    d_str = trace_scale_distance(*ja, *jb)
    ori = orientation_change(za_x, za_y, zb_x, zb_y, ka, kb)
    ev = torch.cat([d_shape, d_str, ka, kb, ori], dim=1)
    parts = {"d_shape": d_shape, "d_str": d_str, "kappa_a": ka, "kappa_b": kb,
             "ori": ori, "valid": valid, "valid_a": va, "valid_b": vb}
    return ev, parts


# ---------------------------------------------------------------------------
# 4. 可部署模块（γ 零初始化门；构造不消耗全局 RNG）
# ---------------------------------------------------------------------------
STGEO_FORM = "stgeo_spd2_v1"
STGEO_SEED = 23


def _pool_evidence(ev, size, mode="max_shape"):
    """全分辨率证据 → 1/4（64²）。

    `max_shape`（默认）：**形状/朝向通道取 4×4 cell 内最大值**——1–15px 小目标与细边缘在
    平均池化下被稀释（1px 线 → 1/16 幅度），max 保底（与 Run7 HFAE 的
    `adaptive_max_pool2d` 保底理由一致）；尺度/κ 通道用平均池化（更平滑稳定）。
    """
    B, K, H, W = ev.shape
    out = F.adaptive_avg_pool2d(ev, size)
    if mode == "max_shape":
        s = int(round(H / size))
        if s > 1 and H % size == 0 and W % size == 0:
            take = [0, 4] if K >= 5 else [0]         # d_shape, ori
            mx = F.max_pool2d(ev[:, take], kernel_size=s, stride=s)
            out = torch.cat([mx, out[:, [i for i in range(K) if i not in take]]], dim=1)
    return out


class STGeoEvidence(nn.Module):
    """SPD(2)/H² 结构张量测地证据 → 1/4 尺度增量（**推理期真实算子**，非训练期专用）。

    `form`：
      · `inject`：γ 零初始化门控，增量加到 DCR 最小尺度输入（feats[0]，1/4），
        与 Run8 的 `dcr1` 注入点一致 ⇒ 可直接用已跑完的 `CTRL_R8` 作同期对照
        （epoch-0 逐位恒等由 dry-run 断言）。
      · `none`  ：关闭（等价于不加本模块）。

    参数：3 尺度 × 5 通道 → 1×1 → mid → DW3×3 → 1×1 → out_dim，加 1 个 γ。
    """
    FORMS = ("none", "inject")

    def __init__(self, in_ch=3, sigmas=(1.0, 2.0, 4.0), out_dim=192, mid=32,
                 form="inject", tr_floor=TR_FLOOR, pool_mode="max_shape",
                 init_seed=STGEO_SEED):
        super().__init__()
        assert form in self.FORMS, form
        self.in_ch = int(in_ch)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.out_dim = int(out_dim)
        self.mid = int(mid)
        self.form = form
        self.tr_floor = float(tr_floor)
        self.pool_mode = pool_mode
        self.k_per_scale = 5
        self.form_tag = STGEO_FORM if form != "none" else None

        with torch.random.fork_rng(devices=[]):   # 不推进全局 RNG
            torch.manual_seed(int(init_seed))
            self.proj = nn.Conv2d(self.k_per_scale * len(self.sigmas), self.mid, 1,
                                  bias=False)
            self.dw = nn.Conv2d(self.mid, self.mid, 3, 1, 1, groups=self.mid, bias=False)
            self.bn = nn.BatchNorm2d(self.mid)
            self.act = nn.SiLU()
            self.out = nn.Conv2d(self.mid, self.out_dim, 1, bias=False)
            self.apply(self._init_weights)
        # γ 零初始化：**必须在 apply 之后**（HFAE 首版踩过被 init 覆盖的坑）
        self.gamma = nn.Parameter(torch.zeros(1))
        self._diag = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, pre, post, ref_hw):
        """pre/post: (B,3,H,W)；ref_hw: 目标尺度 (h,w)（=1/4 的 64×64）。"""
        if self.form == "none":
            return None, {}
        evs, diags = [], {}
        for s in self.sigmas:
            ev, parts = spd_geodesic_evidence(pre, post, sigma_d=s,
                                              tr_floor=self.tr_floor)
            ev = _pool_evidence(ev, ref_hw[0], mode=self.pool_mode)
            evs.append(ev)
            with torch.no_grad():
                ds = parts["d_shape"]
                diags[f"sigma{s:g}"] = {
                    "d_shape_mean": float(ds.mean()),
                    "d_shape_max": float(ds.amax()),
                    "d_str_mean": float(parts["d_str"].mean()),
                    "kappa_mean": float(parts["kappa_a"].mean()),
                    "valid_frac": float(parts["valid"].mean()),
                }
        h = torch.cat(evs, dim=1)
        z = self.act(self.bn(self.dw(self.proj(h))))
        z = self.out(z)
        self._diag = diags
        return self.gamma * z, diags

    # ------------------------------------------------------------------
    def diag(self):
        return dict(self._diag)

    def param_report(self):
        n = sum(p.numel() for p in self.parameters())
        return {"form": self.form_tag, "sigmas": list(self.sigmas), "mid": self.mid,
                "out_dim": self.out_dim, "k_per_scale": self.k_per_scale,
                "params": int(n),
                "gamma_init_zero": bool(float(self.gamma.detach()) == 0.0),
                "gamma": float(self.gamma.detach())}
