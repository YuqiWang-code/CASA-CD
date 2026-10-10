"""CASA-CD TViM 小目标瓶颈诊断 —— 共享基础模块（Run-Diag / Diag1）。

本模块只提供**只读**诊断所需的基础设施，不改变任何训练/推理行为：
  - 路径、变体配置表、SHA256 与 manifest 记录；
  - 严格 TEST RESULTS 日志解析（复用 analyse/extract_metrics_to_excel.py 的函数，不运行其 main）；
  - checkpoint 选择（唯一 best 文件校验，禁止按字典序猜）；
  - 模型构造 + state_dict 严格缺失/多余键核对；
  - 与 models/train.py / models/model/metric_tool.py 完全同口径的混淆矩阵与六指标；
  - DataLoader（复用 models/dataset + Transforms 的 eval 变换链）；
  - image-level bootstrap 置信区间；
  - gate.json / JSON 落盘工具。

纪律：不 train()、不改 BN、不改 optimizer、不写回任何历史文件。
"""
import os
import sys
import json
import glob
import hashlib
import platform
import datetime

import numpy as np

_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

# --------------------------------------------------------------------------------------
# 路径（服务器）；本地只用于读取代码，不含 .pth 时脚本会自行报错退出
# --------------------------------------------------------------------------------------
PROJECT = os.environ.get("CASA_PROJECT", "/home/yqwang/projects/CASA-CD")
DATA_ROOT = os.environ.get("CASA_DATA", "/share_datasets/CD")
CKPT_ROOT = os.environ.get("CASA_CKPT", "/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM")
OUTPUT_ROOT = os.environ.get("CASA_OUTPUT", "/home/yqwang/outputs/CASA-CD")
PRETRAIN_TINYVIM = os.path.join(PROJECT, "pretrained_weight", "tinyvim_s_1000e.pth")
DIAG_BASE = os.path.join(OUTPUT_ROOT, "diagnostics", "TViM-TinyLoss-Diag1")

DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]

# 变体 → CASATViMSTRNet 构造实参（与 train_scripts 下的实参一致；M1 已逐项核对 Run1 脚本）
VARIANT_CFG = {
    "M1_FULL":      dict(caacp=True,  rep_mode="full",  caacp_score_mode="rank", frh=False,
                         caacp_residual_mode="current",   fs_tar=False),
    "A1_CAACP":     dict(caacp=True,  rep_mode="plain", caacp_score_mode="rank", frh=False,
                         caacp_residual_mode="current",   fs_tar=False),
    "A2_STR":       dict(caacp=False, rep_mode="full",  caacp_score_mode="rank", frh=False,
                         caacp_residual_mode="current",   fs_tar=False),
    "A0_TVIM_PLAIN": dict(caacp=False, rep_mode="plain", caacp_score_mode="rank", frh=False,
                          caacp_residual_mode="current",  fs_tar=False),
    "E1_CP_CAACP":  dict(caacp=True,  rep_mode="full",  caacp_score_mode="cp",   frh=False,
                         caacp_residual_mode="current",   fs_tar=False),
    "E2_FRH":       dict(caacp=True,  rep_mode="full",  caacp_score_mode="rank", frh=True,
                         caacp_residual_mode="current",   fs_tar=False),
    "E3_CP_FRH":    dict(caacp=True,  rep_mode="full",  caacp_score_mode="cp",   frh=True,
                         caacp_residual_mode="current",   fs_tar=False),
    "E4_RA_CAACP":  dict(caacp=True,  rep_mode="full",  caacp_score_mode="rank", frh=False,
                         caacp_residual_mode="avg_anchor", fs_tar=False),
    "E5_FS_TAR":    dict(caacp=True,  rep_mode="full",  caacp_score_mode="rank", frh=False,
                         caacp_residual_mode="current",   fs_tar=True),
}

# eval 变换的 6 通道 mean/std（train.py 默认：ImageNet 统计 ×2，BGR 顺序）
MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text_lines(path):
    """对文件行内容做 SHA256（行末统一 \n，避免 CRLF 差异）。"""
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            h.update(line.rstrip("\r\n").encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def env_info():
    import torch
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "platform": platform.platform(),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = None
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        info["sklearn"] = None
    try:
        import cv2
        info["cv2"] = cv2.__version__
    except Exception:
        info["cv2"] = None
    return info


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_gate(out_dir, stage, status, checks, extra=None):
    """写 gate.json。status ∈ {PASS, WARN, FAIL, BLOCKED, NOT_RUN}。"""
    assert status in ("PASS", "WARN", "FAIL", "BLOCKED", "NOT_RUN"), status
    payload = {
        "stage": stage,
        "status": status,
        "checks": checks,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    write_json(os.path.join(out_dir, "gate.json"), payload)
    return payload


# --------------------------------------------------------------------------------------
# 严格 TEST RESULTS 日志解析
# --------------------------------------------------------------------------------------
def parse_test_block(log_path):
    """返回 (block_text, info_dict, lineno_range) —— 只读取最后一个完整 TEST 区块。

    解析逻辑与 `analyse/extract_metrics_to_excel.py` 的 `extract_test_block` / `parse_block`
    **逐字一致**（见下方 `_extract_test_block` / `_parse_block`），在此重新实现是为了避免
    诊断环境依赖 openpyxl（服务器 casacd 未安装，且本轮不修改主环境包版本）。
    一致性由 `analyse/tests/test_tvim_log_parser.py` 与官方实现直接对拍验证。
    """
    if not os.path.isfile(log_path):
        return None, None, None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    text = "".join(lines)
    block = _extract_test_block(text)
    if block is None:
        return None, None, None
    info = _parse_block(block)
    # 定位行号范围（便于报告溯源）
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "=== TEST RESULTS ==="]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "=== END TEST RESULTS ==="]
    span = (starts[-1] + 1, ends[-1] + 1) if starts and ends else None
    # 额外字段
    import re
    for key, pat in (("BEST_F1", r"\[BEST-F1\]\s+([0-9.]+)"),
                     ("PRETRAIN_SHA256", r"\[PRETRAIN-SHA256\]\s+(\S+)"),
                     ("CAACP_BETA", r"\[CAACP-BETA\]\s+([0-9.eE+-]+)"),
                     ("REPARAM_MAX_ABS", r"\[REPARAM-MAX-ABS-ERROR\]\s+([0-9.eE+-]+)"),
                     ("REPARAM_DISAGREE", r"\[REPARAM-ARGMAX-DISAGREE\]\s+([0-9.eE+-]+)"),
                     ("REPARAM_REAL_MAX_ABS", r"\[REPARAM-REAL-MAX-ABS-ERROR\]\s+([0-9.eE+-]+)"),
                     ("REPARAM_REAL_DISAGREE", r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s+([0-9.eE+-]+)")):
        m = re.search(pat, block)
        if m:
            info[key] = m.group(1)
    return block, info, span


def _extract_test_block(text):
    """与 analyse/extract_metrics_to_excel.py::extract_test_block 逐字一致。"""
    import re
    blocks = re.findall(r"=== TEST RESULTS ===\n(.*?)\n=== END TEST RESULTS ===", text, re.S)
    return blocks[-1] if blocks else None


def _parse_block(block):
    """与 analyse/extract_metrics_to_excel.py::parse_block 逐字一致（仅取诊断需要的键）。"""
    import re
    d = {}
    m = re.search(r"Recall=([0-9.]+)\s*\|\s*Precision=([0-9.]+)\s*\|\s*OA=([0-9.]+)\s*\|\s*F1=([0-9.]+)\s*\|\s*IoU=([0-9.]+)\s*\|\s*Kappa=([0-9.]+)", block)
    if m:
        d["Recall"], d["Precision"], d["OA"], d["F1"], d["IoU"], d["Kappa"] = [
            float(x) for x in m.groups()
        ]

    def _num(pattern):
        mm = re.search(pattern + r"\s+([0-9.]+)", block)
        return float(mm.group(1)) if mm else None

    d["Params(M)"] = _num(r"\[EFFECTIVE-PARAMS\]")
    if d["Params(M)"] is None:
        mm = re.search(r"\[DEPLOY-PARAMS\]\s+.*effective=([0-9.]+)", block)
        d["Params(M)"] = float(mm.group(1)) if mm else None
    if d["Params(M)"] is None:
        d["Params(M)"] = _num(r"\[(?:TOTAL-PARAMS|PARAMS|DEPLOY-PARAMS|TOTAL-TRAIN-GRAPH-PARAMS)\]")
    d["Trainable(M)"] = _num(r"\[TRAINABLE-PARAMS\]")
    if d["Trainable(M)"] is None:
        mm = re.search(r"\[DEPLOY-PARAMS\]\s+.*trainable=([0-9.]+)", block)
        d["Trainable(M)"] = float(mm.group(1)) if mm else None
    d["FLOPs(G)"] = _num(r"\[(?:FLOPS|DEPLOY-FLOPS)\]")
    mm = re.search(r"\[MODE\]\s+(\S+)", block)
    d["Mode"] = mm.group(1) if mm else None
    if d["Mode"] is None:
        mm = re.search(r"\[ARCH\]\s+(\S+)", block)
        d["Mode"] = mm.group(1) if mm else None
    return d


def log_path_for(run, variant, dataset):
    return os.path.join(OUTPUT_ROOT, "CASA-TViM", run, variant, dataset, "train_log.txt")


# --------------------------------------------------------------------------------------
# checkpoint 选择（唯一 best 文件）
# --------------------------------------------------------------------------------------
def pick_best_ckpt(run, variant, dataset, ckpt_root=None):
    """返回 (path, meta)。多个 best 文件 → 抛错（禁止按字典序猜），交由用户确认。"""
    root = ckpt_root or CKPT_ROOT
    d = os.path.join(root, run, variant, dataset)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"checkpoint dir not found: {d}")
    cands = sorted(glob.glob(os.path.join(d, "best_F1=*.pth")))
    meta = {"ckpt_dir": d, "candidates": [os.path.basename(c) for c in cands]}
    if len(cands) == 0:
        raise FileNotFoundError(f"[BLOCKED-CKPT] no best_F1=*.pth in {d}")
    if len(cands) > 1:
        raise RuntimeError(f"[BLOCKED-CKPT] ambiguous best checkpoints in {d}: {meta['candidates']}")
    path = cands[0]
    arch_json = os.path.join(d, "arch.json")
    manifest = os.path.join(d, "run_manifest.json")
    meta["path"] = path
    meta["sha256"] = sha256_file(path)
    meta["arch"] = read_json(arch_json) if os.path.isfile(arch_json) else None
    meta["run_manifest"] = read_json(manifest) if os.path.isfile(manifest) else None
    return path, meta


def verify_arch_sidecar(variant, meta, dataset):
    """arch.json 与 VARIANT_CFG 一致性核对（防止静默加载错结构）。

    Run1 时期的 arch.json 只有 {arch, backbone, caacp, rep_mode, str_dim}（5 字段）；
    Run2/Run3 起才有 caacp_score_mode / caacp_residual_mode / frh / fs_tar。
    缺失字段按**当时的默认值**补齐后比较，并把缺失情况记入 detail（与 models/eval.py 同策略）。
    """
    arch = meta.get("arch")
    if arch is None:
        return "WARN", "no arch.json"
    cfg = VARIANT_CFG[variant]
    defaults = {"caacp_score_mode": "rank", "caacp_residual_mode": "current",
                "frh": 0, "fs_tar": 0}
    exp = {"caacp": int(cfg["caacp"]), "caacp_score_mode": cfg["caacp_score_mode"],
           "frh": int(cfg["frh"])}
    if "caacp_residual_mode" in arch:
        exp["caacp_residual_mode"] = cfg["caacp_residual_mode"]
    if "fs_tar" in arch:
        exp["fs_tar"] = int(cfg["fs_tar"])
    missing = [k for k in ("caacp_score_mode", "caacp_residual_mode", "frh", "fs_tar")
               if k not in arch]
    bad = {}
    for k, v in exp.items():
        got = arch[k] if k in arch else defaults.get(k)
        if got != v:
            bad[k] = {"arch": got, "expected": v}
    detail = {"arch_json": arch, "legacy_fields_missing": missing,
              "checked": exp, "mismatch": bad}
    if bad:
        return "FAIL", detail
    return "PASS", detail


# --------------------------------------------------------------------------------------
# 模型构造 / 加载
# --------------------------------------------------------------------------------------
def build_model(variant, device="cuda:0", ckpt_path=None, pretrain=None, strict=True):
    """构造 CASATViMSTRNet（train 图）并加载 checkpoint，严格核对键。"""
    import torch
    from model.casa_tvim_str_net import CASATViMSTRNet

    cfg = VARIANT_CFG[variant]
    pretrain = pretrain or PRETRAIN_TINYVIM
    model = CASATViMSTRNet(
        pretrain,
        caacp=cfg["caacp"], rep_mode=cfg["rep_mode"], str_dim=96,
        caacp_score_mode=cfg["caacp_score_mode"], frh=cfg["frh"],
        caacp_residual_mode=cfg["caacp_residual_mode"], fs_tar=cfg["fs_tar"],
    ).float().to(device)
    info = {"variant": variant, "cfg": cfg, "strict": strict}
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        info["missing_keys"] = list(missing)
        info["unexpected_keys"] = list(unexpected)
        if strict and (len(missing) or len(unexpected)):
            raise RuntimeError(f"[D0] state_dict mismatch: missing={list(missing)[:5]} "
                               f"unexpected={list(unexpected)[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info


def set_eval_numerics():
    """与 train.py / eval.py 的 casa_tvim_str TEST 协议一致：TF32 off + cudnn deterministic。"""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------
# DataLoader（与 train.py eval 链一致：Normalize → Scale → ToTensor）
# --------------------------------------------------------------------------------------
def make_loader(dataset, list_name="test", batch_size=16, num_workers=4, size=256,
                mean=None, std=None, shuffle=False):
    import torch
    from dataset import dataset as myDataLoader
    from dataset import Transforms as myTransforms

    root = os.path.join(DATA_ROOT, dataset)
    list_path = os.path.join(root, "list", f"{list_name}.txt")
    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean or MEAN, std=std or STD),
        myTransforms.Scale(size, size),
        myTransforms.ToTensor(),
    ])
    ds = myDataLoader.Dataset(file_root=root, list_path=list_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=False)
    return loader, list_path


def read_list_names(list_path):
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.strip() for ln in f if ln.strip()]


# --------------------------------------------------------------------------------------
# 像素级混淆矩阵 / 六指标（与 models/model/metric_tool.py 完全同口径）
# --------------------------------------------------------------------------------------
EPS32 = float(np.finfo(np.float32).eps)


def cm_to_counts(confusion):
    """confusion: 2x2 numpy (rows=gt, cols=pred)。返回 tp, fp, fn, tn。"""
    c = np.asarray(confusion, dtype=np.float64)
    tp = c[1, 1]
    fn = c[1, 0]
    fp = c[0, 1]
    tn = c[0, 0]
    return tp, fp, fn, tn


def counts_to_scores(tp, fp, fn, tn):
    """严格复刻 metric_tool.cm2score（含 float32 eps 与 Kappa 定义）。"""
    oa = (tp + tn) / (tp + fn + fp + tn + EPS32)
    recall = tp / (tp + fn + EPS32)
    precision = tp / (tp + fp + EPS32)
    f1 = 2 * recall * precision / (recall + precision + EPS32)
    iou = tp / (tp + fp + fn + EPS32)
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    kappa = (oa - pre) / (1 - pre)
    return {"recall": float(recall), "precision": float(precision), "OA": float(oa),
            "F1": float(f1), "IoU": float(iou), "Kappa": float(kappa), "Pre": float(pre)}


def confusion_from_bool(pred_bool, gt_bool):
    """pred/gt 为同形状 bool 数组，返回 2x2 int64 混淆矩阵（与 bincount 等价）。"""
    p = np.asarray(pred_bool).astype(bool).ravel()
    g = np.asarray(gt_bool).astype(bool).ravel()
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


# --------------------------------------------------------------------------------------
# image-level bootstrap（配对差异 95% CI）
# --------------------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=1000, seed=16, alpha=0.05):
    """values: 每张图一个统计量（或配对差异）。返回 dict(mean, lo, hi, n)。"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, v.size, v.size)
        means[i] = v[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_bootstrap_ci(a, b, n_boot=1000, seed=16, alpha=0.05):
    """配对差异 a-b 的 image-level bootstrap CI（bootstrap 抽样单位 = 图像）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    return bootstrap_ci(a - b, n_boot=n_boot, seed=seed, alpha=alpha)


# --------------------------------------------------------------------------------------
# 代码身份（服务器无 .git，用文件 SHA256 记录）
# --------------------------------------------------------------------------------------
DIAG_CODE_FILES = [
    "models/model/casa_tvim_str_net.py",
    "models/model/tinyvim_s_slim.py",
    "models/model/layers/caacp_ss2d.py",
    "models/model/layers/ss2d.py",
    "models/model/str_tar.py",
    "models/model/str_dcr.py",
    "models/model/str_reparam.py",
    "models/model/metric_tool.py",
    "models/dataset/dataset.py",
    "models/dataset/Transforms.py",
    "models/train.py",
    "models/eval.py",
]


def code_identity(project=PROJECT):
    """返回诊断相关源码的 SHA256 表。服务器工作副本不是 git 仓库 → 以文件哈希为准。"""
    out = {"project": project, "files": {}, "git": None}
    for rel in DIAG_CODE_FILES:
        p = os.path.join(project, rel)
        out["files"][rel] = sha256_file(p) if os.path.isfile(p) else None
    git_head = os.path.join(project, ".git", "HEAD")
    if os.path.isfile(git_head):
        out["git"] = open(git_head, encoding="utf-8").read().strip()
    return out


# ======================================================================================
# D0：环境/权重/数据/预测对拍审计
# ======================================================================================
def _sum_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def _deploy_flops(model, size=256):
    """fvcore 口径的部署图 FLOPs（与 models/smoke_test.py::measure_flops 同款）。"""
    import torch
    from fvcore.nn import flop_count
    model.eval()
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    label = torch.zeros(1, 1, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post, label))
    return float(sum(counts.values())), sorted(unsupported.keys())


def run_audit(args):
    """D0 审计：复算 TEST 指标 + 折叠等价性 + manifest。写 out_dir/{test_result,fold_equivalence,manifest,gate}.json。"""
    import torch
    import copy
    from tvim_object_metrics import label_components, bin_index, AREA_GROUPS

    out_dir = ensure_dir(args.out_dir)
    device = args.device
    set_eval_numerics()

    # ---------- 代码 / 环境 / 数据身份
    identity = code_identity()
    env = env_info()
    root = os.path.join(DATA_ROOT, args.dataset)
    list_path = os.path.join(root, "list", "test.txt")
    names = read_list_names(list_path)
    ds_manifest = {
        "dataset": args.dataset,
        "dataset_root": root,
        "test_list": list_path,
        "test_list_sha256": sha256_text_lines(list_path),
        "n_images": len(names),
        "index_to_name": names,
    }

    # ---------- checkpoint
    ckpt_path, ckpt_meta = pick_best_ckpt(args.run, args.variant, args.dataset)
    sidecar_status, sidecar_detail = verify_arch_sidecar(args.variant, ckpt_meta, args.dataset)
    model, build_info = build_model(args.variant, device=device, ckpt_path=ckpt_path)

    # ---------- 原始日志 TEST 区块
    log_path = log_path_for(args.run, args.variant, args.dataset)
    block, logged, span = parse_test_block(log_path)

    # ---------- 全测试集复算
    loader, _ = make_loader(args.dataset, "test", batch_size=args.batch_size,
                            num_workers=args.num_workers)
    tp = fp = fn = tn = 0
    gt_cc_by_size = {g: 0 for g in AREA_GROUPS}
    empty_mask = 0
    n_seen = 0
    nonfinite = 0
    per_image = []
    idx = 0
    with torch.no_grad():
        for img, target in loader:
            pre = img[:, 0:3].to(device).float()
            post = img[:, 3:6].to(device).float()
            tgt = target.to(device).float()
            out = model(pre, post, tgt)
            if not torch.isfinite(out).all():
                nonfinite += int((~torch.isfinite(out)).sum().item())
            pred_b = (out > 0.5)
            gt_b = (tgt > 0.5)
            pr = pred_b.cpu().numpy().astype(bool)
            gb = gt_b.cpu().numpy().astype(bool)
            for i in range(pr.shape[0]):
                p2 = pr[i, 0]
                g2 = gb[i, 0]
                tp += int(np.count_nonzero(p2 & g2))
                fp += int(np.count_nonzero(p2 & ~g2))
                fn += int(np.count_nonzero(~p2 & g2))
                tn += int(np.count_nonzero(~p2 & ~g2))
                if not g2.any():
                    empty_mask += 1
                lab, n = label_components(g2, 4)
                if n:
                    cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                    for a in cnt:
                        gt_cc_by_size[bin_index(int(a))] += 1
                per_image.append({"sample_name": names[idx] if idx < len(names) else None,
                                  "tp": int(np.count_nonzero(p2 & g2)),
                                  "fp": int(np.count_nonzero(p2 & ~g2)),
                                  "fn": int(np.count_nonzero(~p2 & g2)),
                                  "tn": int(np.count_nonzero(~p2 & ~g2))})
                idx += 1
                n_seen += 1

    repro = counts_to_scores(tp, fp, fn, tn)
    cmp_keys = [("Recall", "recall"), ("Precision", "precision"), ("OA", "OA"),
                ("F1", "F1"), ("IoU", "IoU"), ("Kappa", "Kappa")]
    deltas = {}
    for lk, mk in cmp_keys:
        if logged and logged.get(lk) is not None:
            deltas[lk] = abs(float(logged[lk]) - repro[mk])
    max_delta = max(deltas.values()) if deltas else None
    repro_ok = bool(max_delta is not None and max_delta <= args.tol)

    test_result = {
        "run": args.run, "variant": args.variant, "dataset": args.dataset,
        "checkpoint": ckpt_meta, "arch_sidecar_status": sidecar_status,
        "arch_sidecar_detail": sidecar_detail, "build_info": build_info,
        "log_path": log_path, "test_block_lines": span,
        "logged": {k: logged.get(k) for k, _ in cmp_keys} if logged else None,
        "reproduced": {mk: repro[mk] for _, mk in cmp_keys},
        "reproduced_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "deltas": deltas, "max_abs_delta": max_delta, "tolerance": args.tol,
        "reproduce_ok": repro_ok,
        "n_images_seen": n_seen, "nonfinite_values": nonfinite,
        "gt_cc_by_size": gt_cc_by_size, "empty_mask_count": empty_mask,
        "identity": identity, "env": env, "dataset_manifest": ds_manifest,
        "params": {
            "train_graph_total": _sum_params(model),
            "train_graph_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        },
        "logged_extra": {k: v for k, v in (logged or {}).items()
                         if k in ("BEST_F1", "PRETRAIN_SHA256", "CAACP_BETA", "DEPLOY-PARAMS",
                                  "REPARAM_MAX_ABS", "REPARAM_DISAGREE",
                                  "REPARAM_REAL_MAX_ABS", "REPARAM_REAL_DISAGREE", "FLOPs(G)",
                                  "Params(M)", "Trainable(M)")},
        "per_image_counts": per_image if args.write_per_image else None,
    }

    # ---------- 折叠等价性（train 图 vs deploy 图）
    torch.manual_seed(16)
    pre_fix = torch.randn(args.fold_batch, 3, 256, 256, device=device)
    post_fix = torch.randn(args.fold_batch, 3, 256, 256, device=device)
    real_loader, _ = make_loader(args.dataset, "test", batch_size=args.fold_batch,
                                 num_workers=args.num_workers)
    img_r, _ = next(iter(real_loader))
    pre_r = img_r[:, 0:3].to(device).float()
    post_r = img_r[:, 3:6].to(device).float()

    model.eval()
    with torch.no_grad():
        y_train = model(pre_fix, post_fix)
        y_train_real = model(pre_r, post_r)
    model_d = copy.deepcopy(model)
    model_d.eval()
    model_d.switch_to_deploy()
    with torch.no_grad():
        y_deploy = model_d(pre_fix, post_fix)
        y_deploy_real = model_d(pre_r, post_r)

    def _fold_stats(a, b):
        err = (a - b).abs().max().item()
        dis = ((a > 0.5) != (b > 0.5)).float().mean().item()
        n_el = a.numel()
        return {"max_abs_error": err, "binary_disagreement": dis, "n_elements": int(n_el),
                "n_flipped": int((((a > 0.5) != (b > 0.5))).sum().item())}

    fold = {
        "random_batch": _fold_stats(y_train, y_deploy),
        "real_batch": _fold_stats(y_train_real, y_deploy_real),
        "deploy_params_total": _sum_params(model_d),
        "train_graph_params_total": _sum_params(model),
        "note": "disagreement 分母 = N×H×W（全部元素）；hard gate = 0",
    }
    try:
        flops, unsup = _deploy_flops(model_d, size=256)
        fold["deploy_flops_G"] = flops
        fold["deploy_flops_unsupported_ops"] = unsup
        fold["deploy_flops_unsupported_count"] = len(unsup)
    except Exception as e:  # noqa: BLE001
        fold["deploy_flops_error"] = f"{type(e).__name__}: {e}"

    write_json(os.path.join(out_dir, "manifest.json"), {
        "identity": identity, "env": env, "dataset": ds_manifest,
        "checkpoint": ckpt_meta, "variant_cfg": VARIANT_CFG[args.variant],
    })
    write_json(os.path.join(out_dir, "test_result.json"), test_result)
    write_json(os.path.join(out_dir, "fold_equivalence.json"), fold)

    checks = {
        "checkpoint_unique": len(ckpt_meta["candidates"]) == 1,
        "state_dict_strict": (not build_info.get("missing_keys")) and (not build_info.get("unexpected_keys")),
        "arch_sidecar": sidecar_status,
        "log_test_block_found": block is not None,
        "reproduce_within_tol": repro_ok,
        "reproduce_max_abs_delta": max_delta,
        "fold_random_disagreement_zero": fold["random_batch"]["binary_disagreement"] == 0.0,
        "fold_real_disagreement_zero": fold["real_batch"]["binary_disagreement"] == 0.0,
        "no_nonfinite": nonfinite == 0,
        "n_images_seen_full": n_seen == len(names),
    }
    ok = (checks["checkpoint_unique"] and checks["state_dict_strict"]
          and sidecar_status != "FAIL" and checks["log_test_block_found"]
          and repro_ok and checks["fold_random_disagreement_zero"]
          and checks["fold_real_disagreement_zero"] and checks["no_nonfinite"]
          and checks["n_images_seen_full"])
    gate = write_gate(out_dir, "D0-REPRO", "PASS" if ok else "FAIL", checks)
    print(f"[D0] {args.variant}/{args.dataset}: gate={gate['status']} "
          f"max_delta={max_delta} fold_dis(rand)={fold['random_batch']['binary_disagreement']} "
          f"fold_dis(real)={fold['real_batch']['binary_disagreement']}", flush=True)
    return gate, test_result, fold


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CASA-CD TViM 诊断共享工具（D0 审计 / P0 协议落盘）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="D0：复现 TEST 指标 + 折叠等价性 + manifest")
    a.add_argument("--dataset", default="SYSU-CD-256")
    a.add_argument("--run", default="Run1")
    a.add_argument("--variant", default="M1_FULL", choices=sorted(VARIANT_CFG))
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--fold-batch", type=int, default=16)
    a.add_argument("--num-workers", type=int, default=4)
    a.add_argument("--tol", type=float, default=1e-4)
    a.add_argument("--write-per-image", action="store_true")
    b = sub.add_parser("protocol", help="P0：落盘 metrics_protocol.json（指标口径协议）")
    b.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "audit":
        run_audit(args)
    elif args.cmd == "protocol":
        from tvim_object_metrics import METRICS_PROTOCOL
        ensure_dir(args.out_dir)
        path = write_json(os.path.join(args.out_dir, "metrics_protocol.json"), METRICS_PROTOCOL)
        print(f"[P0] wrote {path}")


if __name__ == "__main__":
    main()
