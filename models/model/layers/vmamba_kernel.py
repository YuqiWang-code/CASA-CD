"""Run8 selective-scan 后端适配层（VMamba v2 四向扫描用）。

## 为什么需要这一层（2026-10-15 服务器实测，见交接/探针记录）

RSML-3 上同时存在**两套** selective-scan CUDA 扩展，只有一套能用：

| 环境 | `selective_scan_cuda` | 状态 |
|---|---|---|
| `casacd`（本项目） | `site-packages/selective_scan_cuda...so`（10-03 编译） | **ABI 坏**：`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationE...` |
| `mamba_base`（SSM 模板） | `site-packages/selective_scan_cuda...so`（09-19 编译） | **可用**：fwd+bwd 实测 finite（B8/D64/L4096，0.037 ms/call） |

两边的 torch 都是 **2.14.0+cu132 / py3.10 / sm_120**，所以 `mamba_base` 的扩展在
`casacd` 进程里**可以直接用**——阻碍只是 Python 的导入顺序（`casacd` 的陈旧 `.so`
先被找到）。因此本模块的做法是：

1. 把 `mamba_base` 的 `site-packages` 插到 `sys.path` 最前（可用 `CASA_MAMBA_SITE`
   环境变量覆盖），再 `import selective_scan_cuda`；
2. 顺手把 `mamba_ssm`（提供官方 `selective_scan_fn`）也走同一条路导入；
3. 都失败时**显式抛错**，绝不静默退回一个语义不同的实现。

**不做的事**：不复制、不覆盖、不删除任何 `.so`；不修改 `casacd` 里的任何既有文件。
"""
import os
import sys

# mamba_base site-packages（可用环境变量覆盖，便于将来环境搬迁）
_MAMBA_SITE = os.environ.get(
    "CASA_MAMBA_SITE", "/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages"
)

_SITE_INSERTED = False


def ensure_mamba_site_path():
    """把 mamba_base 的 site-packages 提到 sys.path 最前（幂等）。返回是否成功。"""
    global _SITE_INSERTED
    if not os.path.isdir(_MAMBA_SITE):
        return False
    if not _SITE_INSERTED:
        while _MAMBA_SITE in sys.path:
            sys.path.remove(_MAMBA_SITE)
        sys.path.insert(0, _MAMBA_SITE)
        _SITE_INSERTED = True
    return True


_STATUS = {}


def load_selective_scan_cuda():
    """返回可用的 `selective_scan_cuda` 模块（提供 `fwd`/`bwd`）。失败抛 RuntimeError。"""
    if "ssc" in _STATUS:
        mod = _STATUS["ssc"]
        if mod is None:
            raise RuntimeError(_STATUS["err"])
        return mod
    ensure_mamba_site_path()
    try:
        import selective_scan_cuda as _ssc  # type: ignore
    except Exception as e:  # noqa: BLE001
        _STATUS["ssc"] = None
        _STATUS["err"] = (
            f"[SCAN-BACKEND] 无法导入可用的 selective_scan_cuda：{type(e).__name__}: {e}\n"
            f"  已尝试把 {_MAMBA_SITE} 插到 sys.path 最前。\n"
            f"  排查：`ls {_MAMBA_SITE}/selective_scan_cuda*.so`；"
            f"`python -c \"import sys;sys.path.insert(0,'{_MAMBA_SITE}');import selective_scan_cuda\"`"
        )
        raise RuntimeError(_STATUS["err"]) from e
    _STATUS["ssc"] = _ssc
    return _ssc


def load_mamba_selective_scan_fn():
    """返回 mamba_ssm 官方的 `selective_scan_fn`（非 CUDA 扩展的 python 包装层）。"""
    if "fn" in _STATUS:
        fn = _STATUS["fn"]
        if fn is None:
            raise RuntimeError(_STATUS["fn_err"])
        return fn
    ensure_mamba_site_path()
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
    except Exception as e:  # noqa: BLE001
        _STATUS["fn"] = None
        _STATUS["fn_err"] = f"[SCAN-BACKEND] mamba_ssm.selective_scan_fn 不可用：{type(e).__name__}: {e}"
        raise RuntimeError(_STATUS["fn_err"]) from e
    _STATUS["fn"] = selective_scan_fn
    return selective_scan_fn


def load_selective_scan_oflex():
    """返回 `selective_scan_cuda_oflex`（TinyViM/Run1–Run7 生产同款，`casacd` 自带）。

    这是**本项目已验证**的路径（Run1–Run7 全部训练都走它，四向扫描 max_abs<2e-5 对拍过
    torch 参考实现），因此 VMamba 也优先用它，而不是引入未在本项目验证过的第二条内核。
    """
    if "oflex" in _STATUS:
        mod = _STATUS["oflex"]
        if mod is None:
            raise RuntimeError(_STATUS["oflex_err"])
        return mod
    ensure_mamba_site_path()
    try:
        import selective_scan_cuda_oflex as _oflex  # type: ignore
    except Exception as e:  # noqa: BLE001
        _STATUS["oflex"] = None
        _STATUS["oflex_err"] = f"[SCAN-BACKEND] selective_scan_cuda_oflex 不可用：{type(e).__name__}: {e}"
        raise RuntimeError(_STATUS["oflex_err"]) from e
    _STATUS["oflex"] = _oflex
    return _oflex


def backend_report():
    """只读诊断：一条字符串，供日志/预注册记录。"""
    out = [f"mamba_site={_MAMBA_SITE}", f"exists={os.path.isdir(_MAMBA_SITE)}"]
    try:
        oflex = load_selective_scan_oflex()
        out.append(f"selective_scan_cuda_oflex=OK({getattr(oflex, '__file__', '?')})")
    except Exception as e:  # noqa: BLE001
        out.append(f"selective_scan_cuda_oflex=FAIL({type(e).__name__})")
    try:
        ssc = load_selective_scan_cuda()
        out.append(f"selective_scan_cuda=OK({getattr(ssc, '__file__', '?')})")
        out.append(f"symbols={[a for a in dir(ssc) if not a.startswith('_')]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"selective_scan_cuda=FAIL({type(e).__name__})")
    try:
        load_mamba_selective_scan_fn()
        out.append("mamba_ssm.selective_scan_fn=OK")
    except Exception as e:  # noqa: BLE001
        out.append(f"mamba_ssm.selective_scan_fn=FAIL({type(e).__name__})")
    return " | ".join(out)
