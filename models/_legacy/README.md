# models/_legacy/ — past-run 的一次性脚本（**不属当前主线**）

这里是从 `models/` 顶层移出来的历史脚本，只为**备查与复现**，不参与当前 Run9 的任何流程。

| 文件 | 原用途 | 现状 |
|---|---|---|
| `run4_dry_run.py` | Run4 真实数据 dry-run（6 道门 + 折叠刀锋定位） | Run4 已判 `MECHANISM-FAIL` |
| `run7_dry_run.py` | Run7 HFAE 真实数据 dry-run | Run7 已收官（`MECHANISM-PASS`） |
| `run8_dry_run.py` | Run8 CAACP-HQKV drop-run | **已失效**：它 import 的 `caacp_hqkv_v2.py` 已于 2026-10-20 拆除并归档到 `train_scripts/CASA-TViM/Run8b/_SUPERSEDED_hqkv_v2_2026-10-20/` |
| `test_run4_fine_tap.py` | Run4 FET1 预检（T1..T10） | 同上 |
| `test_run7_hf_edge.py` | Run7 HFAE 预检（T1..T11） | 同上 |
| `test_run8_cpu_precheck.py` | Run8 CPU 预检 26 项（含退化定理复现） | 同上；退化定理的结论已并入 `test_hsam_cpu.py` 的固定断言 |

## 注意

- 这些脚本把 `models/` 当作工作目录、靠相对路径 import `model.*`。
  现在它们在 `models/_legacy/` 下，**直接跑会 import 失败**；要用请在 `models/` 下执行：
  `python _legacy/run7_dry_run.py ...`，或在脚本里补 `sys.path`。
- `models/smoke_test.py` 里 `T8` 会自动调用 `_legacy/test_run4_fine_tap.py`，路径已同步修正。
- 若某个脚本的依赖文件已被拆除（Run8 相关的就是），**不要为了让它能跑而改现行代码** ——
  需要复现 Run8 的话，按归档目录的 `MANIFEST.sha256` 把原始字节放回去，并显式换协议串。
