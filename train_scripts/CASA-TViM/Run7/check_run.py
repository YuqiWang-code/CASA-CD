#!/usr/bin/env python
"""Run7 单 run 收尾硬门（Run4 `check_run.py` 的 Run7 版本，方案 §收口）。

只从日志的**最后一个完整 TEST 区块**判定，不看 python 退出码（沿用 Run4 §8.4 的教训：
python rc=0 不代表 run 合规）。

    python train_scripts/CASA-TViM/Run7/check_run.py \
      --log <...>/E9_HFAE_A/SYSU-CD-256/train_log.txt \
      --variant E9_HFAE_A --dataset SYSU-CD-256 \
      --ckpt-dir <...>/E9_HFAE_A/SYSU-CD-256

它比 `collect_results.py` 的 12-run 门**多管两件消融可解释性的关键事**：

1. **CTRL 必须"干净"**：`M1_R7CTRL` 的日志里除 `[HFAE-CONFIG] hf_edge=0 form=none` 外，
   不得出现任何 `[HFAE]/[HFAE-FORM]/[HFAE-TAPS]/[HFAE-DEPLOY]/[HFAE-GRAD]/[HFAE-LOSS]`
   行 ⇒ 保证对照组没有偷偷带上被消融的机制（"唯一变量"）。
2. **实验组必须"真的接上了"**：`[HFAE-GRAD]` 行必须显示 `hf_tap_grad_norm > 0` 且
   `nonzero_tap_grad_tensors > 0` ⇒ 边缘辅助损失的梯度确实回流进了高频分支，
   而不是只挂了一个旁路头（这正是 Run5「只读 HF 通道」被否证的那种失败模式）。

硬门（任一不满足 ⇒ rc=1，该 run 记 INVALID）：
  * 最后一个完整 `=== TEST RESULTS ===` … `=== END TEST RESULTS ===` 区块存在；
  * 区块内 `[ACTUAL-OPT-STEPS] 80000`；
  * 区块内 `[PROTOCOL-VERSION] run7_hf_edge_exact80k_v1`；
  * 区块内 `[DEPLOY-PARAMS] total=` ≤ 5.000 M 且等于该变体的期望值；
  * 区块内 `[REPARAM-REAL-ARGMAX-DISAGREE] == 0`（折叠等价性，真实数据 batch）；
  * 日志无 `[MANIFEST-MISMATCH]` / `[PROTOCOL-MISMATCH]` / `budget already reached`；
  * 变体契约（见上 1/2，以及 λ=0.2 固定、taps=1_4_1_8、`[HFAE-DEPLOY]` 的 dropped/deploy_params）；
  * `arch.json` / `run_manifest.json` 与日志一致（hf_edge/hf_edge_form/hf_edge_taps/
    hf_edge_lambda/protocol_version），且 `source_code_sha256` 非空（Run7 协议要求）；
  * **冻结协议字段**（`run_manifest.json`）：batch_size=32、seed=16、max_steps=80000、
    exact_max_steps=1、data_contract=legacy_6ch_reverse_v1、backbone_lr_ratio=0.1、
    str_dim=96、feature_taps=冻结四tap、dataset 相符；
  * **崩溃恢复溯源**：`resume_source` 必须为 None，或指向**本 run 自己**的 `last.pth`
    （指向别的变体/库等于拿别人的权重续训，会静默污染受控消融）。
"""
import argparse
import json
import os
import re
import sys

EXPECT_STEPS = 80000
PROTOCOL = "run7_hf_edge_exact80k_v1"
BUDGET = 5_000_000
EXPECT_LAMBDA = 0.2
EXPECT_TAPS = "1_4_1_8"
# 冻结协议字段（从**真实** run_manifest.json 实测得到，2026-10-11 波一四库核对）
EXPECT_BATCH = 32
EXPECT_SEED = 16
EXPECT_DATA_CONTRACT = "legacy_6ch_reverse_v1"
EXPECT_FEATURE_TAPS = ["stage1_1_4_48", "stage2_1_8_64", "stage3_1_16_168", "stage4_1_32_224"]
EXPECT_BACKBONE_LR_RATIO = 0.1
EXPECT_STR_DIM = 96

# 变体契约表（**按实测的 sidecar/log 真实取值**，2026-10-11 用真实 train.py 产出核对）：
#   hf_edge, form, [DEPLOY-PARAMS] total(M), [HFAE-DEPLOY] deploy_params, dropped,
#   sidecar hf_edge_taps, sidecar hf_edge_lambda, sidecar hf_edge_tag
# 注意 CTRL 的 sidecar 记的是**生效值**（taps='none' / lambda=0.0 / tag=None），
# 而日志里的 `[HFAE-CONFIG]` 回显的是 CLI 参数（taps=1_4_1_8 lambda=0.2）——
# 两者口径不同是设计如此，检查必须分别对齐，否则会把正常 run 判死。
VARIANTS = {
    "M1_R7CTRL": (0, "none", 4.880, None, None, "none", 0.0, None),
    "E9_HFAE_A": (1, "aux_only", 4.880, 4_880_190, "entire_head", EXPECT_TAPS, EXPECT_LAMBDA,
                  "hfae_v1"),
    "E10_HFAE_B": (1, "aux_inject", 4.890, 4_890_366, "edge_out_only", EXPECT_TAPS,
                   EXPECT_LAMBDA, "hfae_v1"),
}
# 只在实验组出现的机制行（CTRL 出现任何一条即违反"唯一变量"）
MECH_TAGS = ("[HFAE]", "[HFAE-FORM]", "[HFAE-TAPS]", "[HFAE-DEPLOY]",
             "[HFAE-GRAD]", "[HFAE-LOSS]")

_BLOCK = re.compile(r"={3} TEST RESULTS ={3}\r?\n(.*?)\r?\n={3} END TEST RESULTS ={3}", re.S)


def last_block(log_path):
    if not os.path.isfile(log_path):
        return None, ""
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    got = _BLOCK.findall(txt)
    return (got[-1] if got else None), txt


def _get(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def check(log_path, variant, dataset, ckpt_dir):
    checks = []
    exp = VARIANTS.get(variant)
    if exp is None:
        return [("known variant", False, f"unknown variant {variant!r}")], None
    exp_hf, exp_form, exp_depl_M, exp_hfae_depl, exp_dropped = exp[:5]
    exp_taps, exp_lambda, exp_tag = exp[5], exp[6], exp[7]

    block, log_text = last_block(log_path)
    checks.append(("last complete TEST block present", block is not None,
                   f"{log_path} (len={len(log_text)})"))

    # ---- 区块内硬门。没有完整 TEST 区块时**显式记一条 FAIL**，不静默跳过
    # （Run6 的教训：空集"通过"会让门看起来没事）。
    if block is None:
        checks.append(("block gates [steps/protocol/deploy/fold] evaluated", False,
                       "skipped: no complete TEST block"))
    else:
        steps = _get(r"\[ACTUAL-OPT-STEPS\]\s*(\d+)", block, int)
        checks.append((f"[ACTUAL-OPT-STEPS] == {EXPECT_STEPS}", steps == EXPECT_STEPS,
                       f"got {steps}"))

        proto = _get(r"\[PROTOCOL-VERSION\]\s*(\S+)", block, str)
        checks.append((f"[PROTOCOL-VERSION] == {PROTOCOL}", proto == PROTOCOL, f"got {proto}"))

        depl = _get(r"\[DEPLOY-PARAMS\][^\n]*?total=([0-9.]+)\s*M", block)
        checks.append((f"[DEPLOY-PARAMS] total == {exp_depl_M} M",
                       depl is not None and abs(depl - exp_depl_M) < 5e-4, f"got {depl} M"))
        checks.append((f"[DEPLOY-PARAMS] total <= {BUDGET/1e6} M",
                       depl is not None and depl <= BUDGET / 1e6, f"got {depl} M"))

        dis_real = _get(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", block)
        checks.append(("fold binary disagreement == 0 (real-data batch; OPERATIONAL gate)",
                       dis_real == 0.0, f"got {dis_real}"))

    for tag in ("[MANIFEST-MISMATCH]", "[PROTOCOL-MISMATCH]"):
        checks.append((f"log has no {tag}", tag not in log_text, ""))
    checks.append(("[EXACT-STEPS] budget not pre-exhausted at launch",
                   "budget already reached" not in log_text, ""))

    # ---- 变体契约：CTRL 必须干净 / 实验组必须真的接上
    # 注意：这些 `[HFAE-*]` 行是**启动期/收尾期**写的，不在 TEST 区块里 ⇒ 与区块无关，
    # 无论有没有区块都必须检查。
    cfg = re.search(r"\[HFAE-CONFIG\][^\n]*hf_edge=(\d+)\s+form=(\S+)", log_text)
    checks.append(("[HFAE-CONFIG] logged", cfg is not None, "config line present"))
    if cfg:
        checks.append((f"[HFAE-CONFIG] hf_edge={exp_hf} form={exp_form}",
                       int(cfg.group(1)) == exp_hf and cfg.group(2) == exp_form,
                       f"got hf_edge={cfg.group(1)} form={cfg.group(2)}"))

    present = {t: (t in log_text) for t in MECH_TAGS}
    if exp_hf == 0:
        leaked = [t for t, ok in present.items() if ok]
        checks.append(("CTRL: no HFAE mechanism lines anywhere in the log "
                       "(only-variable guarantee)", not leaked,
                       f"leaked={leaked}" if leaked else "clean"))
    else:
        for t in ("[HFAE]", "[HFAE-FORM]", "[HFAE-TAPS]", "[HFAE-DEPLOY]", "[HFAE-GRAD]",
                  "[HFAE-LOSS]"):
            checks.append((f"{t} present", present[t], "mechanism evidence line"))

        form_line = re.search(r"\[HFAE-FORM\]\s*(\S+)\s+taps=(\S+)\s+lambda=(\S+)\s+tag=(\S+)",
                              log_text)
        checks.append((f"[HFAE-FORM] form={exp_form} taps={EXPECT_TAPS} tag=hfae_v1",
                       form_line is not None and form_line.group(1) == exp_form
                       and form_line.group(2) == EXPECT_TAPS and form_line.group(4) == "hfae_v1",
                       form_line.groups() if form_line else "no [HFAE-FORM] line"))
        if form_line:
            lam = float(form_line.group(3))
            checks.append((f"[HFAE-FORM] lambda == {EXPECT_LAMBDA} (预注册固定，禁止搜索)",
                           abs(lam - EXPECT_LAMBDA) < 1e-9, f"got {lam}"))

        hfae_depl = _get(r"\[HFAE-DEPLOY\][^\n]*deploy_params=(\d+)", log_text, int)
        checks.append((f"[HFAE-DEPLOY] deploy_params == {exp_hfae_depl:,}",
                       hfae_depl == exp_hfae_depl, f"got {hfae_depl}"))
        dropped = _get(r"\[HFAE-DEPLOY\][^\n]*dropped=(\S+)", log_text, str)
        checks.append((f"[HFAE-DEPLOY] dropped == {exp_dropped}", dropped == exp_dropped,
                       f"got {dropped}"))

        # 机制真的接上了：辅助损失梯度必须回流进高频分支
        g = re.search(r"\[HFAE-GRAD\][^\n]*hf_tap_grad_norm=([0-9.eE+-]+)"
                      r"[^\n]*nonzero_tap_grad_tensors=(\d+)"
                      r"[^\n]*head_grad_norm=([0-9.eE+-]+)", log_text)
        checks.append(("[HFAE-GRAD] line present", g is not None, "step-1 gradient probe"))
        if g:
            tap_gn, nz, head_gn = float(g.group(1)), int(g.group(2)), float(g.group(3))
            checks.append(("HF tap gradient is non-zero (aux loss reaches the HF branch)",
                           tap_gn > 0 and nz > 0, f"tap_grad_norm={tap_gn:.6e} nonzero={nz}"))
            checks.append(("aux head gradient is non-zero", head_gn > 0, f"head_grad_norm={head_gn:.6e}"))

        if exp_form == "aux_inject":
            zi = re.search(r"\[HFAE\][^\n]*zero_init=(\w+)", log_text)
            checks.append(("[HFAE] zero_init=True (注入分支零初始化)",
                           zi is not None and zi.group(1) == "True",
                           f"got {zi.group(1) if zi else None}"))

        aux = [float(x) for x in
               re.findall(r"\[HFAE-LOSS\][^\n]*aux_loss=([0-9.eE+-]+)", log_text)]
        checks.append(("[HFAE-LOSS] aux_loss > 0 recorded (epoch-level)",
                       len(aux) > 0 and max(aux) > 0,
                       f"n={len(aux)} max={max(aux) if aux else None}"))

    # ---- sidecar 一致性
    for name in ("arch.json", "run_manifest.json"):
        p = os.path.join(ckpt_dir, name)
        if not os.path.isfile(p):
            checks.append((f"{name} present", False, p))
            continue
        try:
            obj = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            checks.append((f"{name} parses", False, repr(e)))
            continue
        checks.append((f"{name}: hf_edge={exp_hf} form={exp_form} taps={exp_taps}",
                       int(obj.get("hf_edge", -1)) == exp_hf
                       and (obj.get("hf_edge_form") or "none") == exp_form
                       and (obj.get("hf_edge_taps") or "none") == exp_taps,
                       f"got hf_edge={obj.get('hf_edge')} form={obj.get('hf_edge_form')} "
                       f"taps={obj.get('hf_edge_taps')}"))
        checks.append((f"{name}: lambda == {exp_lambda}",
                       abs(float(obj.get("hf_edge_lambda", -1)) - exp_lambda) < 1e-9,
                       f"got {obj.get('hf_edge_lambda')}"))
        checks.append((f"{name}: hf_edge_tag == {exp_tag}",
                       (obj.get("hf_edge_tag") or None) == exp_tag,
                       f"got {obj.get('hf_edge_tag')}"))
        checks.append((f"{name}: protocol_version == {PROTOCOL}",
                       obj.get("protocol_version") == PROTOCOL,
                       f"got {obj.get('protocol_version')}"))
        src = obj.get("source_code_sha256") or {}
        # train.py 把每个文件的 sha256 截断成 16 位十六进制（实测 '5bc6515389ad3c09'）；
        # 这里同时接受完整 64 位，避免以后改成全长时误判。
        good = all(re.fullmatch(r"[0-9a-fA-F]{16}|[0-9a-fA-F]{64}", str(v)) for v in src.values())
        checks.append((f"{name}: source_code_sha256 present & well-formed",
                       bool(src) and good, f"{len(src)} files"))

    # ---- 冻结协议字段：把"脚本里参数写对了"升级为"**每个 run 的落盘清单**也对"
    # 这些断言直接对着目标第(4)条（各严格 80,000 step / batch 32 / 从同一预训练权重从头训）。
    # 只在 run_manifest.json 里有这些字段（arch.json 没有 batch/seed）。
    mp = os.path.join(ckpt_dir, "run_manifest.json")
    man = None
    if os.path.isfile(mp):
        try:
            man = json.load(open(mp, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            checks.append(("run_manifest.json parses for protocol check", False, repr(e)))
    else:
        checks.append(("run_manifest.json present for protocol check", False, mp))
    if man is not None:
        for k, want in (("batch_size", EXPECT_BATCH), ("seed", EXPECT_SEED),
                        ("max_steps", EXPECT_STEPS), ("exact_max_steps", 1),
                        ("data_contract", EXPECT_DATA_CONTRACT),
                        ("backbone_lr_ratio", EXPECT_BACKBONE_LR_RATIO),
                        ("str_dim", EXPECT_STR_DIM), ("dataset", dataset)):
            got = man.get(k)
            checks.append((f"manifest {k} == {want!r}", got == want, f"got {got!r}"))
        got_taps = man.get("feature_taps")
        checks.append(("manifest feature_taps == frozen 4 taps",
                       got_taps == EXPECT_FEATURE_TAPS, f"got {got_taps}"))
        # 崩溃恢复：resume_source 必须为 None，或指向**本 run 自己**的 last.pth。
        # 若指向别的变体/库，等于拿别人的权重续训 ⇒ 会静默污染受控消融。
        rs = man.get("resume_source")
        own_suffix = f"{variant}/{dataset}/last.pth"
        rs_norm = str(rs).replace("\\", "/") if rs is not None else None
        checks.append(("manifest resume_source is None or this run's own last.pth",
                       rs is None or rs_norm.endswith(own_suffix),
                       f"got {rs!r} (expect None or .../{own_suffix})"))
    return checks, {"n_fail": sum(1 for c in checks if not c[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    checks, _ = check(a.log, a.variant, a.dataset, a.ckpt_dir)
    bad = [c[0] for c in checks if not c[1]]
    if not a.quiet:
        print(f"\n=== RUN7 CHECK_RUN {a.variant} / {a.dataset} ===")
        for name, ok, extra in checks:
            print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {extra}" if extra else ""))
        print(f"--- {'ALL PASS' if not bad else 'FAILED: ' + '; '.join(bad)} ---")
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"variant": a.variant, "dataset": a.dataset, "log": a.log,
                       "n_checks": len(checks), "n_failed": len(bad),
                       "failed": bad,
                       "checks": [{"name": n, "ok": o, "detail": str(e)}
                                  for n, o, e in checks]}, f, indent=2, ensure_ascii=False)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
