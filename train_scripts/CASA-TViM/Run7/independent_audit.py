#!/usr/bin/env python
"""Run7 独立最小复算审计（与 collect_results / run7_object_metrics / run7_verdict **不同代码路径**）。

项目纪律要求：收口必须有**独立最小复算审计**，且审计脚本与主脚本必须是不同实现，
写出 `INDEPENDENT_AUDIT.json`（学 Run5 F15 教训：主脚本曾把对象口径写错，靠独立复算才发现）。

本脚本**不 import** 上面三个工具，自带：
  * 自己的 TEST 区块抽取与六指标正则（不使用 `tvim_diag_common.parse_test_block`）；
  * 自己的 F1↔IoU 自洽性推导 `F1 == 2·IoU/(1+IoU)`（layer-A 自洽性，与 §7.3 同口径）；
  * 自己的身份/协议/deploy 期望表；
  * 自己的"接受规则"重推导（≥3/4 库 F1 ≥ CTRL 且 SYSU 提升），与 `run7_verdict.json` 对拍。

    python train_scripts/CASA-TViM/Run7/independent_audit.py \
      --log-root /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7 \
      --out /home/yqwang/outputs/CASA-CD/CASA-TViM/Run7/_RESULTS/INDEPENDENT_AUDIT.json
    python train_scripts/CASA-TViM/Run7/independent_audit.py --selftest

对照用（阳性对照：把已完成的历史 Run 当输入，验证解析/自洽性检查在真实日志上成立）：

    python .../independent_audit.py --log-root <历史Run日志> --ckpt-root <历史Run权重> \\
      --variants M1_R4CTRL --datasets CDD-CD-256,LEVIR-CD-256,SYSU-CD-256,WHU-CD-256 \\
      --protocol run4_exact80k_v1
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOCK = os.path.join(HERE, "SELECTION_LOCK.json")
PROTOCOL = "run7_hf_edge_exact80k_v1"
VARIANTS = ["M1_R7CTRL", "E9_HFAE_A", "E10_HFAE_B"]
DATASETS = ["CDD-CD-256", "LEVIR-CD-256", "SYSU-CD-256", "WHU-CD-256"]
# 冻结的 deploy 参数期望（§3 表；由 T8 与 eval.py 实测过）
DEPLOY_EXPECT = {"M1_R7CTRL": 4_880_190, "E9_HFAE_A": 4_880_190, "E10_HFAE_B": 4_890_366}
FORM_EXPECT = {"M1_R7CTRL": "none", "E9_HFAE_A": "aux_only", "E10_HFAE_B": "aux_inject"}
HF_EXPECT = {"M1_R7CTRL": 0, "E9_HFAE_A": 1, "E10_HFAE_B": 1}

# ---- 自己的区块抽取（不用共享解析器）
_BLOCK = re.compile(r"={3} TEST RESULTS ={3}\r?\n(.*?)\r?\n={3} END TEST RESULTS ={3}", re.S)
_SIX = re.compile(r"Recall=([0-9.]+)\s*\|\s*Precision=([0-9.]+)\s*\|\s*OA=([0-9.]+)\s*\|"
                  r"\s*F1=([0-9.]+)\s*\|\s*IoU=([0-9.]+)\s*\|\s*Kappa=([0-9.]+)")


def last_block(log_path):
    if not os.path.isfile(log_path):
        return None
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    got = _BLOCK.findall(txt)
    return got[-1] if got else None


def parse_run(log_path):
    """返回 dict（parse_ok=False 时其余字段可能为 None）。"""
    rec = {"log": log_path, "parse_ok": False}
    b = last_block(log_path)
    if b is None:
        rec["status"] = "NO_COMPLETE_TEST_BLOCK"
        return rec
    m = _SIX.search(b)
    if not m:
        rec["status"] = "NO_SIX_METRICS"
        return rec
    recall, prec, oa, f1, iou, kappa = (float(x) for x in m.groups())
    steps = re.search(r"\[ACTUAL-OPT-STEPS\]\s+(\d+)", b)
    proto = re.search(r"\[PROTOCOL-VERSION\]\s+(\S+)", b)
    depl = re.search(r"\[DEPLOY-PARAMS\][^\n]*?total=([0-9.]+)\s*M", b)
    hfae_depl = re.search(r"\[HFAE-DEPLOY\][^\n]*?deploy_params=(\d+)", b)
    fold = re.search(r"\[REPARAM-REAL-ARGMAX-DISAGREE\]\s*([0-9.eE+-]+)", b)
    rec.update({
        "parse_ok": True,
        "status": "PARSED",
        "metrics": {"Recall": recall, "Precision": prec, "OA": oa, "F1": f1,
                    "IoU": iou, "Kappa": kappa},
        "opt_steps": int(steps.group(1)) if steps else None,
        "protocol_version": proto.group(1) if proto else None,
        "deploy_params_M": float(depl.group(1)) if depl else None,
        "hfae_deploy_params": int(hfae_depl.group(1)) if hfae_depl else None,
        "fold_real_disagree": float(fold.group(1)) if fold else None,
    })
    return rec


def read_arch(ckpt_root, variant, dataset):
    p = os.path.join(ckpt_root, variant, dataset, "arch.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def f1_from_iou(iou):
    return 2.0 * iou / (1.0 + iou) if (1.0 + iou) else None


def audit(log_root, ckpt_root, verdict_path=None, variants=None, datasets=None,
          protocol=None, knife_dir=None):
    variants = variants or VARIANTS
    datasets = datasets or DATASETS
    protocol = protocol or PROTOCOL
    rows = {}
    for v in variants:
        for d in datasets:
            rows[(v, d)] = parse_run(os.path.join(log_root, v, d, "train_log.txt"))
    checks, detail = {}, {}

    ready = [f"{v}/{d}" for (v, d) in rows if not rows[(v, d)].get("parse_ok")]
    checks["A1_all_12_blocks_parsed"] = (len(ready) == 0)
    detail["not_ready"] = ready
    detail["n_rows_parsed"] = len(rows) - len(ready)
    detail["n_rows_expected"] = len(rows)

    # A2 六指标自洽性（F1 == 2IoU/(1+IoU)）与取值范围 —— 独立推导，不看主脚本
    worst = 0.0
    worst_key = None
    rng_bad = []
    for k, r in rows.items():
        if not r.get("parse_ok"):
            continue
        m = r["metrics"]
        exp = f1_from_iou(m["IoU"])
        d_ = abs(exp - m["F1"])
        if d_ > worst:
            worst, worst_key = d_, f"{k[0]}/{k[1]}"
        for name, x in m.items():
            if not (0.0 <= x <= 1.0):
                rng_bad.append(f"{k[0]}/{k[1]}.{name}={x}")
    checks["A2_f1_iou_self_consistent"] = worst <= 1e-3
    detail["f1_iou_max_delta"] = worst
    detail["f1_iou_worst_key"] = worst_key
    checks["A3_metrics_in_unit_range"] = (len(rng_bad) == 0)
    detail["range_violations"] = rng_bad

    # A4 协议与步数门
    bad_gate = []
    for k, r in rows.items():
        if not r.get("parse_ok"):
            continue
        if r.get("opt_steps") != 80000 or r.get("protocol_version") != protocol:
            bad_gate.append(f"{k[0]}/{k[1]}: steps={r.get('opt_steps')} "
                            f"proto={r.get('protocol_version')}")
    checks["A4_steps80000_and_protocol"] = (len(bad_gate) == 0)
    detail["gate_violations"] = bad_gate

    # A5 deploy 参数期望值（按变体；CTRL 用 [DEPLOY-PARAMS]，实验组用 [HFAE-DEPLOY]）
    ctrl = next((v for v in variants if v.startswith("M1_")), variants[0])
    exp_vars = [v for v in variants if v != ctrl]
    dep_bad, dep_skipped = [], []
    for k, r in rows.items():
        if not r.get("parse_ok"):
            continue
        v = k[0]
        if v == ctrl:
            # CTRL 无 [HFAE-DEPLOY] 行，改用 [DEPLOY-PARAMS] 的 M 值
            if r.get("deploy_params_M") != 4.88:
                dep_bad.append(f"{k[0]}/{k[1]}: deploy_M={r.get('deploy_params_M')}")
        elif v in DEPLOY_EXPECT:
            got = r.get("hfae_deploy_params")
            if got != DEPLOY_EXPECT[v]:
                dep_bad.append(f"{k[0]}/{k[1]}: hfae_deploy={got} expect {DEPLOY_EXPECT[v]}")
        else:
            dep_skipped.append(f"{k[0]}/{k[1]}")
    checks["A5_deploy_params_match_frozen"] = (len(dep_bad) == 0)
    detail["deploy_violations"] = dep_bad
    detail["deploy_unchecked_variants"] = sorted(set(dep_skipped))

    # A6 真实折叠 disagreement 必须为 0，**或**有吻合的刀锋定位证据（pre-reg §7.6）。
    # 本处为**独立第三实现**（不 import check_run / run7_verdict），故意用不同的写法交叉验证。
    def _a6_knife_ok(v, d, dis, mx):
        if not knife_dir:
            return False, "nonzero disagreement and no --knife-dir"
        p = os.path.join(knife_dir, f"KNIFE_EDGE_{d}_{v}.json")
        if not os.path.isfile(p):
            return False, "no evidence file"
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return False, f"unreadable: {e}"
        checks_ = {
            "identity": j.get("variant") == v and j.get("dataset") == d,
            "verdict": j.get("verdict") == "KNIFE_EDGE_ARTIFACT",
            "flipped>0": (j.get("n_flipped") or 0) > 0,
            "disagree~log": (dis is not None and j.get("disagree") is not None
                             and abs(float(j["disagree"]) - float(dis))
                             / max(abs(float(dis)), 1e-30) < 1e-3),
            "maxabs~log": (mx is None or j.get("max_abs_diff") is None
                           or abs(float(j["max_abs_diff"]) - float(mx))
                           / max(abs(float(mx)), 1e-30) < 1e-3),
        }
        bad = [k for k, okv in checks_.items() if not okv]
        return (not bad), ("evidence ok" if not bad else f"evidence fails: {bad}")

    fold_bad, knife_used = [], []
    for k, r in rows.items():
        f = r.get("fold_real_disagree")
        if f is None or f == 0.0:
            continue
        ok6, why6 = _a6_knife_ok(k[0], k[1], f, r.get("fold_real_max_abs_error"))
        if ok6:
            knife_used.append(f"{k[0]}/{k[1]}: disagree={f} ({why6})")
        else:
            fold_bad.append(f"{k[0]}/{k[1]}: fold_disagree={f} ({why6})")
    checks["A6_real_fold_disagreement_zero_or_knife_edge"] = (len(fold_bad) == 0)
    detail["fold_violations"] = fold_bad
    detail["fold_knife_edge_accepted"] = knife_used

    # A7 sidecar 身份：变体契约 + 全部 run 必须同源代码身份
    arch_bad, shas = [], {}
    for k in rows:
        v, d = k
        a = read_arch(ckpt_root, v, d)
        if a is None:
            arch_bad.append(f"{v}/{d}: no arch.json")
            continue
        is_ctrl = (v == ctrl)
        exp_hf = 0 if is_ctrl else HF_EXPECT.get(v)
        exp_form = "none" if is_ctrl else FORM_EXPECT.get(v)
        if int(a.get("hf_edge", -1)) != exp_hf:
            arch_bad.append(f"{v}/{d}: hf_edge={a.get('hf_edge')} expect {exp_hf}")
        if (a.get("hf_edge_form", "none") or "none") != exp_form:
            arch_bad.append(f"{v}/{d}: form={a.get('hf_edge_form')} expect {exp_form}")
        if a.get("protocol_version") != protocol:
            arch_bad.append(f"{v}/{d}: protocol={a.get('protocol_version')}")
        src = a.get("source_code_sha256") or {}
        shas[f"{v}/{d}"] = hashlib.sha256(
            json.dumps(src, sort_keys=True).encode()).hexdigest()[:16]
    checks["A7_sidecar_variant_contract"] = (len(arch_bad) == 0)
    detail["arch_violations"] = arch_bad
    uniq = sorted(set(shas.values()))
    n_expect = len(variants) * len(datasets)
    checks["A8_single_code_identity_across_12_runs"] = (len(uniq) == 1 and len(shas) == n_expect)
    detail["code_identity_hashes"] = shas
    detail["distinct_code_identities"] = uniq
    detail["n_code_identity_rows"] = len(shas)

    # A9 独立重推导接受规则，与 verdict.json 对拍
    f1 = {(v, d): (rows[(v, d)].get("metrics") or {}).get("F1")
          for v in variants for d in datasets}
    if f1 and all(x is not None for x in f1.values()):
        picked = []
        for v in exp_vars:
            n_ge = sum(1 for d in datasets if f1[(v, d)] >= f1[(ctrl, d)])
            ref = "SYSU-CD-256" if "SYSU-CD-256" in datasets else datasets[0]
            sysu = f1[(v, ref)] > f1[(ctrl, ref)]
            if n_ge >= 3 and sysu:
                picked.append(v)
        detail["independent_accept"] = {"ctrl": ctrl, "picked": picked,
                                        "delta": {f"{v}/{d}": (f1[(v, d)] - f1[(ctrl, d)])
                                                  for v in exp_vars
                                                  for d in datasets}}
        vj = {}
        if verdict_path and os.path.isfile(verdict_path):
            vj = json.load(open(verdict_path, encoding="utf-8"))
        v_winner = vj.get("winner")
        checks["A9_accept_rule_reproduced"] = (
            bool(vj) and ((v_winner in picked) or (not picked and v_winner is None)))
        detail["verdict_winner"] = v_winner
    else:
        checks["A9_accept_rule_reproduced"] = False
        detail["independent_accept"] = "not all F1 available"

    # A10 lock 未被篡改（阈值来源可核验）
    lock = json.load(open(LOCK, encoding="utf-8"))
    th = lock.get("acceptance_criteria_frozen_numeric") or {}
    checks["A10_lock_thresholds_present"] = bool(
        th.get("_frozen_before_any_experiment_result") is True
        and th.get("min_datasets_ge_ctrl") == 3
        and abs(float(th.get("guardrail_recall_drop_max", -1)) - 0.02) < 1e-12)
    detail["lock_thresholds"] = th

    n_fail = sum(1 for x in checks.values() if not x)
    return {"n_checks": len(checks), "n_failed": n_fail, "checks": checks,
            "detail": detail, "protocol": protocol,
            "note": ("独立最小复算审计：自带区块抽取/正则/身份表/接受规则重推导，"
                     "不 import collect_results / run7_object_metrics / run7_verdict。")}


# --------------------------------------------------------------------------- selftest
def selftest():
    """验证审计自身的数学与规则重推导（合成夹具，不依赖任何真实 run）。"""
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    chk("f1_from_iou(0.5)=0.6667", abs(f1_from_iou(0.5) - 2 / 3) < 1e-12)
    chk("f1_from_iou(0)=0", f1_from_iou(0.0) == 0.0)
    chk("f1_from_iou(1)=1", abs(f1_from_iou(1.0) - 1.0) < 1e-12)

    # 区块抽取必须只取**最后一个**
    txt = ("=== TEST RESULTS ===\nF1=0.1\n=== END TEST RESULTS ===\n"
           "=== TEST RESULTS ===\nF1=0.9\n=== END TEST RESULTS ===\n")
    m = _BLOCK.findall(txt)
    chk("block regex takes last", len(m) == 2 and "0.9" in m[-1])

    # 六指标正则
    six = _SIX.search("Recall=0.1 | Precision=0.2 | OA=0.3 | F1=0.4 | IoU=0.5 | Kappa=0.6")
    chk("six-metric regex", six is not None and six.groups()[3] == "0.4")

    # 接受规则重推导（≥3/4 且 SYSU 提升）
    def accept(delta):
        n_ge = sum(1 for d in DATASETS if delta[d] >= 0)
        return n_ge >= 3 and delta["SYSU-CD-256"] > 0
    chk("accept: 4/4 with SYSU+", accept({d: 0.001 for d in DATASETS}))
    chk("accept: 3/4 with SYSU+", accept({d: (0.001 if d != "WHU-CD-256" else -0.001)
                                          for d in DATASETS}))
    chk("reject: 2/4", not accept({d: (0.001 if d in ("CDD-CD-256", "LEVIR-CD-256")
                                       else -0.001) for d in DATASETS}))
    chk("reject: SYSU flat", not accept({d: (0.0 if d == "SYSU-CD-256" else 0.001)
                                         for d in DATASETS}))
    ok = _selftest_end_to_end(chk) and ok
    print(f"SELFTEST = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _fixture(root, variants, datasets, proto="fixture_proto_v1", bad_f1=None,
             bad_proto=None):
    """造一棵最小可解析的假 Run 树，供 audit() 端到端自测（不依赖真实 run）。"""
    src = {"models/train.py": "a" * 64}
    for v in variants:
        for d in datasets:
            iou, f1 = (0.5, 2 * 0.5 / 1.5) if v.startswith("M1_") else (0.55, 2 * 0.55 / 1.55)
            if bad_f1 == (v, d):
                f1 = 0.9  # 故意破坏 F1 == 2IoU/(1+IoU)
            pr = proto if bad_proto != (v, d) else "wrong_proto"
            ld = os.path.join(root, "log", v, d)
            os.makedirs(ld, exist_ok=True)
            body = [f"[ACTUAL-OPT-STEPS] 80000", f"[PROTOCOL-VERSION] {pr}",
                    f"[DEPLOY-PARAMS] total=4.880 M",
                    f"Recall=0.9 | Precision=0.9 | OA=0.9 | F1={f1:.6f} | "
                    f"IoU={iou} | Kappa=0.8",
                    "[REPARAM-REAL-ARGMAX-DISAGREE] 0"]
            if not v.startswith("M1_"):
                body.append(f"[HFAE-DEPLOY] form=aux_only deploy_params="
                            f"{DEPLOY_EXPECT[v]}")
            with open(os.path.join(ld, "train_log.txt"), "w", encoding="utf-8",
                      newline="\n") as f:
                f.write("=== TEST RESULTS ===\n" + "\n".join(body)
                        + "\n=== END TEST RESULTS ===\n")
            cd = os.path.join(root, "ck", v, d)
            os.makedirs(cd, exist_ok=True)
            with open(os.path.join(cd, "arch.json"), "w", encoding="utf-8",
                      newline="\n") as f:
                json.dump({"hf_edge": 0 if v.startswith("M1_") else 1,
                           "hf_edge_form": "none" if v.startswith("M1_") else "aux_only",
                           "protocol_version": proto,
                           "source_code_sha256": src}, f)
    return os.path.join(root, "log"), os.path.join(root, "ck")


def _selftest_end_to_end(chk):
    """audit() 端到端：全部合规必须 0 失败；两处扰动必须各自被对应检查抓到。"""
    import shutil
    import tempfile

    variants = ["M1_R7CTRL", "E9_HFAE_A"]
    ds = list(DATASETS)
    tmp = tempfile.mkdtemp(prefix="r7_audit_fix_")
    try:
        log_root, ck_root = _fixture(tmp, variants, ds)
        vj = os.path.join(tmp, "verdict.json")
        with open(vj, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"winner": "E9_HFAE_A"}, f)
        r = audit(log_root, ck_root, vj, variants=variants, datasets=ds,
                  protocol="fixture_proto_v1")
        chk("e2e clean fixture -> 0 failed", r["n_failed"] == 0)
        if r["n_failed"]:
            for k in sorted(r["checks"]):
                if not r["checks"][k]:
                    print(f"        (unexpected) {k}: {r['detail'].get(k)}")
        chk("e2e A9 reproduced winner E9_HFAE_A",
            r["checks"]["A9_accept_rule_reproduced"] is True
            and r["detail"]["independent_accept"]["picked"] == ["E9_HFAE_A"])
        chk("e2e A8 rows == len(variants)*len(datasets)",
            r["detail"]["n_code_identity_rows"] == len(variants) * len(ds))

        shutil.rmtree(tmp)
        tmp = tempfile.mkdtemp(prefix="r7_audit_fix2_")
        log_root, ck_root = _fixture(tmp, variants, ds, bad_f1=("E9_HFAE_A", ds[2]))
        r2 = audit(log_root, ck_root, None, variants=variants, datasets=ds,
                   protocol="fixture_proto_v1")
        chk("e2e broken F1 caught by A2", r2["checks"]["A2_f1_iou_self_consistent"] is False
            and r2["detail"]["f1_iou_worst_key"] == f"E9_HFAE_A/{ds[2]}")

        shutil.rmtree(tmp)
        tmp = tempfile.mkdtemp(prefix="r7_audit_fix3_")
        log_root, ck_root = _fixture(tmp, variants, ds, bad_proto=("M1_R7CTRL", ds[0]))
        r3 = audit(log_root, ck_root, None, variants=variants, datasets=ds,
                   protocol="fixture_proto_v1")
        chk("e2e wrong protocol caught by A4",
            r3["checks"]["A4_steps80000_and_protocol"] is False
            and any("wrong_proto" in x for x in r3["detail"]["gate_violations"]))
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="/home/yqwang/outputs/CASA-CD/CASA-TViM/Run7")
    ap.add_argument("--ckpt-root",
                    default="/share_datasets/yqwang/checkpoints/CASA-CD/CASA-TViM/Run7")
    ap.add_argument("--verdict", default=None)
    ap.add_argument("--out", default=None)
    # 允许指向别的 Run（例如把 Run4 的 M1_R4CTRL 当**阳性对照**，验证解析与自洽性检查
    # 在真实已完成日志上成立）。不传时保持 Run7 冻结默认值。
    ap.add_argument("--variants", default=None,
                    help="逗号分隔；默认 M1_R7CTRL,E9_HFAE_A,E10_HFAE_B")
    ap.add_argument("--datasets", default=None, help="逗号分隔；默认四数据集")
    ap.add_argument("--protocol", default=None, help="默认 run7_hf_edge_exact80k_v1")
    ap.add_argument("--knife-dir", default=None,
                    help="刀锋定位证据目录；真实 batch disagreement>0 时必须有吻合证据")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    variants = [s.strip() for s in a.variants.split(",")] if a.variants else None
    datasets = [s.strip() for s in a.datasets.split(",")] if a.datasets else None
    if variants and not all(variants):
        print("[AUDIT][FAIL] --variants 含空项")
        return 2
    res = audit(a.log_root, a.ckpt_root, a.verdict or os.path.join(
        a.log_root, "_RESULTS", "verdict.json"),
        variants=variants, datasets=datasets, protocol=a.protocol,
        knife_dir=a.knife_dir)
    print(f"=== Run7 independent audit ({res['protocol']}): "
          f"{res['n_checks']} checks, {res['n_failed']} FAILED ===")
    for k in sorted(res["checks"]):
        print(f"  [{'OK' if res['checks'][k] else 'FAIL'}] {k}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=str)
        print(f"[AUDIT] wrote {a.out}")
        # 只有**规范跑**（未传 --variants/--datasets/--protocol 覆盖）才写回 SHA256SUMS：
        # 探索性对照跑（例如把审计指向历史 Run4）绝不能改动那个 Run 的证据清单。
        if not (a.variants or a.datasets or a.protocol):
            sums = os.path.join(a.log_root, "SHA256SUMS.txt")
            with open(sums, "a", encoding="utf-8") as f:
                f.write(f"{hashlib.sha256(open(a.out,'rb').read()).hexdigest()}  "
                        f"{os.path.basename(a.out)}\n")
            print(f"[AUDIT] appended to {sums}")
        else:
            print("[AUDIT] 对照跑：按纪律不改动目标 Run 的 SHA256SUMS.txt")
    return 0 if res["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
