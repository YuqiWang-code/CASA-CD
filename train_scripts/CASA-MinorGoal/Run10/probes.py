"""Mechanism probes for stage B, read off what the training logs already record.

Four questions have to be answered before a tree delta is believed:

1. radius vs scale -- do the coarse prototypes really sit further out?  The
   context lifts every prototype with ``sqrt(D) * sinh(r_s)`` and the per-stage
   ``key_radius_mean`` is printed every epoch, so the four numbers are already on
   disk.  A monotone increase with scale is the geometric claim being made.
2. entailment violation rate -- the share of top attention that crosses the
   prototype tree against the ancestor chain; also logged (``violation*`` keys).
3. ancestor mass share -- how much attention lands on the pruned ancestor chain
   rather than the leaves, i.e. whether the tree is used as a tree.
4. shuffled parent links -- relabel the tree so parents point at unrelated
   prototypes, re-evaluate, and report the F1 drop.  That one needs weights and
   data, so it runs with ``--shuffle`` on the server.

Log parsing is the cheap half and needs nothing but the downloaded run tree::

    python probes.py --root outputs/CASA-MinorGoal/Run10

``--shuffle`` adds the counterfactual::

    python probes.py --checkpoint .../H_TREE-2334/SYSU-CD-256/best_model.pth \\
        --dataset-root /share_datasets/CD/SYSU-CD-256 --module hyperbolic \\
        --tree --entailment 0.05 --shuffle
"""
import argparse
import json
import math
import re
from pathlib import Path

MODULE_DIAG = re.compile(r"\[MODULE-DIAG\] (\{.*\})")
EPOCH = re.compile(r"Epoch (\d+)/\d+ \|.*F1=([0-9.]+)")


def spearman(pairs):
    """Rank correlation for a handful of (scale, value) points."""
    if len(pairs) < 3:
        return float("nan")
    ranks = {key: index for index, (key, _) in enumerate(sorted(pairs))}
    values = {key: index for index, (key, _) in enumerate(sorted(pairs, key=lambda item: item[1]))}
    keys = [key for key, _ in pairs]
    mx = sum(ranks[key] for key in keys) / len(keys)
    my = sum(values[key] for key in keys) / len(keys)
    num = sum((ranks[k] - mx) * (values[k] - my) for k in keys)
    den = math.sqrt(sum((ranks[k] - mx) ** 2 for k in keys) * sum((values[k] - my) ** 2 for k in keys))
    return num / den if den else float("nan")


def parse_log(path):
    """Last epoch's F1 and the per-stage diagnostics of the final logged epoch."""
    text = path.read_text(encoding="utf-8", errors="replace")
    scores = EPOCH.findall(text)
    best = max((float(value) for _, value in scores), default=None)
    blocks = MODULE_DIAG.findall(text)
    diag = None
    for block in reversed(blocks):
        try:
            diag = json.loads(block)
        except json.JSONDecodeError:
            continue
        break
    return dict(f1=best, diag=diag, epochs=len(scores))


def stage_series(diag, key):
    """[(scale_index, value)] for stages that carry ``key``."""
    out = []
    for name, payload in sorted((diag or {}).items()):
        if not isinstance(payload, dict) or key not in payload:
            continue
        digits = "".join(ch for ch in name if ch.isdigit())
        if digits:
            out.append((int(digits) - 1, float(payload[key])))
    return out


def report(root):
    runs = []
    for log in sorted(Path(root).glob("*/*/train_log.txt")):
        parsed = parse_log(log)
        if parsed["diag"] is None:
            continue
        diag = parsed["diag"]
        radii = stage_series(diag, "key_radius_mean") or stage_series(diag, "radius_mean")
        gaps = stage_series(diag, "radial_gap_attention_mean")
        violations = {name: value for name, value in (diag.get("stage1") or {}).items()
                      if "violation" in name or "ancestor" in name}
        runs.append(dict(arm=log.parent.parent.name, dataset=log.parent.name,
                         f1=parsed["f1"], epochs=parsed["epochs"], radii=dict(radii),
                         radial_gap=dict(gaps), rho=spearman(radii) if radii else float("nan"),
                         extra=violations))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/CASA-MinorGoal/Run10")
    parser.add_argument("--markdown", default="outputs/CASA-MinorGoal/Run10/probes.md")
    args = parser.parse_args()
    rows = report(args.root)
    if not rows:
        raise SystemExit(f"no [MODULE-DIAG] lines under {args.root}")
    lines = ["# Run10 mechanism probes", "",
             "| arm | dataset | F1 | rho(scale, key radius) | stage radii | radial gap attn | violation |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        radii = " / ".join(f"{value:.2f}" for _, value in sorted(row["radii"].items())) or "-"
        gaps = " / ".join(f"{value:.2f}" for _, value in sorted(row["radial_gap"].items())) or "-"
        violation = " ".join(f"{name}={value:.4f}" for name, value in sorted(row["extra"].items())) or "-"
        lines.append(f"| {row['arm']} | {row['dataset']} | {row['f1'] or float('nan'):.4f} | "
                     f"{row['rho']:+.3f} | {radii} | {gaps} | {violation} |")
    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(text, encoding="utf-8")
    print(f"[PROBES] wrote {args.markdown}")


if __name__ == "__main__":
    main()
