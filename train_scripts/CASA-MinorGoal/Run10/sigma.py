"""Stage A turned into the number stage B gates on.

Stage A runs every recipe twice (seeds 2334, 2335) on three datasets and declares
nothing.  What it produces is the noise floor: the paired same-config difference
between the two seeds, which is the only honest scale for a one-seed delta.  With
nine pairs per dataset we report the RMS of those differences as ``sigma`` and
lock ``gate = max(0.36, 1.5 * sigma)`` -- 0.36 pp is the floor the earlier rounds
pre-registered, and 1.5 sigma is the margin a single delta needs before it counts.

Run it on the downloaded run tree after stage A finishes::

    python sigma.py --root outputs/CASA-MinorGoal/Run10 --json outputs/CASA-MinorGoal/Run10/gates.json
"""
import argparse
import json
import math
import re
from pathlib import Path

TEST = re.compile(r"=== TEST RESULTS ===(.*?)=== END TEST RESULTS ===", re.S)
F1 = re.compile(r"F1\s*[:=]\s*([0-9.]+)")
RECIPES = ("A_CTRL", "A_PHOTO", "A_60K", "A_60K_PHOTO")
SEEDS = ("2334", "2335")


def parse_test(log):
    """F1 of the last complete TEST RESULTS block, or None."""
    blocks = TEST.findall(log)
    if not blocks:
        return None
    found = F1.findall(blocks[-1])
    return float(found[0]) if found else None


def collect(root):
    """{(recipe, dataset, seed): F1} from ``<root>/<arm>/<dataset>/train_log.txt``."""
    scores = {}
    for arm_dir in sorted(root.glob("A_*")):
        recipe, seed = arm_dir.name.rsplit("-", 1)
        if recipe not in RECIPES or seed not in SEEDS:
            continue
        for run in sorted(arm_dir.glob("*-CD-256")):
            log = run / "train_log.txt"
            if not log.exists():
                continue
            value = parse_test(log.read_text(encoding="utf-8", errors="replace"))
            if value is not None:
                scores[(recipe, run.name, seed)] = value
    return scores


def analyse(scores):
    """Per-dataset sigma, the paired table, and the best recipe per dataset."""
    datasets = sorted({dataset for _, dataset, _ in scores})
    report = {"pairs": {}, "sigma": {}, "gate": {}, "best": {}}
    pooled = []
    for dataset in datasets:
        rows = []
        for recipe in RECIPES:
            pair = [scores.get((recipe, dataset, seed)) for seed in SEEDS]
            if any(value is None for value in pair):
                continue
            rows.append({"recipe": recipe, "seed_%s" % SEEDS[0]: pair[0],
                         "seed_%s" % SEEDS[1]: pair[1], "delta": pair[1] - pair[0]})
            pooled.append(abs(pair[1] - pair[0]))
        report["pairs"][dataset] = rows
        if rows:
            sigma = math.sqrt(sum(row["delta"] ** 2 for row in rows) / len(rows))
            means = {row["recipe"]: (row["seed_%s" % SEEDS[0]] + row["seed_%s" % SEEDS[1]]) / 2
                     for row in rows}
            best = max(means, key=means.get)
            report["sigma"][dataset] = sigma
            report["gate"][dataset] = max(0.36, 1.5 * sigma)
            report["best"][dataset] = {"recipe": best, "mean_f1": means[best],
                                       "runners_up": sorted(means.values(), reverse=True)[1:2]}
    if pooled:
        report["sigma_pooled"] = math.sqrt(sum(value ** 2 for value in pooled) / len(pooled))
        report["gate_pooled"] = max(0.36, 1.5 * report["sigma_pooled"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/CASA-MinorGoal/Run10")
    parser.add_argument("--json", default="outputs/CASA-MinorGoal/Run10/gates.json")
    args = parser.parse_args()
    scores = collect(Path(args.root))
    if not scores:
        raise SystemExit(f"no TEST RESULTS under {args.root}")
    report = analyse(scores)
    print(f"[SIGMA] {len(scores)} finished (arm, dataset) cells")
    for dataset, sigma in report["sigma"].items():
        rows = report["pairs"][dataset]
        print(f"[SIGMA] {dataset}: sigma={sigma:.3f} pp gate={report['gate'][dataset]:.3f} pp "
              f"from {len(rows)} seed pairs; best={report['best'][dataset]['recipe']} "
              f"({report['best'][dataset]['mean_f1']:.4f})")
        for row in rows:
            print(f"        {row['recipe']:<12} {row['seed_2334']:.4f} vs "
                  f"{row['seed_2335']:.4f}  delta {row['delta']:+.4f}")
    if "sigma_pooled" in report:
        print(f"[SIGMA] pooled sigma={report['sigma_pooled']:.3f} pp "
              f"gate={report['gate_pooled']:.3f} pp")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SIGMA] wrote {args.json}")


if __name__ == "__main__":
    main()
