"""Verify that --per-image-metrics does not move a single published number.

Runs the same frozen checkpoint twice on the same fixed test subset: once with the
plain ``evaluate`` call and once with the per-image collector, then compares the
returned score dictionaries entry by entry and the collected TP/FP/FN totals
against the confusion matrix.
"""
import argparse

import numpy as np
import torch

from analyze_geometry_degeneracy import DATASET_ROOT, DATASETS, MODELS, load_checkpoint, subset_loader
from train_CASA import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="DIFF", choices=sorted(MODELS))
    parser.add_argument("--dataset", default="WHU", choices=sorted(DATASETS))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model, path = load_checkpoint(MODELS[args.model], args.dataset)
    root = DATASET_ROOT / DATASETS[args.dataset]
    loader, indices = subset_loader(root, "test", args.limit)
    plain = evaluate(model, loader, device)[1]

    loader, _ = subset_loader(root, "test", args.limit)
    rows = []
    with_rows = evaluate(model, loader, device, per_image=rows)[1]

    assert list(plain) == list(with_rows), (list(plain), list(with_rows))
    drift = {key: float(with_rows[key]) - float(plain[key]) for key in plain}
    assert all(abs(value) == 0.0 for value in drift.values()), drift

    table = np.asarray(rows, dtype=np.int64)
    print(f"[CHECK] {path} {args.dataset} test[:{args.limit}] drifts all zero")
    print(f"[CHECK] rows={len(table)} tp={int(table[:, 1].sum())} "
          f"fp={int(table[:, 2].sum())} fn={int(table[:, 3].sum())} images={indices}")
    print(f"[CHECK] scores={ {key: round(float(value), 4) for key, value in plain.items()} }")


if __name__ == "__main__":
    main()
