"""Compare the Run2 arms: per-seed delta against the matched control, plus the Run1 column.

Every arm here is either a seed replication (with its own same-seed control) or a
2333 arm that can be read directly against the Run1 result, so the table keeps
both references side by side instead of pooling them.
"""
import argparse
import json
import os
import re

import config as C

BLOCK = re.compile(r'^=== TEST RESULTS ===\n(?P<body>.*?)\n=== END TEST RESULTS ===$', re.M | re.S)
F1 = re.compile(r'F1=(?P<value>[0-9.]+)')

CONTROL = {
    "CTRL_S2334": "CTRL_S2334",
    "SPP_S2334": "CTRL_S2334",
    "DIFF_S2334": "CTRL_S2334",
    "CTRL_S2335": "CTRL_S2335",
    "SPP_S2335": "CTRL_S2335",
    "DIFF_S2335": "CTRL_S2335",
    "SPP1": "CTRL",
    "DIFF_SPP1": "CTRL",
    "DIFF_NOSPP": "CTRL",
}
RUN1_ARM = {
    "SPP1": "SPP", "DIFF_SPP1": "DIFF", "DIFF_NOSPP": "DIFF",
    "SPP_S2334": "SPP", "SPP_S2335": "SPP",
    "DIFF_S2334": "DIFF", "DIFF_S2335": "DIFF",
    "CTRL_S2334": "CTRL", "CTRL_S2335": "CTRL",
}


def read_f1(root, arm, dataset):
    path = root / arm / dataset / 'train_log.txt'
    if not path.exists():
        return None
    blocks = BLOCK.findall(path.read_text(encoding='utf-8'))
    if not blocks:
        return None
    match = F1.search(blocks[-1])
    return None if match is None else float(match.group('value')) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('MINOR_LOCAL_ROOT', str(C.ROOT)))
    args = parser.parse_args()
    root = C.Path(args.root)
    table = {}
    for arm in C.ARM_ORDER:
        for dataset in C.DATASETS:
            table.setdefault(dataset, {})[arm] = read_f1(root, arm, dataset)

    lines = ['| dataset | arm | seed | F1 | control | delta pp | Run1 | delta vs Run1 pp |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    summary = {}
    for dataset in C.DATASETS:
        for arm in C.ARM_ORDER:
            value = table[dataset].get(arm)
            seed = C.seed_for(arm)
            control_name = CONTROL[arm]
            control = table[dataset].get(control_name)
            reference = C.RUN1_REFERENCE.get((RUN1_ARM[arm], dataset))
            if control is None and control_name == "CTRL":
                control = C.RUN1_REFERENCE.get(("CTRL", dataset))
            if value is None:
                lines.append(f'| {dataset} | {arm} | {seed} | — | — | — | — | — |')
                continue
            delta = None if control is None else value - control
            versus_run1 = None if reference is None else value - reference
            lines.append(
                f'| {dataset} | {arm} | {seed} | {value:.4f} | '
                f'{"—" if control is None else f"{control:.4f}"} | '
                f'{"—" if delta is None else f"{delta:+.3f}"} | '
                f'{"—" if reference is None else f"{reference:.4f}"} | '
                f'{"—" if versus_run1 is None else f"{versus_run1:+.3f}"} |')
            if delta is not None:
                summary.setdefault(arm, []).append(delta)

    lines.append('')
    lines.append('| arm | n | mean delta pp | per-dataset pp |')
    lines.append('| --- | --- | --- | --- |')
    for arm, values in summary.items():
        lines.append(f'| {arm} | {len(values)} | {sum(values) / len(values):+.3f} | '
                     f'{", ".join(f"{v:+.3f}" for v in values)} |')

    # The scientific question of this run: does the SPP/DIFF gain survive seeds?
    for base in ("SPP", "DIFF"):
        seeds = [f"{base}_S2334", f"{base}_S2335"]
        if all(any(table[d].get(arm) is not None for d in C.DATASETS) for arm in seeds):
            lines.append('')
            lines.append(f'**{base} across seeds** (delta against the same-seed control):')
            for dataset in C.DATASETS:
                row = [f'{dataset}']
                for arm in seeds:
                    value = table[dataset].get(arm)
                    control = table[dataset].get(CONTROL[arm])
                    row.append('—' if (value is None or control is None) else f'{value - control:+.3f}')
                lines.append('| ' + ' | '.join(row) + ' |')

    text = '\n'.join(lines) + '\n'
    (root / 'comparison.md').write_text(text, encoding='utf-8')
    (root / 'comparison.json').write_text(json.dumps(dict(table=table, deltas=summary), indent=2))
    print(text)


if __name__ == "__main__":
    main()
