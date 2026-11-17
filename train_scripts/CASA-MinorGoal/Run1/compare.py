"""Compare the Run1 arms: per-dataset F1, delta against the CTRL arm."""
import argparse
import json
import os
import re

import config as C

BLOCK = re.compile(r'^=== TEST RESULTS ===\n(?P<body>.*?)\n=== END TEST RESULTS ===$', re.M | re.S)
F1 = re.compile(r'F1=(?P<value>[0-9.]+)')


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
    lines = ['| dataset | arm | F1 | control | delta pp |', '| --- | --- | --- | --- | --- |']
    summary = {}
    for dataset in C.DATASETS:
        control = table[dataset].get('CTRL')
        for arm in C.ARM_ORDER:
            value = table[dataset].get(arm)
            if value is None:
                lines.append(f'| {dataset} | {arm} | — | — | — |')
                continue
            delta = None if (control is None or arm == 'CTRL') else value - control
            lines.append(f'| {dataset} | {arm} | {value:.4f} | '
                         f'{"—" if control is None else f"{control:.4f}"} | '
                         f'{"—" if delta is None else f"{delta:+.3f}"} |')
            if delta is not None:
                summary.setdefault(arm, []).append(delta)
    lines.append('')
    lines.append('| arm | n | mean delta pp | per-dataset pp |')
    lines.append('| --- | --- | --- | --- |')
    for arm, values in summary.items():
        lines.append(f'| {arm} | {len(values)} | {sum(values) / len(values):+.3f} | '
                     f'{", ".join(f"{v:+.3f}" for v in values)} |')
    text = '\n'.join(lines) + '\n'
    (root / 'comparison.md').write_text(text, encoding='utf-8')
    (root / 'comparison.json').write_text(json.dumps(dict(table=table, deltas=summary), indent=2))
    print(text)


if __name__ == "__main__":
    main()
