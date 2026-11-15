"""Build the per-version module-removal table for Run16.

Every arm is compared with the *same version's* Run15 validation-selected
``none`` control, so a delta answers "what does this stage contribute on this
line?" rather than "which version scores higher".
"""
import json
import re
from pathlib import Path

import config as C

BLOCK = re.compile(r'=== TEST RESULTS ===\n.*?\[MODE\] (?P<mode>\S+).*?F1=(?P<f1>[\d.]+).*?'
                   r'\[BEST-EPOCH\] (?P<epoch>\d+).*?=== END TEST RESULTS ===', re.S)


def read_f1(directory):
    path = Path(directory) / 'train_log.txt'
    if not path.is_file():
        return None
    blocks = BLOCK.findall(path.read_text(encoding='utf-8'))
    if not blocks:
        return None
    mode, f1, epoch = blocks[-1]
    return dict(f1=float(f1) * 100, best_epoch=int(epoch), mode=mode)


def main():
    payload, lines = {}, ['# Run16 — TAR / DCR 模块移除在三份历史版本上的复现（WHU + SYSU，val 选模）', '']
    summary = {arm: [] for arm in C.arms()}
    for version in C.VERSIONS:
        rows = []
        for arm in C.arms():
            row = dict(arm=arm, label=C.label(version, arm), note=C.ARMS[arm]['note'])
            for dataset in C.DATASETS:
                entry = read_f1(C.run_dir(version, arm, dataset))
                control = read_f1(C.baseline_dir(version, dataset))
                row[dataset] = entry['f1'] if entry else None
                row[f'{dataset}_epoch'] = entry['best_epoch'] if entry else None
                row[f'{dataset}_control'] = control['f1'] if control else None
                if entry and control:
                    delta = round(entry['f1'] - control['f1'], 4)
                    summary[arm].append((version, dataset, delta))
                else:
                    delta = None
                row[f'{dataset}_delta'] = delta
            rows.append(row)
        payload[version] = dict(rows=rows, note=C.VERSIONS[version]['note'])
        lines += [f'## {version}', '', C.VERSIONS[version]['note'], '',
                  '| arm | WHU F1 | 控制 | Δ pp | SYSU F1 | 控制 | Δ pp |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            cells = [f"`{row['arm']}`"]
            for dataset in C.DATASETS:
                f1 = '—' if row[dataset] is None else f"{row[dataset]:.4f}"
                control = '—' if row[f'{dataset}_control'] is None else f"{row[f'{dataset}_control']:.4f}"
                delta = '—' if row[f'{dataset}_delta'] is None else f"{row[f'{dataset}_delta']:+.3f}"
                cells += [f1, control, delta]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## 汇总', '', '| arm | n | 平均 Δ pp | 逐项 Δ pp |', '| --- | ---: | ---: | --- |']
    payload['summary'] = {}
    for arm, entries in summary.items():
        deltas = [delta for _, _, delta in entries]
        mean = round(sum(deltas) / len(deltas), 4) if deltas else None
        payload['summary'][arm] = dict(n=len(deltas), mean_delta_pp=mean,
                                       entries=[dict(version=v, dataset=d, delta_pp=x)
                                                for v, d, x in entries])
        shown = ', '.join(f'{x:+.3f}' for x in deltas) or '—'
        lines.append(f"| `{arm}` | {len(deltas)} | {'—' if mean is None else f'{mean:+.3f}'} | {shown} |")
    lines.append('')
    (C.ROOT / 'comparison.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                            encoding='utf-8')
    (C.ROOT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
