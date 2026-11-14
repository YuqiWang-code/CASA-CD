"""Build the per-version ablation tables from the Run15 test blocks."""
import json
import re
from pathlib import Path

import config as C

BLOCK = re.compile(r'=== TEST RESULTS ===\n.*?\[MODE\] (?P<mode>\S+).*?F1=(?P<f1>[\d.]+).*?'
                   r'\[BEST-EPOCH\] (?P<epoch>\d+).*?=== END TEST RESULTS ===', re.S)


def read_f1(version, arm, dataset):
    path = C.run_dir(version, arm, dataset) / 'train_log.txt'
    if not path.is_file():
        return None
    blocks = BLOCK.findall(path.read_text(encoding='utf-8'))
    if not blocks:
        return None
    mode, f1, epoch = blocks[-1]
    return dict(f1=float(f1) * 100, best_epoch=int(epoch), mode=mode)


def version_table(version):
    rows = []
    control = read_f1(version, 'none', 'WHU-CD-256')
    for arm in C.arms(version):
        row = dict(arm=arm, label=C.label(version, arm))
        for dataset in C.DATASETS:
            entry = read_f1(version, arm, dataset)
            row[dataset] = entry['f1'] if entry else None
            row[f'{dataset}_epoch'] = entry['best_epoch'] if entry else None
        rows.append(row)
    baseline = {dataset: next((r[dataset] for r in rows if r['arm'] == 'none'), None) for dataset in C.DATASETS}
    for row in rows:
        row['delta'] = {dataset: (None if row[dataset] is None or baseline[dataset] is None
                                  else round(row[dataset] - baseline[dataset], 4)) for dataset in C.DATASETS}
    return rows, baseline


def main():
    payload, lines = {}, ['# Run15 — 三个历史版本的统一口径消融（WHU + SYSU，val 选模）', '']
    for version in C.VERSIONS:
        rows, baseline = version_table(version)
        payload[version] = dict(rows=rows, control=baseline, note=C.VERSIONS[version]['note'])
        lines += [f'## {version}', '', C.VERSIONS[version]['note'], '',
                  '| arm | 角色 | WHU F1 | Δ pp | SYSU F1 | Δ pp |',
                  '| --- | --- | --- | --- | --- | --- |']
        for row in rows:
            whu = '—' if row['WHU-CD-256'] is None else f"{row['WHU-CD-256']:.4f}"
            sysu = '—' if row['SYSU-CD-256'] is None else f"{row['SYSU-CD-256']:.4f}"
            dwhu = '—' if row['delta']['WHU-CD-256'] is None else f"{row['delta']['WHU-CD-256']:+.3f}"
            dsysu = '—' if row['delta']['SYSU-CD-256'] is None else f"{row['delta']['SYSU-CD-256']:+.3f}"
            lines.append(f"| {row['label']} (`{row['arm']}`) |  | {whu} | {dwhu} | {sysu} | {dsysu} |")
        lines.append('')
    (C.ROOT / 'comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (C.ROOT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
