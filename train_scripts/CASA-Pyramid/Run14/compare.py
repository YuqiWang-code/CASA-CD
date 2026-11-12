"""Compare Run14: paired H/E seeds, and the three-level TAR/DCR ladders."""
import json
import statistics
from pathlib import Path

from config import ARMS, DATASETS, LADDER, LADDER_DATASETS, SEEDS

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs/CASA-Pyramid/Run14'


def read_f1(name):
    log = BASE/name/'train_log.txt'
    if not log.is_file():
        return None
    text = log.read_text(encoding='utf-8')
    start, end = text.rfind('=== TEST RESULTS ==='), text.rfind('=== END TEST RESULTS ===')
    if start < 0 or end < start:
        return None
    for line in text[start:end].splitlines():
        if line.startswith('F1='):
            return float(line.split('F1=')[1].split()[0])
        if '| F1=' in line:
            return float(line.split('F1=')[1].split()[0])
    return None


def pairing():
    rows, deltas = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            hyperbolic = read_f1(f'H_S{seed}/{dataset}')
            euclidean = read_f1(f'E_S{seed}/{dataset}')
            if hyperbolic is None or euclidean is None:
                rows.append(dict(dataset=dataset, seed=seed, hyperbolic=hyperbolic,
                                 euclidean=euclidean, delta_pp=None))
                continue
            delta = 100*(hyperbolic - euclidean)
            deltas.append(delta)
            rows.append(dict(dataset=dataset, seed=seed, hyperbolic=100*hyperbolic,
                             euclidean=100*euclidean, delta_pp=delta))
    per_dataset = {}
    for dataset in DATASETS:
        values = [row['delta_pp'] for row in rows if row['dataset'] == dataset and row['delta_pp'] is not None]
        per_dataset[dataset] = dict(n=len(values), mean_pp=statistics.fmean(values) if values else None,
                                    sd_pp=statistics.stdev(values) if len(values) > 1 else None)
    summary = dict(H_minus_E=dict(n=len(deltas),
                                  mean_pp=statistics.fmean(deltas) if deltas else None,
                                  sd_pp=statistics.stdev(deltas) if len(deltas) > 1 else None,
                                  std_error_pp=(statistics.stdev(deltas)/len(deltas)**.5
                                                if len(deltas) > 1 else None)),
                   per_dataset=per_dataset, rows=rows,
                   protocol='epoch selection on val, single threshold 0.5 on test')
    return summary


def ladders():
    table = {}
    for dataset in LADDER_DATASETS:
        anchor = LADDER[dataset]
        control = 100*anchor['control'][1] if anchor['control'][1] < 1 else anchor['control'][1]
        tar_plain = 100*anchor['tar_plain'][1] if anchor['tar_plain'][1] < 1 else anchor['tar_plain'][1]
        dcr_plain = 100*anchor['dcr_plain'][1] if anchor['dcr_plain'][1] < 1 else anchor['dcr_plain'][1]
        tar_diff, dcr_single = read_f1(f'TAR_DIFF/{dataset}'), read_f1(f'DCR_SINGLE/{dataset}')
        table[dataset] = dict(
            noise_pp=anchor['noise_pp'],
            tar=dict(control=control, plain=tar_plain, plain_name=anchor['tar_plain'][0],
                     none=100*tar_diff if tar_diff is not None else None,
                     none_minus_plain_pp=100*(tar_diff - tar_plain/100)
                     if tar_diff is not None else None),
            dcr=dict(control=control, plain=dcr_plain, plain_name=anchor['dcr_plain'][0],
                     none=100*dcr_single if dcr_single is not None else None,
                     none_minus_plain_pp=100*(dcr_single - dcr_plain/100)
                     if dcr_single is not None else None),
            protocol='epoch selection on test (v12r3)', )
    return table


def markdown(pair, ladder):
    lines = ['# Run14: paired geometry seeds and module-removal ladders', '',
             '## A. Hyperbolic vs matched Euclidean (val-selected epoch)', '',
             '| dataset | seed | H F1 | E F1 | H - E (pp) |', '|---|---|---|---|---|']
    for row in pair['rows']:
        values = ('-' if row['hyperbolic'] is None else f"{row['hyperbolic']:.4f}",
                  '-' if row['euclidean'] is None else f"{row['euclidean']:.4f}",
                  '-' if row['delta_pp'] is None else f"{row['delta_pp']:+.3f}")
        lines.append(f"| {row['dataset']} | {row['seed']} | {values[0]} | {values[1]} | {values[2]} |")
    total = pair['H_minus_E']
    lines += ['', '| dataset | pairs | mean H - E (pp) | sd (pp) |', '|---|---|---|---|']
    for dataset, values in pair['per_dataset'].items():
        mean = '-' if values['mean_pp'] is None else f"{values['mean_pp']:+.3f}"
        sd = '-' if values['sd_pp'] is None else f"{values['sd_pp']:.3f}"
        lines.append(f"| {dataset} | {values['n']} | {mean} | {sd} |")
    mean = '-' if total['mean_pp'] is None else f"{total['mean_pp']:+.3f}"
    sd = '-' if total['sd_pp'] is None else f"{total['sd_pp']:.3f}"
    lines += ['', f"All pairs: n={total['n']}, mean {mean} pp, sd {sd} pp.", '']
    lines += ['## B. Module removal (test-selected epoch, v12r3)', '',
              '| dataset | control | TAR plain | TAR diff | diff - plain (pp) | DCR plain | DCR single | single - plain (pp) |',
              '|---|---|---|---|---|---|---|---|']
    for dataset, values in ladder.items():
        tar, dcr = values['tar'], values['dcr']
        def fmt(x):
            return '-' if x is None else f"{x:.4f}"
        tar_delta = '-' if tar['none_minus_plain_pp'] is None else f"{tar['none_minus_plain_pp']:+.3f}"
        dcr_delta = '-' if dcr['none_minus_plain_pp'] is None else f"{dcr['none_minus_plain_pp']:+.3f}"
        lines.append(f"| {dataset} | {fmt(tar['control'])} | {fmt(tar['plain'])} | {fmt(tar['none'])} | {tar_delta} "
                     f"| {fmt(dcr['plain'])} | {fmt(dcr['none'])} | {dcr_delta} |")
    lines += ['', 'Removal-delta is against the Run13 plain arm, which already deleted the foldable branches,',
              'so a negative delta means the module content itself is worth its cost.', '']
    return '\n'.join(lines)


def main():
    pair, ladder = pairing(), ladders()
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE/'comparison.json').write_text(json.dumps(dict(pairing=pair, ladders=ladder), indent=2),
                                        encoding='utf-8')
    (BASE/'comparison.md').write_text(markdown(pair, ladder) + '\n', encoding='utf-8')
    print(json.dumps(pair['H_minus_E']), flush=True)
    print(markdown(pair, ladder), flush=True)


if __name__ == '__main__':
    main()
