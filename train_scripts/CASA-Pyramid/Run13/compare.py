"""Compare each ablation arm against its Run12 control arm and the seed noise floor."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASELINES, DATASETS, REPEATS, TRAINABLE_PARAMS, VARIANTS, experiments

BASE = ROOT/'outputs/CASA-Pyramid/Run13'
sys.path.insert(0, str(ROOT/'tools'))
from extract_metrics_to_excel import extract_test_block, parse_block


def f1_of(name):
    log = BASE/name/'train_log.txt'
    if not log.is_file():
        return None
    block = extract_test_block(log.read_text(encoding='utf-8'))
    return parse_block(block)['F1']*100 if block else None


def main():
    rows, complete = [], 0
    for name in experiments():
        variant, dataset = name.split('/')
        measured = f1_of(name)
        control, control_f1 = BASELINES[dataset]
        control_f1 *= 100
        repeats = [value*100 for _, value in REPEATS[dataset]]
        noise = max(repeats) - min(repeats) if len(repeats) > 1 else None
        if measured is None:
            rows.append(dict(dataset=dataset, arm=variant, f1=None, control_arm=control,
                             control_f1=round(control_f1, 4), delta_pp=None,
                             noise_pp=None if noise is None else round(noise, 4),
                             verdict='missing'))
            continue
        complete += 1
        delta = measured - control_f1
        if noise is None:
            verdict = ('exploratory: lower than the single control' if delta < 0
                       else 'exploratory: higher than the single control')
        elif delta < -noise:
            verdict = 'removal costs more than seed noise'
        elif abs(delta) <= noise:
            verdict = 'within seed noise'
        else:
            verdict = 'removal helps'
        rows.append(dict(dataset=dataset, arm=variant, f1=round(measured, 4), control_arm=control,
                         control_f1=round(control_f1, 4), delta_pp=round(delta, 4),
                         noise_pp=None if noise is None else round(noise, 4),
                         trainable_params=TRAINABLE_PARAMS[variant], verdict=verdict))
    payload = dict(protocol='casa_pyramid_40k_testselect_heads_detail_v12r3',
                   note='delta_pp = ablation F1 - Run12 control F1; noise_pp = same-config rerun spread',
                   complete=complete, experiments=len(rows), rows=rows)
    (BASE/'ablation_table.json').write_text(json.dumps(payload, indent=2))
    lines = ['| dataset | arm | F1 | control | control F1 | delta pp | noise pp | verdict |',
             '|---|---|---:|---|---:|---:|---:|---|']
    for row in rows:
        lines.append('| {dataset} | {arm} | {f1} | {control_arm} | {control_f1} | {delta_pp} | '
                     '{noise_pp} | {verdict} |'.format(**{k: ('—' if v is None else v)
                                                          for k, v in row.items()}))
    table = '\n'.join(lines)
    (BASE/'ablation_table.md').write_text(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
