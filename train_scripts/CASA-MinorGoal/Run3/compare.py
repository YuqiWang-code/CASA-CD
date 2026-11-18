"""Score Run3 against the pre-registered gates.

Four jobs, four questions, one table:

* ``ARM_C`` vs the same-seed Run2 ``SPP`` reference -- does the context module
  itself add anything once the KV content is the plain fused feature?
* ``SUPP_PIX`` vs ``ARM_C`` on SYSU -- is the pixel-area support prior a second
  lever on top of the module?
* ``DETAIL`` vs the Run1 ``DIFF`` reference on LEVIR -- does the boundary branch
  move total F1 *and* the boundary/interior split?

The gates live in ``config.GATES`` / ``config.LEVIR_DETAIL_GATES`` and were fixed
before the queue started; this script only applies them.
"""
import argparse
import json
import os
import re

import config as C

BLOCK = re.compile(r'^=== TEST RESULTS ===\n(?P<body>.*?)\n=== END TEST RESULTS ===$', re.M | re.S)
F1 = re.compile(r'F1=(?P<value>[0-9.]+)')

JOB_REFERENCE = {
    ("ARM_C", "WHU-CD-256"): ("SPP", "WHU-CD-256", 2334),
    ("ARM_C", "SYSU-CD-256"): ("SPP", "SYSU-CD-256", 2334),
    ("DETAIL", "LEVIR-CD-256"): ("DIFF", "LEVIR-CD-256", 2333),
}


def read_result(root, arm, dataset):
    directory = root / arm / dataset
    payload = {}
    result = directory / 'result.json'
    if result.is_file():
        payload = json.loads(result.read_text(encoding='utf-8'))
        f1 = payload.get('metrics', {}).get('F1')
        payload['f1'] = None if f1 is None else f1 * 100
    if payload.get('f1') is None:
        log = directory / 'train_log.txt'
        if log.is_file():
            blocks = BLOCK.findall(log.read_text(encoding='utf-8'))
            match = F1.search(blocks[-1]) if blocks else None
            payload['f1'] = None if match is None else float(match.group('value')) * 100
    return payload


def ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def verdict(delta, gate):
    if delta is None:
        return 'missing'
    if delta >= gate['minimum']:
        return 'effect confirmed'
    if delta <= -gate['minimum']:
        return 'effect reversed (removal would help)'
    if abs(delta) <= gate['noise']:
        return 'no effect (inside the seed noise)'
    return 'direction only (below the pre-registered minimum)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.environ.get('MINOR_LOCAL_ROOT', str(C.ROOT)))
    args = parser.parse_args()
    root = C.Path(args.root)

    rows, table = [], {}
    for arm, dataset in C.JOBS:
        payload = read_result(root, arm, dataset)
        f1 = payload.get('f1')
        spatial = payload.get('spatial_diagnostics') or {}
        boundary = (spatial.get('boundary_0px') or {}).get('F1')
        interior = (spatial.get('interior') or {}).get('F1')
        boundary_px = (spatial.get('boundary_0px') or {}).get('target_pixels')
        interior_px = (spatial.get('interior') or {}).get('target_pixels')
        share = ratio(boundary_px, (boundary_px or 0) + (interior_px or 0))
        record = dict(arm=arm, dataset=dataset, seed=C.seed_for(arm), f1=f1,
                      boundary_0px=None if boundary is None else boundary * 100,
                      interior=None if interior is None else interior * 100,
                      positive_share=share)
        key = JOB_REFERENCE.get((arm, dataset))
        if key is not None:
            record['reference_name'] = f"{key[0]}/{key[1]}/s{key[2]}"
            record['reference'] = C.REFERENCE.get(key)
            record['delta_pp'] = (None if f1 is None or record['reference'] is None
                                  else f1 - record['reference'])
            record['verdict'] = verdict(record['delta_pp'], C.GATES[arm])
        elif arm == 'SUPP_PIX':
            base = table.get((arm, dataset)) or {}
            control = read_result(root, 'ARM_C', dataset).get('f1')
            record['reference_name'] = f"ARM_C/{dataset}/s{C.seed_for('ARM_C')}"
            record['reference'] = control
            record['delta_pp'] = None if f1 is None or control is None else f1 - control
            record['verdict'] = verdict(record['delta_pp'], C.GATES['SUPP_PIX'])
        table[(arm, dataset)] = record
        rows.append(record)

    detail = table.get(('DETAIL', 'LEVIR-CD-256'))
    if detail is not None and detail['f1'] is not None:
        gates = C.LEVIR_DETAIL_GATES
        detail['gates'] = dict(
            total=dict(value=detail['f1'], minimum=gates['total'],
                       passed=detail['f1'] >= gates['total']),
            boundary=dict(value=detail['boundary_0px'], minimum=gates['boundary'],
                          passed=None if detail['boundary_0px'] is None
                          else detail['boundary_0px'] >= gates['boundary']),
            interior=dict(value=detail['interior'], minimum=gates['interior'],
                          passed=None if detail['interior'] is None
                          else detail['interior'] >= gates['interior']),
            positive_share=dict(value=detail['positive_share'],
                                reference=gates['positive_share']))
        passed = [item['passed'] for item in detail['gates'].values()
                  if isinstance(item, dict) and item.get('passed') is not None]
        detail['gate_passed'] = bool(passed) and all(passed)

    def fmt(value, pattern="{:.4f}", missing="—"):
        return missing if value is None else pattern.format(value)

    lines = ['| arm | dataset | seed | F1 | reference | delta pp | verdict |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for record in rows:
        reference = record.get('reference_name', '—')
        if record.get('reference') is not None:
            reference = reference + ' = ' + fmt(record['reference'])
        lines.append('| ' + ' | '.join([
            record['arm'], record['dataset'], str(record['seed']),
            fmt(record['f1']), reference,
            fmt(record.get('delta_pp'), '{:+.3f}'),
            record.get('verdict', '—')]) + ' |')
    if detail is not None and detail.get('gates'):
        lines.append('')
        lines.append('**DETAIL boundary gates (LEVIR, seed 2333)**')
        lines.append('| metric | value | minimum | passed |')
        lines.append('| --- | --- | --- | --- |')
        for name in ('total', 'boundary', 'interior'):
            gate = detail['gates'][name]
            lines.append('| ' + ' | '.join([
                name, fmt(gate['value']), fmt(gate['minimum']), str(gate['passed'])]) + ' |')
        share = detail['gates']['positive_share']
        lines.append('| ' + ' | '.join([
            'boundary share of positives', fmt(share['value']),
            fmt(share['reference']) + ' (reference)', '—']) + ' |')
        lines.append('')
        lines.append('**DETAIL gate passed: {}**'.format(detail['gate_passed']))

    text = '\n'.join(lines) + '\n'
    (root / 'attribution.md').write_text(text, encoding='utf-8')
    (root / 'attribution.json').write_text(json.dumps(dict(rows=rows, gates=C.GATES), indent=2))
    print(text)


if __name__ == '__main__':
    main()
