"""Publish one completed experiment round's logs, epoch tables and curves."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['LEVIR-CD-256', 'SYSU-CD-256', 'WHU-CD-256', 'CDD-CD-256']
VARIANTS = {'M0_CTRL': 'CTRL', 'M1_HYPER_QKV': 'Hyperbolic', 'M2_CA_QKV': 'Euclidean'}
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def values(line):
    return {name: float(value) for name, value in re.findall(r'(\w+)=(' + NUMBER + ')', line)}


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_log(text):
    epochs, diagnostics = {}, []
    current = None
    for line in text.splitlines():
        if line.startswith('[TRAIN-EPOCH] '):
            data = values(line)
            current = int(data.pop('epoch'))
            row = epochs.setdefault(current, {'epoch': current})
            row.update({'train_' + key: value for key, value in data.items()})
        match = re.match(r'Epoch (\d+)/(\d+) \| ', line)
        if match:
            current = int(match[1])
            row = epochs.setdefault(current, {'epoch': current})
            row.update({'val_' + key: value for key, value in values(line).items()})
        if line.startswith('[MODULE-DIAG] '):
            diagnostics.append(dict(epoch=current, stages=json.loads(line.split(' ', 1)[1])))
        if line.startswith('[SPATIAL-DIAG] '):
            spatial = json.loads(line.split(' ', 1)[1])
            row = epochs.setdefault(int(spatial['epoch']), {'epoch': int(spatial['epoch'])})
            for region in ('boundary_1px', 'boundary_2px', 'interior'):
                row.update({f'val_{region}_{name}': value for name, value in spatial[region].items()})
    return [epochs[key] for key in sorted(epochs)], diagnostics


def diagnostic_rows(events):
    rows = []
    for event in events:
        for stage, data in event['stages'].items():
            row = dict(epoch=event['epoch'], stage=stage)
            for name, value in data.items():
                if isinstance(value, list):
                    if name == 'scale_mass' or name.endswith('_scale_mass'):
                        for head, masses in enumerate(value):
                            for source_scale, mass in enumerate(masses, 1):
                                row[f'{name}_head{head}_source{source_scale}'] = mass
                    elif name == 'class_attention_mass':
                        for head, masses in enumerate(value):
                            for label, mass in zip(('change', 'background'), masses):
                                row[f'class_attention_mass_head{head}_{label}'] = mass
                    elif name.startswith('prototype_reliability_'):
                        for label, item in zip(('change', 'background'), value):
                            row[f'{name}_{label}'] = item
                    else:
                        for head, item in enumerate(value):
                            row[f'{name}_head{head}'] = item
                else:
                    row[name] = value
            rows.append(row)
    return rows


def curves(destination, dataset, records, selection_split):
    colors = {'M0_CTRL': '#555555', 'M1_HYPER_QKV': '#2266AA', 'M2_CA_QKV': '#D47718'}
    manifest = next(iter(records.values()))[2]
    guide_weight, boundary_weight = manifest.get('guide_weight', 0.2), manifest.get('boundary_weight', 0.0)
    has_boundary = boundary_weight != 0
    fig, axes = plt.subplots(3 if has_boundary else 2, 2,
                             figsize=(12, 11 if has_boundary else 7.8), constrained_layout=True)
    main_axis, f1_axis, guide_axis = axes.flat[:3]
    boundary_axis, total_axis, spatial_axis = axes.flat[3:] if has_boundary else (None, axes.flat[3], None)
    evaluation_label = 'test selection' if selection_split == 'test' else 'validation'
    for variant, (rows, result, _) in records.items():
        color, label = colors[variant], VARIANTS[variant]
        x = [row['epoch'] for row in rows]
        main_axis.plot(x, [row['train_main'] for row in rows], color=color, label=label + ' train')
        validation = [row for row in rows if 'val_F1' in row]
        vx = [row['epoch'] for row in validation]
        main_axis.plot(vx, [row['val_Loss'] for row in validation], color=color, ls='--', alpha=.85,
                        label=label + ' ' + evaluation_label)
        f1_axis.plot(vx, [row['val_F1'] * 100 for row in validation], color=color, label=label)
        best = next(row for row in validation if row['epoch'] == result['best_epoch'])
        f1_axis.plot(best['epoch'], best['val_F1'] * 100, marker='o', color=color, ms=4)
        guide_axis.plot(x, [row['train_guide'] for row in rows], color=color, label=label)
        total_axis.plot(x, [row['train_loss'] for row in rows], color=color, label=label)
        if has_boundary:
            boundary_axis.plot(x, [row['train_boundary'] for row in rows], color=color, label=label)
            spatial = [row for row in rows if 'val_boundary_1px_F1' in row]
            sx = [row['epoch'] for row in spatial]
            spatial_axis.plot(sx, [row['val_boundary_1px_F1'] * 100 for row in spatial],
                              color=color, label=label + ' boundary 1px')
            spatial_axis.plot(sx, [row['val_interior_recall'] * 100
                                  if row['val_interior_recall'] is not None else float('nan')
                                  for row in spatial], color=color, ls='--', label=label + ' interior recall')
    formula = f'Main + {guide_weight:g} x guide'
    if has_boundary:
        formula += f' + {boundary_weight:g} x boundary'
    titles = [('Main loss: train / ' + evaluation_label, 'BCE + Dice (four outputs)'),
              (evaluation_label.capitalize() + ' F1; dots mark selected checkpoints', 'F1 (%)'),
              (f'Guide loss before the {guide_weight:g} multiplier', 'Guide BCE (four scales)')]
    if has_boundary:
        titles += [(f'Boundary loss before the {boundary_weight:g} multiplier', 'Boundary BCE + Dice (two scales)')]
    titles.append(('Total training loss', formula))
    if has_boundary:
        titles.append(('Boundary / interior diagnostics on ' + evaluation_label, 'Score (%)'))
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Epoch (0-based)', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(dataset + ' | ' + destination.parent.name + '/' + destination.name, fontsize=14)
    fig.savefig(destination / 'curves' / (dataset + '.png'), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    source = ROOT / 'outputs' / args.tag / args.run
    destination = ROOT / 'docs/experiments' / args.tag / args.run
    (destination / 'curves').mkdir(parents=True, exist_ok=True)
    state = json.loads((source / 'pipeline_status.json').read_text())
    variants = {name: VARIANTS[name] for name in state.get('variants', VARIANTS)}
    if state['status'] != 'complete':
        raise RuntimeError('export requires a completed queue')
    summary, table = [], []
    for dataset in DATASETS:
        records = {}
        for variant in variants:
            original = source / variant / dataset
            published = destination / variant / dataset
            text = (original / 'train_log.txt').read_text(encoding='utf-8')
            if not re.search(r'=== TEST RESULTS ===.*?=== END TEST RESULTS ===', text, re.S):
                raise RuntimeError(f'no complete TEST RESULTS: {variant}/{dataset}')
            result = json.loads((original / 'result.json').read_text())
            manifest = json.loads((original / 'run_manifest.json').read_text())
            rows, diagnostics = parse_log(text)
            assert len(rows) == manifest['epochs'], (variant, dataset, len(rows))
            evaluation_start = manifest.get('evaluation_starts_at_epoch', 1)
            selection_split = manifest.get('selection_split', 'val')
            expected_evaluation_epochs = set(range(evaluation_start, manifest['epochs']))
            assert {row['epoch'] for row in rows if 'val_F1' in row} == expected_evaluation_epochs
            if variant != 'M0_CTRL':
                assert len(diagnostics) == len(expected_evaluation_epochs)
                assert {event['epoch'] for event in diagnostics} == expected_evaluation_epochs
            best = next(row for row in rows if row['epoch'] == result['best_epoch'])
            assert best['val_F1'] == max(row['val_F1'] for row in rows if 'val_F1' in row)
            published.mkdir(parents=True, exist_ok=True)
            for filename in ('train_log.txt', 'run_manifest.json', 'evaluation_manifest.json',
                             'result.json', 'deploy_check.json', 'arch.json', 'placement.json'):
                shutil.copyfile(original / filename, published / filename)
            write_csv(published / 'epochs.csv', rows)
            with (published / 'module_diagnostics.jsonl').open('w', encoding='utf-8') as file:
                for event in diagnostics:
                    file.write(json.dumps(event, ensure_ascii=False) + '\n')
            if diagnostics:
                write_csv(published / 'module_diagnostics.csv', diagnostic_rows(diagnostics))
            records[variant] = rows, result, manifest
            row = dict(dataset=dataset, variant=variant, best_epoch=result['best_epoch'],
                       selection_split=selection_split,
                       independent_test=manifest.get('independent_test', selection_split != 'test'),
                       evaluation_starts_at_epoch=evaluation_start,
                       best_val_F1_pct=best['val_F1'] * 100, optimizer_steps=result['optimizer_steps'],
                       **{'test_' + key + '_pct': result['metrics'][key] * 100
                          for key in ('recall', 'precision', 'OA', 'F1', 'IoU', 'Kappa')},
                       training_params_M=result['total_params_m'], deploy_params_M=result['deploy_params_m'],
                       deploy_max_diff=result['deploy_max_diff'],
                       deploy_binary_disagreement=result['deploy_binary_disagreement'])
            summary.append(row)
            link = f'{variant}/{dataset}'
            table.append(f"| {dataset.split('-')[0]} | {variant} | {result['best_epoch']} | "
                         f"{row['best_val_F1_pct']:.3f} | {row['test_F1_pct']:.3f} | "
                         f"[日志]({link}/train_log.txt) / [逐轮]({link}/epochs.csv) / "
                         f"[结果]({link}/result.json) |")
        curves(destination, dataset, records, selection_split)
    write_csv(destination / 'summary.csv', summary)
    source_record = 'source_manifest.json' if (source / 'source_manifest.json').exists() else 'source_baseline.json'
    inference_record = 'inference_benchmark.json' if (source / 'inference_benchmark.json').exists() else 'inference.json'
    for filename in ('environment.json', source_record, 'pipeline_status.json', 'pipeline.log',
                     'scheduling.json', 'profile.json', inference_record):
        shutil.copyfile(source / filename, destination / filename)
    for filename in ('validation.log', 'startup_health.json'):
        if (source / filename).exists():
            shutil.copyfile(source / filename, destination / filename)
    probe_names = [path.name for path in sorted((source / 'probes').iterdir())
                   if (path / 'health_result.json').exists()]
    for name in probe_names:
        published = destination / 'probes' / name
        published.mkdir(parents=True, exist_ok=True)
        for filename in ('probe.log', 'run_manifest.json', 'health_history.json', 'health_result.json'):
            shutil.copyfile(source / 'probes' / name / filename, published / filename)
    baseline = json.loads((source / source_record).read_text())
    source_commit = baseline['code_baseline'] if source_record == 'source_manifest.json' else baseline['model_training_commit']
    started = min(json.loads((source / variant / dataset / 'placement.json').read_text())['started']
                  for dataset in DATASETS for variant in variants)
    evaluation_name = 'test选模' if selection_split == 'test' else '验证'
    selection_note = ('每个epoch在test集评估并按最高F1选模，相同F1更新；最后用选中的checkpoint在同一test集复测。'
                      'test参与checkpoint选择，最终数值不属于独立盲测。'
                      if selection_split == 'test' else
                      '最高验证F1选模，相同F1更新；测试使用被选中的checkpoint。')
    epoch_note = (f'逐轮评估从epoch{evaluation_start}开始。' if evaluation_start == 0 else
                  f'逐轮评估从epoch{evaluation_start}开始；此前评估字段为空，空白不是0。')
    curve_note = ('圆点为实际选中checkpoint的test选模F1；最终折叠模型的复测结果单独保存在result.json。'
                  if selection_split == 'test' else
                  '圆点为实际选中checkpoint的验证F1，正式测试值不混入验证曲线。')
    guide_weight, boundary_weight = manifest.get('guide_weight', 0.2), manifest.get('boundary_weight', 0.0)
    loss_formula = f'main+{guide_weight:g}×guide'
    loss_note = '评分损失单独展示'
    boundary_note = ''
    if boundary_weight:
        loss_formula += f'+{boundary_weight:g}×boundary'
        loss_note = f'评分与边界损失分别展示未乘权重的值，权重由manifest给出（{guide_weight:g}、{boundary_weight:g}）'
        boundary_note = ('\n- `train_boundary`是两细尺度边界BCEWithLogits+Dice之和，未乘边界权重；'
                         '`train_encoder_lr`记录预训练骨干实际学习率。'
                         '`val_boundary_1px_*`、`val_boundary_2px_*`与`val_interior_*`来自整轮评估集的累计统计：'
                         '边界为replicate填充的3×3形态学梯度，容差按输入像素的Chebyshev距离计算；'
                         '内部统计排除GT膨胀2像素减腐蚀2像素的边界带，无正类内部时Recall为空。')
    readme = f'''# {args.tag} / {args.run} 完整训练记录

协议`{state['protocol']}`，模型/训练源码基线`{source_commit}`。本目录保存{len(summary)}项正式实验的实际日志、逐轮指标、模块诊断和最终测试，来源为`outputs/{args.tag}/{args.run}`。本轮于{started} CST启动，{state['updated']} CST完成。

## 正式结果与最佳epoch

epoch沿用日志的0起始索引，例如177表示第178轮。{selection_note}`summary.csv`提供选模split、独立测试标记、全部Recall/Precision/OA/F1/IoU/Kappa、参数与部署误差。

| 数据集 | 变体 | best_epoch (0起始) | 最佳{evaluation_name}F1(%) | 最终test复测F1(%) | 实际记录 |
|---|---|---:|---:|---:|---|
''' + '\n'.join(table) + f'''

## 文件与数值口径

- 每项的`train_log.txt`是完整原始日志，包括训练后期、{evaluation_name}评估行、实际MODULE-DIAG及最终TEST RESULTS。
- `epochs.csv`保存每轮训练总损失、主损失、评分损失、学习率，以及{evaluation_name}Loss/Recall/Precision/OA/F1/IoU/Kappa和耗时；指标沿用日志的0～1单位。{epoch_note}{boundary_note}
- CSV沿用`val_`列名前缀和`best_val_F1_pct`字段，数据来源由`summary.csv`及manifest的`selection_split`明确：本轮为`{selection_split}`。字段前缀不表示独立验证集。
- `module_diagnostics.jsonl`按epoch保存四级完整诊断；`module_diagnostics.csv`按epoch/stage展开温度各头、gamma、归一化注意力熵、原型分配TV、预测变化概率、分支RMS、跨尺度注意力质量，以及双曲半径和裁剪比例。源日志未记录的量不补造。
- MODULE-DIAG来自当轮{evaluation_name}的首个batch，不是整个评估集的统计。温度跨四尺度共享；`scale_mass_head0_source1`表示第0头分配给第1尺度KV的注意力质量。CTRL没有注意力模块，诊断JSONL为空。
- `result.json`保留完整精度的正式测试与实际最佳epoch；`run_manifest.json`、`evaluation_manifest.json`和`deploy_check.json`给出配置及折叠检查，`placement.json`记录物理GPU。`arch.json`记录结构配置。
- 根目录的`environment.json`、`{source_record}`、`pipeline_status.json`、`pipeline.log`及`scheduling.json`保存实际环境、源码基线和调度记录；`profile.json`与`{inference_record}`记录显存和推理测量。
- `probes/`保存{len(probe_names)}项成功健康探针的完整日志、逐轮损失和任务梯度；探针不作为正式指标。

## 完整训练与{evaluation_name}曲线

曲线逐epoch绘制，不平滑、不省略后期。主损失图比较相同定义的训练/{evaluation_name}BCE+Dice；{loss_note}，总训练损失为{loss_formula}。{curve_note}

'''
    for dataset in DATASETS:
        readme += f'### {dataset}\n\n![{dataset}训练与{evaluation_name}曲线](curves/{dataset}.png)\n\n'
    readme += '## 关联材料\n\n[运行说明](../../../../train_scripts/' + args.tag + '/' + args.run + '/README.md)、[冻结源码与指标](../../../temporary/models_and_metrics_' + args.tag + '_' + args.run + '.txt)。\n'
    (destination / 'README.md').write_text(readme, encoding='utf-8', newline='\n')
    files = [p for p in destination.rglob('*') if p.is_file()]
    print(f'Published {len(summary)} runs, {len(files)} files, {sum(p.stat().st_size for p in files)/1024**2:.2f} MiB to {destination}')


if __name__ == '__main__':
    main()
