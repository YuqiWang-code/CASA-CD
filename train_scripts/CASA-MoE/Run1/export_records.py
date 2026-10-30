"""Publish the completed MoE matrix using the project log parser."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from export_training_records import parse_log, write_csv, plt
from config import experiments


def main():
    source = ROOT / 'outputs/CASA-MoE/Run1'
    target = ROOT / 'docs/experiments/CASA-MoE/Run1'
    state = json.loads((source / 'pipeline_status.json').read_text())
    assert state['status'] == 'complete'
    summary, links, grouped = [], [], {}
    for name in experiments():
        variant, dataset = name.split('/')
        original, published = source / name, target / name
        text = (original / 'train_log.txt').read_text(encoding='utf-8')
        assert '=== TEST RESULTS ===' in text and '=== END TEST RESULTS ===' in text, name
        result = json.loads((original / 'result.json').read_text())
        manifest = json.loads((original / 'run_manifest.json').read_text())
        rows, diagnostics = parse_log(text)
        assert len(rows) == manifest['epochs'], name
        best = next(r for r in rows if r['epoch'] == result['best_epoch'])
        assert best['val_F1'] == max(r['val_F1'] for r in rows), name
        published.mkdir(parents=True, exist_ok=True)
        for filename in ('train_log.txt', 'result.json', 'run_manifest.json',
                         'evaluation_manifest.json', 'deploy_check.json', 'placement.json', 'arch.json'):
            shutil.copyfile(original / filename, published / filename)
        # The shared parser calls evaluation columns val_; this round selects on test.
        write_csv(published / 'epochs.csv', [{k.replace('val_', 'test_selection_', 1) if k.startswith('val_') else k: v
                                             for k, v in r.items()} for r in rows])
        (published / 'module_diagnostics.jsonl').write_text(
            ''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in diagnostics), encoding='utf-8')
        summary.append(dict(dataset=dataset, variant=variant, best_epoch=result['best_epoch'],
                            selection_split='test', independent_test=False,
                            **{k + '_pct': result['metrics'][k] * 100 for k in ('recall', 'precision', 'F1', 'IoU')},
                            boundary_1px_F1_pct=result['spatial_diagnostics']['boundary_1px']['F1'] * 100,
                            deploy_params_m=result['deploy_params_m'], deploy_max_diff=result['deploy_max_diff']))
        links.append(f"| {dataset.split('-')[0]} | {variant} | {result['best_epoch']} | {result['metrics']['F1']*100:.4f} | [日志]({name}/train_log.txt) · [逐轮]({name}/epochs.csv) · [诊断]({name}/module_diagnostics.jsonl) |")
        grouped.setdefault(dataset, []).append((variant, rows))
    write_csv(target / 'summary.csv', summary)
    (target / 'curves').mkdir(exist_ok=True)
    for dataset, records in grouped.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for variant, rows in records:
            for ax, field in zip(axes, ('train_main', 'val_Loss', 'val_F1')):
                ax.plot([r['epoch'] for r in rows], [r[field] * (100 if field == 'val_F1' else 1) for r in rows], label=variant)
        for ax, title in zip(axes, ('Training main loss', 'Test-selection loss', 'Test-selection F1 (%)')):
            ax.set(title=title, xlabel='Epoch (0-based)')
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6)
        fig.suptitle(dataset + ' | CASA-MoE Run1')
        fig.savefig(target / 'curves' / (dataset + '.png'), dpi=160)
        plt.close(fig)
    shutil.copyfile(source / 'pipeline_status.json', target / 'pipeline_status.json')
    text = '# CASA-MoE / Run1 完整训练记录\n\n24项正式实验，物理GPU1；2026-10-28 12:51:18至2026-10-29 20:51:42 CST。\n\n'
    text += 'epoch使用日志的0起始索引。每epoch在test选模，最终结果为同一test集上的最佳checkpoint复测，不是独立盲测。探针与停止项不计入正式指标。\n\n'
    text += '| 数据集 | 变体 | best_epoch | F1 (%) | 记录 |\n|---|---|---:|---:|---|\n' + '\n'.join(links)
    text += '\n\n[全部指标CSV](summary.csv) · [结果分析](analysis.md) · [启动与性能记录](startup.md)\n\n'
    text += '\n'.join(f'![{d}](curves/{d}.png)' for d in grouped) + '\n'
    (target / 'README.md').write_text(text, encoding='utf-8')
    print(f'Published {len(summary)} complete experiments')


if __name__ == '__main__':
    main()
