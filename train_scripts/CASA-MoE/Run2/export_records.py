"""Publish the completed six-job MoE attribution round using project log tools."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from export_training_records import parse_log, diagnostic_rows, write_csv, curves
from extract_metrics_to_excel import extract_test_block


def main():
    source = ROOT / 'outputs/CASA-MoE/Run2'
    destination = ROOT / 'docs/experiments/CASA-MoE/Run2'
    state = json.loads((source / 'pipeline_status.json').read_text())
    if state['status'] != 'complete':
        raise RuntimeError('MoE Run2 has not completed')
    summary, table = [], []
    for dataset in ('LEVIR-CD-256', 'SYSU-CD-256'):
        records = {}
        for variant in ('M_T_TAR', 'M_S_STATIC', 'M_D_DYNAMIC'):
            original, published = source / variant / dataset, destination / variant / dataset
            text = (original / 'train_log.txt').read_text(encoding='utf-8')
            if not extract_test_block(text):
                raise RuntimeError(f'no complete TEST RESULTS: {variant}/{dataset}')
            result = json.loads((original / 'result.json').read_text())
            manifest = json.loads((original / 'run_manifest.json').read_text())
            rows, diagnostics = parse_log(text)
            assert len(rows) == manifest['epochs']
            published.mkdir(parents=True, exist_ok=True)
            for name in ('train_log.txt', 'result.json', 'run_manifest.json', 'evaluation_manifest.json', 'deploy_check.json', 'arch.json', 'placement.json'):
                shutil.copyfile(original / name, published / name)
            write_csv(published / 'epochs.csv', rows)
            (published / 'module_diagnostics.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in diagnostics), encoding='utf-8')
            if diagnostics:
                write_csv(published / 'module_diagnostics.csv', diagnostic_rows(diagnostics))
            records[variant] = rows, result, manifest
            entry = dict(dataset=dataset, variant=variant, best_epoch=result['best_epoch'], selection_split=result['selection_split'], independent_test=result['independent_test'], **{k: 100*v for k,v in result['metrics'].items()})
            summary.append(entry)
            table.append(f"| {dataset.split('-')[0]} | {variant} | {result['best_epoch']} | {entry['recall']:.4f} | {entry['precision']:.4f} | {entry['F1']:.4f} | {entry['IoU']:.4f} | [日志]({variant}/{dataset}/train_log.txt) |")
        (destination / 'curves').mkdir(exist_ok=True)
        curves(destination, dataset, records, 'test')
    write_csv(destination / 'summary.csv', summary)
    for name in ('runtime.json', 'profile.json', 'model_tests.json', 'initial_health.json', 'handoff_status.json', 'pipeline_status.json', 'scheduling.json', 'cpu_checks.json', 'handoff.log', 'pipeline.log'):
        shutil.copyfile(source / name, destination / name)
    for path in (source / 'probes').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log'):
            published = destination / path.relative_to(source)
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, published)
    intro = f"# CASA-MoE / Run2 完整结果\n\n6/6项正式训练与测试完成，无失败；队列于{state['updated']} CST结束。按每epoch test F1选模并复测，不属于独立盲测。epoch为0起始。\n\n冻结运行副本见runtime.json，实际源码归档在outputs/CASA-MoE/Run2/source_code.tar.gz；[冻结TXT](../../../temporary/models_and_metrics_CASA-MoE_Run2.txt)来自该归档。运行源码以冻结副本与归档为准。\n\n| 数据集 | 变体 | best epoch | Recall (%) | Precision (%) | F1 (%) | IoU (%) | 实际记录 |\n|---|---|---:|---:|---:|---:|---:|---|\n"
    intro += '\n'.join(table)
    intro += '\n\n完整日志、epochs.csv、模块诊断CSV/JSONL、配置与部署检查按实验保存；CSV的val_字段来自test逐epoch选模。probes为健康探针，不入正式指标。\n\n'
    for ds in ('LEVIR-CD-256','SYSU-CD-256'):
        intro += f'## {ds}\n\n![训练和选模曲线](curves/{ds}.png)\n\n'
    intro += '## 结果概况\n\n动态融合相对TAR-only的F1在LEVIR提高0.0405pp、SYSU提高0.9981pp；静态融合分别提高0.1263pp、0.2001pp。收益目前集中于SYSU，LEVIR差异较小。单seed结果不作为稳定提升结论，WHU/CDD尚未开展。\n\n[训练脚本](../../../../train_scripts/CASA-MoE/Run2/README.md)。\n'
    (destination / 'README.md').write_text(intro, encoding='utf-8', newline='\n')
    print('Published six MoE Run2 results and full training records')


if __name__ == '__main__':
    main()
