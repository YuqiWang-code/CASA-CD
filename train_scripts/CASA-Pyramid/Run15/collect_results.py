"""Download Run15 logs, verify the protocol, and refresh the shared metric tables."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import config as C

REPO = Path('/home/yqwang/projects/CASA-CD')
TOLERANCE = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='Run15')
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    if not args.skip_download:
        subprocess.run([sys.executable, str(REPO / 'tools/download_training_results.py'),
                        '--tag', 'CASA-Pyramid', '--run', args.run], check=True)
    checked = {}
    for version, arm, dataset in C.experiments():
        folder = C.run_dir(version, arm, dataset)
        manifest = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
        assert manifest['protocol'] == C.PROTOCOL, (folder, manifest['protocol'])
        split = manifest.get('selection_split') or (
            'val' if 'validation' in str(manifest.get('selection', '')).lower() else 'test')
        assert split == 'val', (folder, split)
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        assert result.get('independent_test', True) is True, folder
        assert abs(result['total_params_m'] - C.TRAINABLE_PARAMS[f'{version}/{arm}'] / 1e6) < 1e-6, folder
        deploy = abs(result['deploy_params_m'] - C.DEPLOY_PARAMS[f'{version}/{arm}'] / 1e6)
        assert deploy < 1e-6, (folder, result['deploy_params_m'])
        checked[f'{version}/{arm}/{dataset}'] = dict(f1=result['metrics']['F1'],
                                                    best_epoch=result['best_epoch'],
                                                    threshold=result.get('test_threshold'))
        print(f'[CHECK] {version}/{arm}/{dataset} F1={result["metrics"]["F1"]:.6f}', flush=True)
    (C.ROOT / 'collected.json').write_text(json.dumps(checked, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, str(REPO / 'tools/extract_metrics_to_excel.py')], check=True)
    subprocess.run([sys.executable, str(REPO / 'tools/models_to_txt.py'),
                    '--tag', 'CASA-Pyramid', '--run', args.run], check=False)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'compare.py')], check=True)
    subprocess.run([sys.executable, str(REPO / 'tools/export_training_records.py'),
                    '--tag', 'CASA-Pyramid', '--run', args.run], check=False)


if __name__ == '__main__':
    main()
