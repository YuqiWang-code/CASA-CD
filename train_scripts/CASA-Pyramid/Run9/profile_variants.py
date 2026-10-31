"""Measure each exact Run9 configuration before scheduling."""
import json
from pathlib import Path
import subprocess
import sys
from config import ROOT, PROTOCOL, VARIANTS


def main():
    repo = Path(__file__).resolve().parents[3]
    results = {}
    for arm, spec in VARIANTS.items():
        destination = ROOT / f'profile_{arm}.json'
        subprocess.run([sys.executable, str(repo / 'models/profile_CASA.py'),
                        '--dataset-root', '/share_datasets/CD/SYSU-CD-256',
                        '--pretrained', str(repo / 'pretrained_weight/mobilenet_v2-b0353104.pth'),
                        '--modules', spec['module'], '--guide-weight', '.1',
                        '--retrieval-weight', '0', '--refinement-beta', str(spec['beta']),
                        '--output', str(destination)], check=True)
        result = json.loads(destination.read_text())[spec['module']]
        if result['protocol'] != PROTOCOL:
            raise RuntimeError(f'profile used a different model protocol: {arm}')
        result['module'] = spec['module']
        results[arm] = result
    (ROOT / 'profile.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
