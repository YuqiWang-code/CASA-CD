"""Use measured peaks with one GiB device headroom on physical GPU0."""
import json
import math
import subprocess

from config import PROTOCOL, ROOT, VARIANTS


def main():
    profiles = json.loads((ROOT / 'profile.json').read_text())
    health = [json.loads(path.read_text()) for path in (ROOT / 'probes').glob('*/health_result.json')]
    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip())
    reservations = {}
    for variant, mode in VARIANTS.items():
        profile = profiles[mode]
        if profile['protocol'] != PROTOCOL or profile['batch_size'] != 32:
            raise RuntimeError(f'incompatible memory profile: {mode}')
        peak = max([profile['peak_reserved_mib']] +
                   [item['peak_reserved_mib'] for item in health if item['module'] == mode])
        reservations[variant] = math.ceil(peak + max(0, profile['context_overhead_mib']))
        if reservations[variant] > free - 1024:
            raise RuntimeError(f'{variant} cannot fit GPU0 current free memory with one GiB headroom')
    result = dict(protocol=PROTOCOL, gpu_budget_mib={0: free-1024},
                  variant_reserved_mib=reservations, free_headroom_mib=1024, max_active=1)
    (ROOT / 'scheduling.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
