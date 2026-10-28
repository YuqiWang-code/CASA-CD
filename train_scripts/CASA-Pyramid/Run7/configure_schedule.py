"""Reserve measured variant memory and leave one GiB free on GPU0."""
import json
import math
from pathlib import Path
import subprocess

from pipeline import PROTOCOL, ROOT, VARIANTS, check_selection


def main():
    decision, profile = check_selection()
    free = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    budgets = {int(row.split(',')[0]): int(row.split(',')[1]) - 1024 for row in free.splitlines()
               if int(row.split(',')[0]) == 0}
    requirements = {}
    for variant, mode in VARIANTS.items():
        peak = profile[mode]['peak_reserved_mib']
        requirements[variant] = math.ceil(peak + profile[mode]['context_overhead_mib'])
        if requirements[variant] > max(budgets.values()):
            raise RuntimeError(f'{variant} cannot fit current GPU memory budgets')
    config = dict(protocol=PROTOCOL, selected_arm=decision['selected_arm'], gpu_budget_mib=budgets,
                  variant_reserved_mib=requirements, free_headroom_mib=1024)
    (ROOT / 'scheduling.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
