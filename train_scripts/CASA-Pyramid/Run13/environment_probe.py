"""Record the interpreter, CUDA and GPU facts of the Run13 runtime."""
import json
import platform
import subprocess
import sys

import torch

from config import GPUS, PROTOCOL, ROOT

gpus = []
for index in GPUS:
    name, driver = subprocess.check_output(
        ['nvidia-smi', '-i', str(index), '--query-gpu=name,driver_version',
         '--format=csv,noheader'], text=True).strip().split(', ')
    gpus.append(dict(index=index, name=name, driver=driver))

with open(ROOT/'environment.json', 'w') as stream:
    json.dump(dict(protocol=PROTOCOL, python=sys.version, platform=platform.platform(),
                   pytorch=torch.__version__, cuda=torch.version.cuda,
                   physical_gpus=gpus), stream, indent=2)
print('[ENVIRONMENT]', torch.__version__, torch.version.cuda, [gpu['name'] for gpu in gpus])
