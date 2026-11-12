"""Batch-1 deployment latency for the control and both ablation arms."""
import json
import statistics
import subprocess
import sys

import torch

from config import BENCH, GPUS, PROTOCOL, ROOT, RUNTIME

sys.path.insert(0, str(RUNTIME/'models'))
from model.casa_net import CASANet


def main():
    gpu = GPUS[0]
    name, uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=name,uuid',
                                          '--format=csv,noheader'], text=True).strip().split(', ')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True).splitlines()
    report = dict(protocol=PROTOCOL, gpu=gpu, device=name, measured=uuid not in apps,
                  batch_size=1, image_size=256, dtype='float32', cudnn_tf32=True,
                  matmul_tf32=False, warmup=30, repeats=100)
    if report['measured']:
        torch.set_num_threads(4)
        torch.manual_seed(2333)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        image = torch.randn(1, 6, 256, 256, device='cuda')
        rows = {}
        with torch.inference_mode():
            for variant, (module, heads, detail, tar, decoder) in BENCH.items():
                model = CASANet(module=module, context_heads=heads, detail_head=detail,
                                tar_variant=tar, decoder_variant=decoder).cuda().eval().switch_to_deploy()
                for _ in range(30):
                    model.predict(image[:, :3], image[:, 3:])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(100)]
                for start, end in events:
                    start.record()
                    model.predict(image[:, :3], image[:, 3:])
                    end.record()
                torch.cuda.synchronize()
                rows[variant] = dict(median_ms=statistics.median(a.elapsed_time(b) for a, b in events),
                                     deploy_params=sum(p.numel() for p in model.parameters()),
                                     peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                del model
        report['results'] = rows
    else:
        report['reason'] = f'GPU{gpu} shared at benchmark start'
    (ROOT/'inference_benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
