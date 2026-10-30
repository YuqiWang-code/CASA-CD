"""Measure folded FP32 batch-one inference on an idle GPU."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modules',nargs='+',choices=('none','hyperbolic','qkv','matched'),
                        default=('none','hyperbolic','qkv'))
    parser.add_argument('--refinement-beta',type=float,choices=(0.,1.),default=0.)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    sys.path.insert(0,str(args.models_dir.resolve()))
    from model.casa_net import CASANet
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2333)
    a,b=(torch.randn(1,3,256,256,device='cuda') for _ in range(2))
    results={}
    for mode in args.modules:
        torch.manual_seed(2333)
        model=CASANet(module=mode,refinement_beta=args.refinement_beta).cuda().eval()
        training_params=sum(p.numel() for p in model.parameters())
        model.switch_to_deploy()
        deployment_params=sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            for _ in range(30):model.predict(a,b)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times=[];repeat_medians=[]
            for repeat in range(args.repeats):
                current=[]
                for _ in range(100):
                    torch.cuda.synchronize();start=time.perf_counter()
                    model.predict(a,b)
                    torch.cuda.synchronize();current.append((time.perf_counter()-start)*1000)
                times.extend(current);repeat_medians.append(statistics.median(current))
            peak=torch.cuda.max_memory_allocated()/1024**2
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profile:
                model.predict(a,b);torch.cuda.synchronize()
            flops=sum(event.flops for event in profile.key_averages())
        results[mode]=dict(training_learnable_params=training_params,deployment_params=deployment_params,
            latency_ms_median=statistics.median(times),latency_ms_mean=statistics.mean(times),
            latency_ms_p95=sorted(times)[int(.95*len(times))-1],repeat_medians_ms=repeat_medians,
            repeat_median_std_ms=statistics.pstdev(repeat_medians),
            peak_allocated_mib=peak,profiled_flops=flops,
            flops_scope='Supported convolution/matmul operators; elementwise manifold operations excluded')
        print(mode,results[mode],flush=True)
        del model
        torch.cuda.empty_cache()
    report=dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,batch_size=1,
        input_shape=[1,3,256,256],precision='FP32',allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,outputs=1,
        output_scope='Final decoder prediction',warmup=30,iterations=100,
        seed=2333,context_seed=2333,repeats=args.repeats,refinement_beta=args.refinement_beta,pretrained=False,weights_scope='Architecture cost with random weights',variants=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
