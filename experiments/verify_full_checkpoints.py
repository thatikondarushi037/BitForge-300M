"""Verify existing full-size exports sequentially without overwriting evidence."""
import argparse
import gc
import json
from pathlib import Path
import torch
from bitforge.inference import load_local_checkpoint, benchmark_model
from bitforge.resources import cpu_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error('output already exists')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rows=[]
    reference=None
    for name in ('full-master.pt','full.bft'):
        for device in ('cpu','cuda'):
            gc.collect()
            if device=='cuda':
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model,metadata=load_local_checkpoint(args.directory/name,device)
            load_peak=({'allocated_bytes':torch.cuda.max_memory_allocated(),
                        'reserved_bytes':torch.cuda.max_memory_reserved()} if device=='cuda' else None)
            x=torch.arange(8,device=device)[None,:]
            with torch.inference_mode():
                output=model(x).cpu()
            if reference is None:
                reference=output
            # Master QAT recomputes reductions on each device. Near-threshold
            # ternary codes can differ between CPU and CUDA. Packed decoding
            # fixes the source CPU codes, permitting a much tighter comparison.
            tolerance={'rtol':1e-3,'atol':1e-3} if name=='full-master.pt' and device=='cuda' else {'rtol':1e-4,'atol':1e-5}
            torch.testing.assert_close(output,reference,**tolerance)
            row={'checkpoint':name,'device':device,'metadata':metadata,
                 'max_absolute_error_vs_cpu_quantized_reference':(output-reference).abs().max().item(),
                 'comparison_tolerance':tolerance,
                 'load_cuda_peak':load_peak,'benchmark':benchmark_model(model,repeats=5,warmup=2,generation_tokens=2),
                 'cpu_memory':cpu_memory()}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del model,x
            gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
    with args.output.open('x') as f:
        json.dump({'completed':True,'rows':rows},f,indent=2)


if __name__=='__main__':main()
