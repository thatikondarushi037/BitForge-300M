"""Compare packed vs identical dense reconstructed weights in isolation.

No training or downloads. One model at a time, FP32 weights, optional BF16
autocast. Load peaks are separate from warm forward peaks.
"""
import argparse
import gc
import json
from pathlib import Path
import torch
from bitforge.execution import autocast_context
from bitforge.inference import load_local_checkpoint, save_inference_checkpoint, benchmark_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sequence-length',type=int,default=8)
    p.add_argument('--generation-tokens',type=int,default=2)
    p.add_argument('--training-status',required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load_local_checkpoint(args.packed)
    x=torch.arange(args.sequence_length)[None,:] % min(256,model.config.vocab_size)
    with torch.inference_mode():reference=model(x)
    dense=args.output_dir/'dense-reconstructed.pt'
    save_inference_checkpoint(dense,model,training=args.training_status)
    del model
    gc.collect()
    rows=[]
    for path in (dense,args.packed):
        for device,precision in [('cpu','fp32'),('cuda','fp32'),('cuda','bf16')]:
            gc.collect()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model,metadata=load_local_checkpoint(path,device)
            load_peak=({'allocated_bytes':torch.cuda.max_memory_allocated(),
                        'reserved_bytes':torch.cuda.max_memory_reserved()} if device=='cuda' else None)
            with torch.inference_mode(),autocast_context(device,precision):
                output=model(x.to(device)).float().cpu()
            error=(output-reference).abs().max().item()
            if precision=='fp32':torch.testing.assert_close(output,reference,rtol=1e-4,atol=1e-5)
            row={'path':str(path),'checkpoint_type':metadata['checkpoint_type'],
                 'metadata':metadata,'load_cuda_peak':load_peak,'max_absolute_error_vs_cpu_fp32':error,
                 'benchmark':benchmark_model(model,sequence_length=args.sequence_length,
                                             warmup=2,repeats=5,precision=precision,generation_tokens=args.generation_tokens)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del model
            gc.collect()
    report={'packed_bytes':args.packed.stat().st_size,'dense_reconstructed_bytes':dense.stat().st_size,
            'compression_ratio':dense.stat().st_size/args.packed.stat().st_size,'rows':rows}
    with (args.output_dir/'results.json').open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
