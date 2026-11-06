"""Repeated generation measurements, separate from forward/load timings."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import torch
from bitforge.inference import generate, load_local_checkpoint
from bitforge.resources import cpu_memory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-packed',type=Path,default=Path('artifacts/release-full-final/full.bft'))
    p.add_argument('--full-dense',type=Path,default=Path('artifacts/release-full-storage/dense-reconstructed.pt'))
    p.add_argument('--demo-packed',type=Path,default=Path('artifacts/release-demo-final/demo.bft'))
    p.add_argument('--demo-dense',type=Path,default=Path('artifacts/release-demo-storage/dense-reconstructed.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    if not 1<=args.repeats<=10:p.error('repeats must be 1..10')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for path in (args.full_dense,args.full_packed,args.demo_dense,args.demo_packed):
        for device,precision in [('cpu','fp32'),('cuda','fp32'),('cuda','bf16')]:
            gc.collect();torch.cuda.empty_cache()
            model,metadata=load_local_checkpoint(path,device)
            tokens=2 if metadata['parameters']>100_000_000 else 16
            generate(model,'The ',max_new_tokens=tokens,precision=precision)
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            observations=[generate(model,'The ',max_new_tokens=tokens,precision=precision) for _ in range(args.repeats)]
            assert all(r['token_ids']==observations[0]['token_ids'] for r in observations)
            row={'checkpoint':str(path),'checkpoint_type':metadata['checkpoint_type'],'parameters':metadata['parameters'],
                 'device':device,'precision':precision,'prompt':'The ','prompt_tokens':4,'max_new_tokens':tokens,
                 'warmup':1,'repeats':args.repeats,'seed':42,'sampling':'greedy, byte IDs only',
                 'seconds':[r['seconds'] for r in observations],
                 'tokens_per_second':[r['tokens_per_second'] for r in observations],
                 'median_tokens_per_second':statistics.median(r['tokens_per_second'] for r in observations),
                 'output_token_ids':observations[0]['token_ids'],
                 'cuda_peak':({'allocated_bytes':torch.cuda.max_memory_allocated(),
                               'reserved_bytes':torch.cuda.max_memory_reserved()} if device=='cuda' else None),
                 'cpu_memory':cpu_memory()}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('seconds','tokens_per_second','output_token_ids')}),flush=True)
            del model
    with args.output.open('x') as f:json.dump({'rows':rows},f,indent=2)


if __name__=='__main__':main()
