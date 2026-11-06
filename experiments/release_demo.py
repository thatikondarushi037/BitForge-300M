"""Bounded offline demo training, portable checkpoints and CPU/CUDA benchmarks."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from bitforge.config import TransformerConfig
from bitforge.data import load_corpus, NextTokenBatcher
from bitforge.execution import construct_model, configure_execution, preflight
from bitforge.inference import save_inference_checkpoint, load_local_checkpoint, benchmark_model, generate
from bitforge.packed import export_packed
from bitforge.training import evaluate, save_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--from-training',type=Path,help='Finalize a previously completed local release-demo training checkpoint; no additional updates')
    args=p.parse_args()
    if not 1<=args.steps<=1000:p.error('steps must be 1..1000')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    config=TransformerConfig(vocab_size=259,max_seq_len=64,d_model=192,n_heads=6,n_layers=3,d_ff=768,projection_backend='ternary_ls')
    check=preflight(config,4)
    if not check['training_allowed']:raise RuntimeError('training preflight failed')
    train,val,corpus=load_corpus(Path('data/engineering_corpus.txt'))
    model=configure_execution(construct_model(config,'cuda'),sdpa=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    batcher=NextTokenBatcher(train,64,4,42)
    initial=evaluate(model,val,64,16)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.perf_counter()
    losses=[]
    for step in range(0 if args.from_training else args.steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        x,y=batcher.sample('cuda')
        loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();losses.append(loss.item())
        if (step+1)%100==0:print(json.dumps({'step':step+1,'loss':losses[-1]}),flush=True)
    torch.cuda.synchronize()
    training={'steps':args.steps,'seed':42,'batch_size':4,'sequence_length':64,
              'training_targets':args.steps*256,'optimizer':'AdamW','lr':.001,'weight_decay':.01,
              'gradient_clip':1.,'precision':'fp32','sdpa':True,'initial_validation':initial,
              'validation':evaluate(model,val,64,16),'training_seconds':time.perf_counter()-start,
              'finite_steps':len(losses),'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,
              'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'corpus':corpus}
    if args.from_training:
        # Narrow repair for our own pre-release metadata bug: TorchVersion is a
        # str subclass saved in progress. Never enable arbitrary pickle globals.
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            payload=torch.load(args.from_training,map_location='cpu',weights_only=True)
        if payload['config']!=asdict(config):raise ValueError('incompatible training source')
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        batcher.generator.set_state(payload['batch_rng'])
        training=json.loads(json.dumps(payload['progress']))
        del payload
    model.eval()
    save_checkpoint(args.output_dir/'training.pt',model,optimizer,batcher,training)
    save_inference_checkpoint(args.output_dir/'demo.pt',model,training=training)
    packed=export_packed(args.output_dir/'demo.bft',model)
    report={'training':training,'config':asdict(config),
            'packed':{k:v for k,v in packed.items() if k!='manifest'},
            'dense_bytes':(args.output_dir/'demo.pt').stat().st_size,'benchmarks':[]}
    report['compression_ratio']=report['dense_bytes']/packed['packed_bytes']
    x=torch.arange(64,device='cuda')[None,:]
    with torch.inference_mode():reference=model(x).cpu()
    report['generation']=generate(model,'The technician will ',max_new_tokens=64,seed=42)
    del model,optimizer,x
    gc.collect();torch.cuda.empty_cache()
    for checkpoint in ('demo.pt','demo.bft'):
        for device in ('cpu','cuda'):
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            model,metadata=load_local_checkpoint(args.output_dir/checkpoint,device)
            load_peak=({'allocated_bytes':torch.cuda.max_memory_allocated(),
                        'reserved_bytes':torch.cuda.max_memory_reserved()} if device=='cuda' else None)
            with torch.inference_mode():output=model(torch.arange(64,device=device)[None,:]).cpu()
            torch.testing.assert_close(output,reference,rtol=1e-4,atol=1e-4)
            row={'checkpoint':checkpoint,'metadata':metadata,'load_cuda_peak':load_peak,
                 'max_absolute_error_vs_cuda_reference':(output-reference).abs().max().item(),
                 'benchmark':benchmark_model(model,sequence_length=64,warmup=3,repeats=10,generation_tokens=16)}
            report['benchmarks'].append(row)
            print(json.dumps(row),flush=True)
            del model
            gc.collect();torch.cuda.empty_cache()
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
