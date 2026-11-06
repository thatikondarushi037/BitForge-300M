"""Bounded Phase 1 CUDA experiment. No downloads, full pretraining or auto scaling."""
import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from bitforge.config import TransformerConfig, target_300m_config, estimate_parameters
from bitforge.data import load_corpus, NextTokenBatcher
from bitforge.execution import construct_model, configure_execution, preflight, autocast_context
from bitforge.packed import export_packed, load_packed
from bitforge.training import evaluate
from bitforge.transformer import count_parameters


def sync():
    torch.cuda.synchronize()


def timed(call, repeats=1):
    sync(); start=time.perf_counter()
    for _ in range(repeats): result=call()
    sync()
    return result, (time.perf_counter()-start)/repeats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    args=parser.parse_args()
    if not 1 <= args.steps <= 300: parser.error('steps must be 1..300')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    train, val, corpus=load_corpus(Path('data/engineering_corpus.txt'))
    config=TransformerConfig(vocab_size=259,max_seq_len=64,d_model=192,n_heads=6,n_layers=3,d_ff=768)
    target=target_300m_config()
    meta=construct_model(target,'meta')
    assert count_parameters(meta)==estimate_parameters(target)['total']
    report={'torch':torch.__version__, 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'corpus':corpus,'steps':args.steps,'config':asdict(config),
            'target_architecture':{'config':asdict(target),'actual_meta_parameters':count_parameters(meta),
                                   'preflight_context_2048':preflight(target),
                                   'preflight_context_64':preflight(target,sequence_length=64)},
            'small_preflight':preflight(config,4), 'qat':[], 'execution':[]}
    if not report['small_preflight']['training_allowed']: raise RuntimeError('small preflight failed')
    # Full 300M execution is deliberately excluded by this bounded runner.
    for seed in (42,43,44):
        for backend in ('fp32','ternary','ternary_ls'):
            gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(seed)
            model=construct_model(replace(config,projection_backend=backend),'cuda')
            opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
            batcher=NextTokenBatcher(train,64,4,seed)
            torch.cuda.reset_peak_memory_stats(); sync(); start=time.perf_counter()
            initial=evaluate(model,val,64,16)
            losses=[]; grads=[]
            for step in range(args.steps):
                model.train(); opt.zero_grad(set_to_none=True)
                x,y=batcher.sample('cuda'); loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step(); losses.append(loss.item()); grads.append(grad.item())
            final=evaluate(model,val,64,16); sync()
            row={'seed':seed,'backend':backend,'parameters':count_parameters(model),'initial_validation':initial,
                 'validation':final,'seconds':time.perf_counter()-start,'loss_first':losses[0], 'loss_last':losses[-1],
                 'loss_min':min(losses),'loss_max':max(losses),'max_gradient_norm_before_clip':max(grads),
                 'finite_steps':len(losses),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            report['qat'].append(row); print(json.dumps(row),flush=True)
            if backend=='ternary_ls' and seed==42:
                model.eval(); x=torch.arange(64,device='cuda')[None,:]%259
                with torch.no_grad(): expected=model(x)
                info,export_seconds=timed(lambda:export_packed(args.output_dir/'improved.bft',model))
                dense=args.output_dir/'dense.pt'
                torch.save(model.state_dict(),dense)
                # Compare packed decoding against a frozen dense reconstructed model.
                (restored,header),packed_load=timed(lambda:load_packed(args.output_dir/'improved.bft','cuda'))
                with torch.no_grad():
                    actual=restored(x); torch.testing.assert_close(expected,actual,rtol=1e-5,atol=2e-6)
                    max_logit_error=(expected-actual).abs().max().item()
                dense_quantized=args.output_dir/'dense_quantized.pt'; torch.save(restored.state_dict(),dense_quantized)
                def dense_load():
                    state=torch.load(dense_quantized,map_location='cpu',weights_only=True)
                    result=construct_model(replace(config,projection_backend='fp32'),'cpu')
                    result.load_state_dict(state,assign=True); return result.eval().requires_grad_(False).to('cuda')
                loaded,dense_seconds=timed(dense_load)
                with torch.no_grad():
                    restored(x); loaded(x)
                    _,packed_latency=timed(lambda:restored(x),20)
                    _,dense_latency=timed(lambda:loaded(x),20)
                report['packed']={'packed_bytes':info['packed_bytes'],'dense_master_bytes':dense.stat().st_size,
                    'dense_quantized_bytes':dense_quantized.stat().st_size,'compression_ratio':dense.stat().st_size/info['packed_bytes'],
                    'max_logit_absolute_error':max_logit_error,'logit_tolerance':{'rtol':1e-5,'atol':2e-6},'export_seconds':export_seconds,
                    'packed_load_seconds':packed_load,'dense_load_seconds':dense_seconds,
                    'packed_restored_latency_seconds':packed_latency,'dense_latency_seconds':dense_latency,
                    'execution':'Both paths use dense FP32 GEMM after loading; no native ternary kernel.'}
                del restored,loaded
            del model,opt
    # Matched initialization and batches for execution comparisons.
    report['execution_preflight'] = preflight(config, 8)
    if not report['execution_preflight']['training_allowed']:
        raise RuntimeError('execution benchmark preflight failed')
    for precision,ckpt,sdpa in [('fp32',False,False),('fp32',False,True),('bf16',False,True),('bf16',True,True)]:
        gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(77)
        model=configure_execution(construct_model(replace(config,projection_backend='ternary_ls'),'cuda'),checkpointing=ckpt,sdpa=sdpa)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        x,y=NextTokenBatcher(train,64,8,77).sample('cuda')
        def step():
            opt.zero_grad(set_to_none=True)
            with autocast_context('cuda',precision): loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            return loss.item()
        step(); sync();torch.cuda.reset_peak_memory_stats()
        loss,latency=timed(step,10)
        row={'precision':precision,'checkpointing':ckpt,'sdpa':sdpa,'batch':8,'sequence_length':64,
             'steps':11,'last_loss':loss,'seconds_per_step':latency,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        report['execution'].append(row);print(json.dumps(row),flush=True)
        del model,opt
    for backend in ('fp32','ternary','ternary_ls'):
        rows=[r['validation']['loss'] for r in report['qat'] if r['backend']==backend]
        report.setdefault('summary',{})[backend]={'mean_validation_ce':statistics.mean(rows),'sample_sd':statistics.stdev(rows)}
    report['paired_ls_minus_original_ce']=[next(r['validation']['loss'] for r in report['qat'] if r['seed']==seed and r['backend']=='ternary_ls')-next(r['validation']['loss'] for r in report['qat'] if r['seed']==seed and r['backend']=='ternary') for seed in (42,43,44)]
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
