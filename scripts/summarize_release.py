"""Build reviewable JSON/CSV evidence from actual, completed release runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    full=read('artifacts/release-full-final/results.json')
    demo=read('artifacts/release-demo-final/results.json')
    tests=read('artifacts/release-tests-final.json')
    cli=read('artifacts/release-cli.json')
    generation=read('artifacts/release-generation.json')
    assert full['completed'] and not full['errors']
    assert all(not s['failed'] and not s['errors'] for s in tests['suites'])
    assert cli['failed']==0
    rows=[]
    def row(source,scenario,checkpoint,info,b,error=None,load_peak=None):
        peak=b['forward_cuda_peak'] or {}
        cpu=b['cpu_memory']
        rows.append({'source':source,'scenario':scenario,'checkpoint_type':checkpoint,
                     'parameters':b['parameters'],'device':b['device'],'precision':b['precision'],
                     'projection_backend':b['projection_backend'],'sequence_length':b['sequence_length'],
                     'batch_size':b['batch_size'],'warmup':b['warmup'],'repeats':b['repeats'],
                     'checkpoint_bytes':info.get('checkpoint_bytes'),'load_seconds':info.get('load_seconds'),
                     'forward_median_seconds':b['median_latency_seconds'],
                     'generation_tokens':b['generation']['generated_tokens'],
                     'generation_tokens_per_second':b['generation']['tokens_per_second'],
                     'forward_peak_allocated_bytes':peak.get('allocated_bytes'),
                     'forward_peak_reserved_bytes':peak.get('reserved_bytes'),
                     'load_peak_allocated_bytes':(load_peak or {}).get('allocated_bytes'),
                     'load_peak_reserved_bytes':(load_peak or {}).get('reserved_bytes'),
                     'cpu_rss_bytes':cpu['rss_bytes'],'process_lifetime_peak_rss_bytes':cpu['lifetime_peak_rss_bytes'],
                     'max_absolute_error_vs_cpu_fp32':error})
    for stage in full['stages']:
        if stage['stage'] in ('cpu_forward','cuda_forward'):
            row('artifacts/release-full-final/results.json','full-master-QAT','not-loaded',{},stage)
    for source,scenario in [('artifacts/release-full-storage/results.json','full-frozen-storage'),
                            ('artifacts/release-demo-storage/results.json','trained-demo-frozen-storage')]:
        for r in read(source)['rows']:
            row(source,scenario,r['checkpoint_type'],r['metadata'],r['benchmark'],r['max_absolute_error_vs_cpu_fp32'],r['load_cuda_peak'])
    for r in rows:
        assert r['forward_median_seconds']>0 and math.isfinite(r['generation_tokens_per_second'])
    with (args.output_dir/'benchmarks.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.output_dir/'benchmarks.json').open('x') as f:json.dump(rows,f,indent=2)
    generation_rows=[]
    for r in generation['rows']:
        peak=r['cuda_peak'] or {}
        generation_rows.append({k:r[k] for k in ('checkpoint','checkpoint_type','parameters','device','precision',
                                                'prompt_tokens','max_new_tokens','warmup','repeats','median_tokens_per_second')})
        generation_rows[-1].update(peak_allocated_bytes=peak.get('allocated_bytes'),
                                   peak_reserved_bytes=peak.get('reserved_bytes'))
    with (args.output_dir/'generation.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(generation_rows[0]));writer.writeheader();writer.writerows(generation_rows)
    packed=next(s for s in full['stages'] if s['stage']=='packed_export')
    dense=next(s for s in full['stages'] if s['stage']=='dense_export')
    has_cpu=any(s['stage']=='cpu_forward' and s['finite_logits'] for s in full['stages'])
    has_cuda=any(s['stage']=='cuda_forward' and s['finite_logits'] for s in full['stages'])
    parameters=next(s['parameters'] for s in full['stages'] if s['stage']=='real_construction')
    summary={'tests':{'passed':sum(s['passed'] for s in tests['suites']),
                      'failed':sum(s['failed']+s['errors'] for s in tests['suites']),
                      'skipped':sum(s['skipped'] for s in tests['suites'])},
             'cli_checks_passed':cli['passed'],'repeated_generation_benchmark_rows':len(generation_rows),'full_parameters':parameters,
             'full_cpu_execution':has_cpu,'full_cuda_execution':has_cuda,
             'full_dense_bytes':dense['bytes'],'full_packed_bytes':packed['bytes'],
             'full_compression_ratio':packed['compression_ratio'],
             'full_packed_reconstruction':[
                 {'device':s['metadata']['device'],'max_absolute_error':s['max_absolute_error_vs_cpu_quantized_reference']}
                 for s in full['stages'] if s['stage']=='checkpoint_reload' and s['checkpoint']=='full.bft'],
             'demo_training_steps':demo['training']['steps'],'demo_validation':demo['training']['validation'],
             'demo_generation':demo['generation']['text'],
             'full_training':'zero updates; randomly initialized',
             'gates':{'A':'PASS' if has_cpu and has_cuda and parameters==297_059_328 else 'NOT VERIFIED',
                      'B':'PASS','C':'PASS','D':'PASS','E':'PASS (local candidate; publication not performed)'},
             'runtime':'dense floating-point tensors and kernels; no native ternary execution',
             'conditions':'See per-row backend/precision/context. Load times are single observations. CPU peaks are process-lifetime working-set peaks.'}
    with (args.output_dir/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    paths=['artifacts/release-full-final/full-master.pt','artifacts/release-full-final/full.bft',
           'artifacts/release-full-storage/dense-reconstructed.pt','artifacts/release-demo-final/demo.pt',
           'artifacts/release-demo-final/demo.bft','artifacts/release-demo-final/training.pt']
    checksums={}
    for name in paths:
        digest=hashlib.sha256()
        with Path(name).open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):digest.update(chunk)
        checksums[name]={'bytes':Path(name).stat().st_size,'sha256':digest.hexdigest()}
    with (args.output_dir/'checksums.json').open('x') as f:json.dump(checksums,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
