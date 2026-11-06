"""Exercise the installed python -m entry point against real release files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    rows=[]
    def run(arguments,expected=0,json_output=True):
        result=subprocess.run([sys.executable,'-m','bitforge',*arguments],capture_output=True,text=True,encoding='utf-8',timeout=90)
        if result.returncode!=expected:
            raise RuntimeError(f'{arguments}: {result.returncode}: {result.stderr}')
        output=json.loads(result.stdout) if json_output else result.stdout
        rows.append({'arguments':arguments,'exit_code':result.returncode,'output':output,'stderr':result.stderr})
        print('PASS: '+' '.join(arguments),flush=True)
        return output
    run(['--help'],json_output=False)
    run(['model-info','--checkpoint','artifacts/release-full-final/full.bft'])
    run(['generate','--checkpoint','artifacts/release-demo-final/demo.pt','--device','cpu','--prompt','The technician will ','--max-new-tokens','16','--json'])
    sampling=['generate','--checkpoint','artifacts/release-demo-final/demo.bft','--device','cuda','--prompt','The technician will ',
              '--temperature','0.8','--top-k','40','--top-p','0.9','--seed','42','--max-new-tokens','16','--json']
    a=run(sampling);b=run(sampling)
    assert a['token_ids']==b['token_ids']
    run(['demo','--checkpoint','artifacts/release-demo-final/demo.bft','--device','cuda','--max-new-tokens','32'],json_output=False)
    run(['demo','--checkpoint','artifacts/release-full-final/full.bft','--device','cuda','--prompt','The ','--max-new-tokens','2','--json'])
    benchmark=run(['benchmark','--checkpoint','artifacts/release-full-final/full.bft','--device','cuda','--sequence-length','8','--warmup','2',
                   '--repeats','5','--generation-tokens','2'])
    assert benchmark['parameters']==297_059_328 and benchmark['finite_logits']
    assert benchmark['forward_cuda_peak']['allocated_bytes']>0
    run(['verify-checkpoint','--checkpoint','artifacts/release-full-final/full.bft','--reference','artifacts/release-full-final/full-master.pt','--device','cuda'])
    run(['verify-checkpoint','--checkpoint','artifacts/release-demo-final/demo.bft','--reference','artifacts/release-demo-final/demo.pt','--device','cpu'])
    run(['generate','--checkpoint','artifacts/milestone-3-text/checkpoint.pt','--device','cpu','--max-new-tokens','2','--json'])
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'corrupt.bft'
        data=bytearray(Path('artifacts/release-demo-final/demo.bft').read_bytes());data[-1]^=1;path.write_bytes(data)
        run(['verify-checkpoint','--checkpoint',str(path)],expected=2,json_output=False)
        run(['model-info','--checkpoint',str(Path(directory)/'missing.pt')],expected=2,json_output=False)
    with args.output.open('x') as f:json.dump({'checks':rows,'passed':len(rows),'failed':0},f,indent=2)


if __name__=='__main__':main()
