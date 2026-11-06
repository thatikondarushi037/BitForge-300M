"""Measure CPU/CUDA quantizer reduction/code differences, one tensor at a time."""
import argparse
import json
from pathlib import Path
import torch
from bitforge.inference import load_local_checkpoint
from bitforge.layers import BitLinear, LeastSquaresBitLinear
from bitforge.quantization import ternary_codes, least_squares_codes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    torch.set_num_threads(4)
    model,metadata=load_local_checkpoint(args.checkpoint)
    rows=[]
    with torch.inference_mode():
        for name,module in model.named_modules():
            if not isinstance(module,BitLinear):continue
            quantize=least_squares_codes if isinstance(module,LeastSquaresBitLinear) else ternary_codes
            cpu_codes,cpu_scale=quantize(module.weight)
            cuda_codes,cuda_scale=quantize(module.weight.cuda())
            rows.append({'tensor':name+'.weight','count':module.weight.numel(),
                         'different_codes':int((cpu_codes!=cuda_codes.cpu()).sum()),
                         'cpu_scale':cpu_scale.item(),'cuda_scale':cuda_scale.item(),
                         'absolute_scale_error':abs(cpu_scale.item()-cuda_scale.item())})
    report={'checkpoint':str(args.checkpoint),'parameters':metadata['parameters'],'rows':rows,
            'total_different_codes':sum(r['different_codes'] for r in rows),
            'max_absolute_scale_error':max(r['absolute_scale_error'] for r in rows)}
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
