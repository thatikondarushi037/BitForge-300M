"""Isolated CUDA load/inference measurements of locally produced checkpoints."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from bitforge.config import TransformerConfig
from bitforge.execution import construct_model
from bitforge.packed import load_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = TransformerConfig(**json.loads((args.directory/'results.json').read_text())['config'])
    from dataclasses import replace
    config = replace(config, projection_backend='fp32')
    rows = []
    for packed in (False, True):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if packed:
            model, _ = load_packed(args.directory/'improved.bft', 'cuda')
        else:
            state = torch.load(args.directory/'dense_quantized.pt', weights_only=True, map_location='cpu')
            model = construct_model(config)
            model.load_state_dict(state, assign=True)
            model = model.eval().requires_grad_(False).to('cuda')
            del state
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-start
        load_peak = torch.cuda.max_memory_allocated()
        x = torch.arange(64, device='cuda')[None, :] % config.vocab_size
        with torch.inference_mode():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(100):
                model(x)
            torch.cuda.synchronize()
        rows.append(dict(packed=packed, load_seconds=load_seconds,
                         load_peak_allocated_bytes=load_peak,
                         inference_seconds=(time.perf_counter()-start)/100,
                         inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         inference_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del model, x
    output = args.directory/'loading.json'
    with output.open('x') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
