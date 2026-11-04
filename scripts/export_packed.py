"""Export a trusted local ternary training checkpoint to a no-pickle .bft file."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from bitforge.packed import export_packed
from bitforge.training import load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='quantization reduction device; use cuda to match CUDA source inference')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    info = export_packed(args.output, model.eval())
    info.pop('manifest')
    print(json.dumps(info, indent=2))
