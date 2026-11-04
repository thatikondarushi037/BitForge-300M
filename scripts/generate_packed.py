"""Offline generation using strict packed import and dense frozen FP32 execution."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from bitforge.packed import load_packed
from bitforge.tokenizer import ByteTokenizer
from bitforge.training import greedy_generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--prompt', default='The workshop ')
    parser.add_argument('--max-new-tokens', type=int, default=120)
    args = parser.parse_args()
    if not args.prompt or not 0 <= args.max_new_tokens <= 4096:
        parser.error('need a nonempty prompt and 0..4096 new tokens')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, header = load_packed(args.checkpoint, args.device)
    tokenizer = ByteTokenizer.from_metadata(header['tokenizer'])
    tokens = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    result = greedy_generate(model, tokens, args.max_new_tokens)
    print(tokenizer.decode(result[0].tolist(), errors='replace'))
