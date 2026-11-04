"""Greedy offline byte generation from a locally produced checkpoint."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from bitforge.tokenizer import ByteTokenizer
from bitforge.training import greedy_generate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prompt', default='The workshop ')
    parser.add_argument('--max-new-tokens', type=int, default=120)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not args.prompt or not 0 <= args.max_new_tokens <= 4096:
        parser.error('supply a nonempty prompt and 0..4096 new tokens')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    model, payload = load_checkpoint(args.checkpoint, args.device)
    tokenizer = ByteTokenizer.from_metadata(payload['tokenizer'])
    tokens = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    output = greedy_generate(model, tokens, args.max_new_tokens)
    print(tokenizer.decode(output[0].tolist(), errors='replace'))


if __name__ == '__main__':
    main()
