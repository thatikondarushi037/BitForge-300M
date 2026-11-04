"""Bounded synthetic cyclic next-token smoke test, always using the tiny config."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as F

from bitforge import TernaryTransformer, count_parameters, tiny_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone-2-next-token.json'))
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('steps must be between 1 and 1000')
    if args.output.exists():
        parser.error('output exists; choose a new path to preserve experimental results')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; explicitly select --device cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    config = tiny_config()
    device = torch.device(args.device)
    model = TernaryTransformer(config).to(device)
    # Shift the same sequence to form next-token targets. No identity labels.
    offsets = torch.arange(0, config.vocab_size, 4)[:, None]
    positions = torch.arange(17)[None, :]
    sequences = (offsets + positions) % config.vocab_size
    validation = (offsets + 2 + positions) % config.vocab_size
    x, targets = sequences[:, :-1].to(device), sequences[:, 1:].to(device)
    vx, vt = validation[:, :-1].to(device), validation[:, 1:].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)

    def evaluate(inputs, labels):
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten()).item()
            accuracy = (logits.argmax(-1) == labels).float().mean().item()
        return loss, accuracy

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    initial_loss, initial_accuracy = evaluate(x, targets)
    for _ in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
    final_loss, final_accuracy = evaluate(x, targets)
    validation_loss, validation_accuracy = evaluate(vx, vt)
    if not final_loss < initial_loss:
        raise RuntimeError('smoke test failed to reduce next-token loss')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result = {
        'task': 'cyclic token successor modulo vocabulary; shifted sequence targets',
        'seed': args.seed, 'steps': args.steps, 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'config': asdict(config), 'parameters': count_parameters(model),
        'batch_size': x.size(0), 'sequence_length': x.size(1),
        'learning_rate': 0.003, 'weight_decay': 0.01, 'gradient_clip_norm': 1.0,
        'dtype': str(next(model.parameters()).dtype),
        'initial_loss': initial_loss, 'final_loss': final_loss,
        'initial_accuracy': initial_accuracy, 'final_accuracy': final_accuracy,
        'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy,
        'elapsed_seconds': time.perf_counter() - start,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
