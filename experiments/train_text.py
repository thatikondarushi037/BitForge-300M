"""Offline, bounded byte-level text training; resumes to a total step count."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as F

from bitforge import TernaryTransformer, count_parameters, tiny_config
from bitforge.data import NextTokenBatcher, load_corpus
from bitforge.tokenizer import ByteTokenizer
from bitforge.training import (check_small_config, evaluate, greedy_generate,
                               load_checkpoint, restore_training_state, save_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).resolve().parents[1] / 'data/dev_corpus.txt')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=300, help='total completed optimizer steps, also on resume')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--projection-backend', choices=('ternary', 'fp32'))
    for name in ('sequence-length', 'batch-size', 'seed', 'd-model', 'n-heads', 'n-layers', 'd-ff'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--validation-fraction', type=float)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (Path(__file__).resolve().parents[1] / '.venv/Scripts/python.exe').resolve():
        parser.error('use this project\'s .venv interpreter explicitly')
    if args.output_dir.exists():
        parser.error('output directory already exists; choose a new directory')
    if not 1 <= args.steps <= 10000:
        parser.error('steps must be within 1..10000')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; select CPU explicitly')
    defaults = {'sequence_length': 64, 'batch_size': 16, 'seed': 42,
                'learning_rate': 0.003, 'validation_fraction': 0.2,
                'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'd_ff': 128}
    settings = {k: getattr(args, k) if getattr(args, k) is not None else v for k, v in defaults.items()}
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    device = torch.device(args.device)
    payload = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        if args.projection_backend is not None and args.projection_backend != model.config.projection_backend:
            parser.error('cannot change projection backend on resume')
        saved = payload['progress']['settings']
        for name in defaults:
            if getattr(args, name) is not None and getattr(args, name) != saved[name]:
                parser.error(f'cannot change {name} on resume')
        settings = saved
    if not 1 <= settings['batch_size'] <= 64 or not 0 < settings['learning_rate'] < 1:
        parser.error('batch size must be within 1..64 and learning rate within (0,1)')
    random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    if not payload:
        config = replace(tiny_config(), vocab_size=ByteTokenizer.vocab_size,
                         projection_backend=args.projection_backend or 'ternary',
                         max_seq_len=settings['sequence_length'], d_model=settings['d_model'],
                         n_heads=settings['n_heads'], n_layers=settings['n_layers'], d_ff=settings['d_ff'])
        check_small_config(config)
        model = TernaryTransformer(config).to(device)
    train, validation, corpus = load_corpus(args.corpus, settings['validation_fraction'])
    if payload and any(corpus[k] != payload['progress']['corpus'][k]
                       for k in ('sha256', 'split_byte_offset', 'train_tokens', 'validation_tokens')):
        parser.error('resume corpus/split differs from saved training run')
    batcher = NextTokenBatcher(train, settings['sequence_length'], settings['batch_size'], settings['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=0.01)
    start_step = payload['progress']['step'] if payload else 0
    if args.steps <= start_step:
        parser.error('total steps must exceed the checkpoint step')
    if payload:
        restore_training_state(payload, optimizer, batcher, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    initial_train = evaluate(model, train, settings['sequence_length'], settings['batch_size'])
    initial_validation = evaluate(model, validation, settings['sequence_length'], settings['batch_size'])
    sampled_loss_sum, sampled_tokens = 0.0, 0
    for step in range(start_step, args.steps):
        model.train()
        x, y = batcher.sample(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        sampled_loss_sum += loss.item() * y.numel()
        sampled_tokens += y.numel()
    final_train = evaluate(model, train, settings['sequence_length'], settings['batch_size'])
    final_validation = evaluate(model, validation, settings['sequence_length'], settings['batch_size'])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None}
    checkpoint = args.output_dir / 'checkpoint.pt'
    progress = {'step': args.steps, 'settings': settings, 'corpus': corpus,
                'optimizer': 'AdamW', 'weight_decay': 0.01, 'gradient_clip_norm': 1.0}
    save_checkpoint(checkpoint, model, optimizer, batcher, progress)
    # Verify actual on-disk restoration against the final model, before generating.
    prompt = torch.tensor([ByteTokenizer().encode('The workshop ')], device=device)
    model.eval()
    with torch.no_grad():
        expected = model(prompt[:, -model.config.max_seq_len:])
    restored, restored_payload = load_checkpoint(checkpoint, device)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(prompt[:, -model.config.max_seq_len:]), rtol=0, atol=0)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=settings['learning_rate'])
    restored_batcher = NextTokenBatcher(train, settings['sequence_length'], settings['batch_size'], settings['seed'])
    restore_training_state(restored_payload, restored_optimizer, restored_batcher, device)
    assert torch.equal(batcher.generator.get_state(), restored_batcher.generator.get_state())
    generated = greedy_generate(restored, prompt, 120)
    sample = ByteTokenizer().decode(generated[0].tolist(), errors='replace')
    report = {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'parameters': count_parameters(model), 'config': restored_payload['config'],
              'tokenizer': ByteTokenizer().metadata(), 'corpus': corpus,
              'start_step': start_step, 'step': args.steps, 'settings': settings,
              'initial_train': initial_train, 'initial_validation': initial_validation,
              'final_train': final_train, 'final_validation': final_validation,
              'sampled_training_loss_mean': sampled_loss_sum / sampled_tokens,
              'elapsed_seconds': elapsed, **memory,
              'checkpoint_restored_logits_exact': True,
              'checkpoint_restored_step': restored_payload['progress']['step'],
              'checkpoint_restored_batch_rng_exact': True,
              'generated_sample': sample}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
