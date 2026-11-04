"""Fresh-process, paired FP32-vs-ternary benchmark on the unchanged CC0 corpus."""

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def worker(args) -> None:
    import torch
    from torch.nn import functional as F
    from bitforge import TernaryTransformer, count_parameters, tiny_config
    from bitforge.benchmark import state_digest, storage_estimate
    from bitforge.data import NextTokenBatcher, load_corpus
    from bitforge.tokenizer import ByteTokenizer
    from bitforge.training import check_small_config, evaluate, greedy_generate, save_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this GPU benchmark; no CPU fallback')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    config = replace(tiny_config(), vocab_size=259, max_seq_len=args.sequence_length,
                     projection_backend=args.backend)
    check_small_config(config)
    train, validation, corpus = load_corpus(args.corpus, 0.2)
    # CPU construction matches initial master Parameters bit for bit in both modes.
    model = TernaryTransformer(config)
    initial_digest = state_digest(model)
    theoretical_storage = storage_estimate(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    batcher = NextTokenBatcher(train, args.sequence_length, args.batch_size, args.seed)
    initial_train = evaluate(model, train, args.sequence_length, args.batch_size)
    initial_validation = evaluate(model, validation, args.sequence_length, args.batch_size)

    # Equal untimed forward/backward warmup; no updates and no sampler/RNG advance.
    wx = train[:args.sequence_length].repeat(args.batch_size, 1).cuda()
    wy = train[1:args.sequence_length+1].repeat(args.batch_size, 1).cuda()
    model.train()
    for _ in range(args.training_warmup):
        optimizer.zero_grad(set_to_none=True)
        warmup_loss = F.cross_entropy(model(wx).flatten(0, 1), wy.flatten())
        warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    del wx, wy
    if args.training_warmup:
        del warmup_loss
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    training_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stream_digest = hashlib.sha256()
    loss_sum = 0.0
    start = time.perf_counter()
    for _ in range(args.steps):
        # Audit the actual byte batches, not a separately regenerated plan.
        x, y = batcher.sample()
        stream_digest.update(bytes(x.flatten().tolist()))
        stream_digest.update(bytes(y.flatten().tolist()))
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item()
    torch.cuda.synchronize()
    training = {'seconds': time.perf_counter() - start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'baseline_allocated_bytes': training_baseline,
                'sampled_training_loss_mean': loss_sum / args.steps}
    final_train = evaluate(model, train, args.sequence_length, args.batch_size)
    final_validation = evaluate(model, validation, args.sequence_length, args.batch_size)
    settings = {'steps': args.steps, 'batch_size': args.batch_size,
                'sequence_length': args.sequence_length, 'learning_rate': args.learning_rate,
                'weight_decay': 0.01, 'clip_norm': 1.0, 'dtype': 'torch.float32',
                'training_warmup': args.training_warmup,
                'inference_warmup': args.inference_warmup,
                'latency_repeats': args.latency_repeats, 'latency_iterations': args.latency_iterations}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    save_checkpoint(output_dir / 'checkpoint.pt', model, optimizer, batcher,
                    {'step': args.steps, 'seed': args.seed, 'settings': settings, 'corpus': corpus})
    with (output_dir / 'weights.pt').open('xb') as output:
        torch.save(model.state_dict(), output)

    # Remove all optimizer state and gradients before measuring model-only inference.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, x, y, loss
    gc.collect()
    model.eval()
    inputs = train[:args.sequence_length][None, :].cuda()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    inference_baseline = torch.cuda.memory_allocated()
    milliseconds = []
    with torch.inference_mode():
        for _ in range(args.inference_warmup):
            model(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.latency_repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.latency_iterations):
                model(inputs)
            torch.cuda.synchronize()
            milliseconds.append((time.perf_counter() - start) * 1000 / args.latency_iterations)
    inference = {'batch_size': 1, 'sequence_length': args.sequence_length,
                 'workload': 'full-context forward to all-position vocabulary logits; no KV cache',
                 'repeat_mean_ms': milliseconds, 'mean_ms': statistics.mean(milliseconds),
                 'repeat_sample_stdev_ms': statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                 'baseline_allocated_bytes': inference_baseline}
    prompt = torch.tensor([ByteTokenizer().encode('The workshop ')], device='cuda')
    sample = greedy_generate(model, prompt, 120)
    row = {'seed': args.seed, 'backend': args.backend, 'config': asdict(config),
           'parameters': count_parameters(model), 'settings': settings, 'corpus': corpus,
           'tokenizer': ByteTokenizer().metadata(), 'initial_state_sha256': initial_digest,
           'batch_stream_sha256': stream_digest.hexdigest(),
           'initial_train': initial_train, 'initial_validation': initial_validation,
           'final_train': final_train, 'final_validation': final_validation,
           'training': training, 'inference': inference,
           'training_checkpoint_bytes': (output_dir / 'checkpoint.pt').stat().st_size,
           'weights_checkpoint_bytes': (output_dir / 'weights.pt').stat().st_size,
           'storage_arithmetic_only': theoretical_storage,
           'generated_sample': ByteTokenizer().decode(sample[0].tolist(), errors='replace'),
           'environment': {'python': sys.version, 'python_executable': sys.executable,
                           'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(),
                           'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                           'platform': platform.platform(),
                           'deterministic_algorithms': True, 'tf32': False}}
    write_json(output_dir / 'metrics.json', row)
    print(f"seed={args.seed} {args.backend}: val_ppl={final_validation['perplexity']:.6f} "
          f"train={training['seconds']:.3f}s forward={inference['mean_ms']:.3f}ms", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=root / 'data/dev_corpus.txt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--training-warmup', type=int, default=5)
    parser.add_argument('--inference-warmup', type=int, default=20)
    parser.add_argument('--latency-repeats', type=int, default=5)
    parser.add_argument('--latency-iterations', type=int, default=100)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=('fp32', 'ternary'), help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (root / '.venv/Scripts/python.exe').resolve():
        parser.error('use the project .venv interpreter explicitly')
    if args.output_dir.exists():
        parser.error('output directory exists; previous results must be preserved')
    if not 1 <= args.steps <= 1000 or not 1 <= args.batch_size <= 32 or not 1 <= args.sequence_length <= 128:
        parser.error('steps 1..1000, batch size 1..32, context 1..128 required')
    if not 0 < args.learning_rate < 1 or not 0 <= args.training_warmup <= 100:
        parser.error('invalid learning rate or warmup')
    if not 0 <= args.inference_warmup <= 100 or not 1 <= args.latency_repeats <= 20 or not 1 <= args.latency_iterations <= 1000:
        parser.error('invalid latency budget')
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 1:
        parser.error('seeds must be unique')
    if args.worker:
        if args.seed is None or args.backend is None:
            parser.error('worker requires seed and backend')
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs, order = [], []
    for index, seed in enumerate(args.seeds):
        backends = ('fp32', 'ternary') if index % 2 == 0 else ('ternary', 'fp32')
        for backend in backends:
            destination = args.output_dir / f'seed-{seed}-{backend}'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--seed', str(seed),
                       '--backend', backend, '--output-dir', str(destination), '--corpus', str(args.corpus)]
            for name in ('steps', 'batch_size', 'sequence_length', 'learning_rate', 'training_warmup',
                         'inference_warmup', 'latency_repeats', 'latency_iterations'):
                command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
            print(f'Running paired seed {seed}, backend {backend} in a fresh process...', flush=True)
            subprocess.run(command, check=True, cwd=root)
            runs.append(json.loads((destination / 'metrics.json').read_text(encoding='utf-8')))
            order.append({'seed': seed, 'backend': backend})
    from bitforge.benchmark import aggregate_results
    summary = {'format_version': 1, 'execution_order': order, 'runs': runs,
               'aggregate': aggregate_results(runs),
               'timing_scope': 'training optimization loop only, including sampling, batch hashing, HtoD, loss checks, clipping, AdamW; inference forward excludes HtoD',
               'limitation': 'Tiny assistant-generated CC0 corpus. No packed execution. Non-generalizable.'}
    write_json(args.output_dir / 'results.json', summary)
    print(json.dumps(summary['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
