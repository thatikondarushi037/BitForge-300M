"""Export and compare master, frozen-dense, and packed storage of one trained model."""

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def write_json(path: Path, value: dict):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def worker(args):
    import torch
    from bitforge import TernaryTransformer, TransformerConfig
    from bitforge.benchmark import variation
    from bitforge.data import load_corpus
    from bitforge.packed import load_packed
    from bitforge.tokenizer import ByteTokenizer
    from bitforge.training import check_small_config, greedy_generate

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = TransformerConfig(**json.loads(args.source_metrics.read_text())['config'])
    check_small_config(config)
    path = {'master_bitlinear': args.source_weights,
            'dense_quantized': args.output_dir / 'dense-quantized-weights.pt',
            'packed': args.output_dir / 'model.bft'}[args.mode]

    def load(device):
        if args.mode == 'packed':
            return load_packed(path, device, expected_config=config)[0]
        runtime = config if args.mode == 'master_bitlinear' else replace(config, projection_backend='fp32')
        state = torch.load(path, weights_only=True, map_location='cpu')
        model = TernaryTransformer(runtime)
        model.load_state_dict(state, strict=True)
        return model.requires_grad_(False).eval().to(device)

    # Warm OS file cache and CPU loader, then measure complete reconstruction on CPU.
    model = load('cpu')
    del model
    gc.collect()
    cpu_ms, gpu_ms = [], []
    for _ in range(args.load_repeats):
        start = time.perf_counter()
        model = load('cpu')
        cpu_ms.append((time.perf_counter() - start) * 1000)
        del model
        gc.collect()
    model = load('cuda')  # Untimed context and transfer warmup.
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(args.load_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = load('cuda')
        torch.cuda.synchronize()
        gpu_ms.append((time.perf_counter() - start) * 1000)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    model = load('cuda')
    train, _, _ = load_corpus(args.corpus)
    inputs = train[:config.max_seq_len][None, :].cuda()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    latency = []
    with torch.inference_mode():
        for _ in range(20):
            model(inputs)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.latency_repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.latency_iterations):
                model(inputs)
            torch.cuda.synchronize()
            latency.append((time.perf_counter() - start) * 1000 / args.latency_iterations)
    memory = {'baseline_allocated_bytes': baseline,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    prompt = torch.tensor([ByteTokenizer().encode('The workshop ')], device='cuda')
    generated = greedy_generate(model, prompt, 120)
    row = {'mode': args.mode, 'file_bytes': path.stat().st_size,
           'cpu_load_ms': variation(cpu_ms), 'cpu_load_ms_raw': cpu_ms,
           'cuda_load_ms': variation(gpu_ms), 'cuda_load_ms_raw': gpu_ms,
           'inference_forward_ms': variation(latency), 'inference_forward_ms_raw': latency,
           'inference_memory': memory,
           'workload': {'batch_size': 1, 'context': config.max_seq_len, 'warmup': 20,
                        'latency_repeats': args.latency_repeats,
                        'forwards_per_repeat': args.latency_iterations,
                        'loading_repeats': args.load_repeats,
                        'loading_cache': 'warm OS file cache; complete CPU construction/validation, plus HtoD for CUDA'},
           'generated_sample': ByteTokenizer().decode(generated[0].tolist(), errors='replace'),
           'environment': {'python': sys.version, 'torch': torch.__version__,
                           'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}}
    write_json(args.output_dir / f'{args.mode}.json', row)
    print(f"{args.mode}: CPU load {row['cpu_load_ms']['mean']:.3f}ms, "
          f"CUDA load {row['cuda_load_ms']['mean']:.3f}ms, forward {row['inference_forward_ms']['mean']:.3f}ms", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    source = root / 'artifacts/milestone-4-paired/seed-42-ternary'
    parser.add_argument('--source-checkpoint', type=Path, default=source / 'checkpoint.pt')
    parser.add_argument('--source-weights', type=Path, default=source / 'weights.pt')
    parser.add_argument('--source-metrics', type=Path, default=source / 'metrics.json')
    parser.add_argument('--corpus', type=Path, default=root / 'data/dev_corpus.txt')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--load-repeats', type=int, default=5)
    parser.add_argument('--latency-repeats', type=int, default=5)
    parser.add_argument('--latency-iterations', type=int, default=100)
    parser.add_argument('--mode', choices=('master_bitlinear', 'dense_quantized', 'packed'), help=argparse.SUPPRESS)
    parser.add_argument('--prepare', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (root / '.venv/Scripts/python.exe').resolve():
        parser.error('use project .venv explicitly')
    if not 1 <= args.load_repeats <= 20 or not 1 <= args.latency_repeats <= 20 or not 1 <= args.latency_iterations <= 1000:
        parser.error('invalid repetition budget')
    if args.mode:
        worker(args)
        return
    if args.output_dir.exists():
        parser.error('output directory exists; preserve previous artifacts')
    if not args.prepare:
        common = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(args.output_dir),
                  '--source-checkpoint', str(args.source_checkpoint), '--source-weights', str(args.source_weights),
                  '--source-metrics', str(args.source_metrics), '--corpus', str(args.corpus),
                  '--load-repeats', str(args.load_repeats), '--latency-repeats', str(args.latency_repeats),
                  '--latency-iterations', str(args.latency_iterations)]
        # Preparation exits completely before measurement workers start; no idle parent CUDA context.
        subprocess.run(common + ['--prepare'], check=True, cwd=root)
        for mode in ('master_bitlinear', 'dense_quantized', 'packed'):
            subprocess.run(common + ['--mode', mode], check=True, cwd=root)
        info = json.loads((args.output_dir / 'export.json').read_text())
        runs = [json.loads((args.output_dir / f'{mode}.json').read_text())
                for mode in ('master_bitlinear', 'dense_quantized', 'packed')]
        write_json(args.output_dir / 'results.json', {'export': info, 'runs': runs,
                   'limitation': 'Packing reduces disk bytes only; both frozen paths execute dense FP32. Warm-cache single-session timings, not cold startup or packed kernels.'})
        return
    import torch
    from bitforge.benchmark import state_digest
    from bitforge.data import load_corpus
    from bitforge.packed import eligible_weight_names, export_packed, load_packed
    from bitforge.quantization import ternary_quantize
    from bitforge.training import evaluate, load_checkpoint

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    model, payload = load_checkpoint(args.source_checkpoint, 'cuda')
    original = torch.load(args.source_weights, map_location='cpu', weights_only=True)
    if set(original) != set(model.state_dict()):
        raise ValueError('reference state names differ')
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), original[name], rtol=0, atol=0)
    if json.loads(args.source_metrics.read_text())['config'] != payload['config']:
        raise ValueError('source metrics/config mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    info = export_packed(args.output_dir / 'model.bft', model)
    restored, _ = load_packed(args.output_dir / 'model.bft', 'cuda', expected_config=model.config)
    eligible = eligible_weight_names(model)
    for name, value in model.state_dict().items():
        expected = ternary_quantize(value) if name in eligible else value
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    model.eval()
    errors = []
    with torch.inference_mode():
        for length in (1, 8, 32, model.config.max_seq_len):
            tokens = torch.randint(0, model.config.vocab_size, (2, length), device='cuda')
            a, b = model(tokens), restored(tokens)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            errors.append({'batch': 2, 'length': length, 'max_abs_error': (a-b).abs().max().item()})
    train, validation, corpus = load_corpus(args.corpus)
    reference_validation = evaluate(model, validation, model.config.max_seq_len, 16)
    reconstructed_validation = evaluate(restored, validation, model.config.max_seq_len, 16)
    with (args.output_dir / 'dense-quantized-weights.pt').open('xb') as output:
        torch.save(restored.state_dict(), output)
    info.update({'original_weights_bytes': args.source_weights.stat().st_size,
                 'compression_ratio_original_over_packed': args.source_weights.stat().st_size / info['packed_bytes'],
                 'file_size_reduction_percent': (1-info['packed_bytes']/args.source_weights.stat().st_size)*100,
                 'source_weights_sha256': state_digest(model), 'logit_probes': errors,
                 'weights_exact_numerical_match': True, 'corpus': corpus,
                 'reference_validation': reference_validation,
                 'reconstructed_validation': reconstructed_validation})
    write_json(args.output_dir / 'export.json', info)
    print(f"Packed {info['packed_bytes']} bytes versus {info['original_weights_bytes']} original; "
          f"ratio {info['compression_ratio_original_over_packed']:.4f}x", flush=True)


if __name__ == '__main__':
    main()
