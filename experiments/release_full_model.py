"""Bounded real-297M release acceptance experiment; no training/downloads.

Writes new artifacts exclusively. A single full model is resident at a time.
CPU and CUDA use FP32 master weights; CUDA also tests BF16 autocast.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from bitforge.config import target_300m_config
from bitforge.execution import configure_execution, construct_model
from bitforge.inference import (benchmark_model, inference_preflight, load_local_checkpoint,
                               save_inference_checkpoint, synchronize)
from bitforge.packed import export_packed
from bitforge.resources import cpu_memory, disk_preflight
from bitforge.transformer import count_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('threads must be 1..16')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(300)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = target_300m_config()
    report = {'seed': 300, 'training': 'random initialization; zero training updates',
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'threads': args.threads, 'config': asdict(config), 'stages': [], 'errors': []}
    def persist(stage, **data):
        report['stages'].append({'stage': stage, **data, 'cpu_memory': cpu_memory()})
        (args.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report['stages'][-1]), flush=True)
    try:
        report['disk_preflight'] = disk_preflight(args.output_dir, 2 << 30)
        report['cpu_preflight'] = inference_preflight(config, 'cpu', 8)
        if torch.cuda.is_available():
            report['gpu'] = torch.cuda.get_device_name()
            report['cuda_preflight'] = inference_preflight(config, 'cuda', 8)
        start = time.perf_counter()
        model = configure_execution(construct_model(config), sdpa=True).eval().requires_grad_(False)
        assert count_parameters(model) == 297_059_328
        assert all(not p.is_meta for p in model.parameters())
        persist('real_construction', parameters=count_parameters(model), seconds=time.perf_counter()-start)
        x = torch.arange(8)[None, :]
        with torch.inference_mode():
            reference_cpu = model(x).cpu()
        persist('cpu_forward', **benchmark_model(model, repeats=3, generation_tokens=2))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model = model.to('cuda')
            for precision in ('fp32', 'bf16'):
                try:
                    persist('cuda_forward', **benchmark_model(model, precision=precision, repeats=5, warmup=2, generation_tokens=2))
                except (RuntimeError, ValueError) as error:
                    report['errors'].append({'stage': 'cuda_'+precision, 'error': str(error)})
            model = model.cpu()
            torch.cuda.empty_cache()
        start = time.perf_counter()
        save_inference_checkpoint(args.output_dir/'full-master.pt', model, training=report['training'])
        persist('dense_export', bytes=(args.output_dir/'full-master.pt').stat().st_size, seconds=time.perf_counter()-start)
        start = time.perf_counter()
        packed = export_packed(args.output_dir/'full.bft', model)
        persist('packed_export', bytes=packed['packed_bytes'], seconds=time.perf_counter()-start,
                sha256=packed['sha256'], eligible_parameters=packed['eligible_parameters'],
                compression_ratio=(args.output_dir/'full-master.pt').stat().st_size/packed['packed_bytes'])
        del model
        gc.collect()
        # Each load is measured independently, and each model is released first.
        for name in ('full.bft', 'full-master.pt'):
            for device in (('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',)):
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                model, metadata = load_local_checkpoint(args.output_dir/name, device)
                load_peak = ({'allocated_bytes': torch.cuda.max_memory_allocated(),
                              'reserved_bytes': torch.cuda.max_memory_reserved()} if device == 'cuda' else None)
                with torch.inference_mode():
                    output = model(x.to(device)).cpu()
                error = (output-reference_cpu).abs().max().item()
                tolerance = {'rtol': 1e-3, 'atol': 1e-3} if name == 'full-master.pt' and device == 'cuda' else {'rtol': 1e-4, 'atol': 1e-5}
                torch.testing.assert_close(output, reference_cpu, **tolerance)
                persist('checkpoint_reload', checkpoint=name, metadata=metadata,
                        max_absolute_error_vs_cpu_quantized_reference=error, load_cuda_peak=load_peak,
                        comparison_tolerance=tolerance,
                        benchmark=benchmark_model(model, repeats=3, generation_tokens=2))
                del model
                gc.collect()
        report['completed'] = True
        persist('complete')
    except Exception as error:
        report['errors'].append({'stage': 'fatal', 'type': type(error).__name__, 'error': str(error)})
        report['completed'] = False
        persist('failed')
        raise


if __name__ == '__main__':
    main()
