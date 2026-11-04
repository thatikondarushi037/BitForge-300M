"""Shared audit and aggregation helpers for the small paired benchmark."""

import hashlib
import math
import statistics

import torch
from torch import nn

from .layers import BitLinear


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(bytes(tensor.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def storage_estimate(model: nn.Module) -> dict[str, int]:
    """Arithmetic only: no packed export or runtime implementation exists."""
    projections = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    projection_count = sum(p.numel() for p in projections)
    total = sum(p.numel() for p in model.parameters())
    remaining = total - projection_count
    return {
        'actual_parameter_tensor_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'projection_parameters': projection_count,
        'full_precision_nonprojection_parameters': remaining,
        'hypothetical_2bit_projection_bytes': sum(math.ceil(p.numel() * 2 / 8) for p in projections),
        'hypothetical_fp32_scale_bytes': 4 * len(projections),
        'hypothetical_packed_total_bytes': remaining * 4 + 4 * len(projections)
            + sum(math.ceil(p.numel() * 2 / 8) for p in projections),
        'ternary_projection_modules': sum(isinstance(m, BitLinear) for m in model.modules()),
    }


def variation(values: list[float]) -> dict[str, float]:
    return {'mean': statistics.mean(values),
            'sample_stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values)}


def aggregate_results(runs: list[dict]) -> dict:
    getters = {
        'train_loss': lambda r: r['final_train']['loss'],
        'train_perplexity': lambda r: r['final_train']['perplexity'],
        'validation_loss': lambda r: r['final_validation']['loss'],
        'validation_perplexity': lambda r: r['final_validation']['perplexity'],
        'training_seconds': lambda r: r['training']['seconds'],
        'inference_forward_ms': lambda r: r['inference']['mean_ms'],
        'training_peak_allocated_bytes': lambda r: r['training']['peak_allocated_bytes'],
        'training_peak_reserved_bytes': lambda r: r['training']['peak_reserved_bytes'],
        'inference_peak_allocated_bytes': lambda r: r['inference']['peak_allocated_bytes'],
        'inference_peak_reserved_bytes': lambda r: r['inference']['peak_reserved_bytes'],
        'training_checkpoint_bytes': lambda r: r['training_checkpoint_bytes'],
        'weights_checkpoint_bytes': lambda r: r['weights_checkpoint_bytes'],
    }
    seeds = sorted({r['seed'] for r in runs})
    paired = {}
    for seed in seeds:
        rows = [r for r in runs if r['seed'] == seed]
        if len(rows) != 2 or {r['backend'] for r in rows} != {'fp32', 'ternary'}:
            raise ValueError('each seed needs exactly one run of each backend')
        pair = {r['backend']: r for r in rows}
        a, b = pair['fp32'], pair['ternary']
        for key in ('initial_state_sha256', 'batch_stream_sha256', 'settings', 'corpus', 'parameters'):
            if a[key] != b[key]:
                raise ValueError(f'paired experiment mismatch: {key}')
        ca, cb = dict(a['config']), dict(b['config'])
        ca.pop('projection_backend'); cb.pop('projection_backend')
        if ca != cb:
            raise ValueError('paired architecture mismatch')
        paired[seed] = pair
    return {
        'seeds': seeds,
        'variation': 'sample standard deviation across seeds; not confidence intervals',
        'by_backend': {backend: {name: variation([get(pair[backend]) for pair in paired.values()])
                                  for name, get in getters.items()}
                       for backend in ('fp32', 'ternary')},
        'paired_ternary_minus_fp32': {
            name: variation([get(pair['ternary']) - get(pair['fp32']) for pair in paired.values()])
            for name, get in getters.items()},
    }
