"""Allocation-free preflight and explicit execution controls."""
from contextlib import nullcontext
import torch
from .config import estimate_parameters
from .transformer import TernaryTransformer, SinusoidalPositions


def memory_estimate(config, batch_size=1, sequence_length=None):
    """Conservative planning estimate, not a measured peak or fit guarantee.

    FP32 AdamW masters/gradients/moments: 16 bytes/parameter. QAT temporary
    workspace: 24 bytes/weight in largest projection; eager activations include
    scores, probabilities and FFN intermediates. Add 1 GiB runtime reserve.
    """
    t = sequence_length or config.max_seq_len
    if batch_size < 1 or not 1 <= t <= config.max_seq_len:
        raise ValueError('invalid batch or context')
    p = estimate_parameters(config)['total']
    workspace = 24 * max(3*config.d_model**2, config.d_model*config.d_ff)
    activations = 4*batch_size*config.n_layers*(12*t*config.d_model + 4*t*config.d_ff + 2*config.n_heads*t*t)
    logits = 8*batch_size*t*config.vocab_size
    return {'parameters': p, 'fp32_weights_bytes': 4*p,
            'training_budget_bytes': 16*p+workspace+activations+logits+(1<<30),
            'inference_budget_bytes': 4*p+workspace+activations+logits+(1<<30),
            'kind': 'conservative estimate; allocator and kernel dependent'}


def preflight(config, batch_size=1, sequence_length=None, fraction=0.70):
    if not 0 < fraction <= 0.8:
        raise ValueError('fraction must be in (0, 0.8]')
    estimate = memory_estimate(config, batch_size, sequence_length)
    free, total = torch.cuda.mem_get_info()
    return {**estimate, 'free_bytes': free, 'total_bytes': total,
            'allowed_bytes': int(free*fraction),
            'training_allowed': estimate['training_budget_bytes'] <= free*fraction,
            'inference_allowed': estimate['inference_budget_bytes'] <= free*fraction}


def construct_model(config, device='cpu'):
    """Initialize directly on destination, avoiding CPU/GPU model duplication.

    Meta construction performs shape verification without parameter allocation.
    """
    with torch.device(device):
        return TernaryTransformer(config)


def configure_execution(model, *, checkpointing=False, sdpa=False):
    model.activation_checkpointing = checkpointing
    for block in model.blocks:
        block.attention.use_sdpa = sdpa
    return model


def autocast_context(device, precision='fp32'):
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16' or torch.device(device).type != 'cuda' or not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 requires a supported CUDA GPU')
    return torch.autocast('cuda', dtype=torch.bfloat16)
