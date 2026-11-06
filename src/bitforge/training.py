"""Token-weighted evaluation, bounded local checkpoints, and greedy decoding."""

from dataclasses import asdict
import math
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import TransformerConfig, estimate_parameters
from .data import NextTokenBatcher, evaluation_batches
from .tokenizer import ByteTokenizer
from .transformer import TernaryTransformer


def evaluate(model: TernaryTransformer, tokens: Tensor, sequence_length: int, batch_size: int) -> dict:
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.no_grad():
            for x, y in evaluation_batches(tokens, sequence_length, batch_size):
                logits = model(x.to(device))
                loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten(), reduction='sum')
                total_loss += loss.item()
                total_tokens += y.numel()
    finally:
        model.train(previous_mode)
    mean = total_loss / total_tokens
    if not math.isfinite(mean):
        raise RuntimeError('nonfinite evaluation loss')
    return {'loss': mean, 'perplexity': math.exp(mean) if mean < 709 else float('inf'),
            'tokens': total_tokens}


def check_small_config(config: TransformerConfig) -> None:
    if config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError('model vocabulary must match byte tokenizer')
    if estimate_parameters(config)['total'] > 2_000_000 or config.max_seq_len > 256:
        raise ValueError('local pipeline limited to 2M parameters and context 256')


def save_checkpoint(path: Path, model: TernaryTransformer, optimizer: torch.optim.Optimizer,
                    batcher: NextTokenBatcher, progress: dict) -> None:
    check_small_config(model.config)
    payload = {
        'format_version': 1, 'config': asdict(model.config),
        'tokenizer': ByteTokenizer().metadata(), 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'progress': json.loads(json.dumps(progress, allow_nan=False)),
        'batch_rng': batcher.generator.get_state(), 'cpu_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves old experiments. No pickle globals are needed.
    with path.open('xb') as output:
        torch.save(payload, output)


def load_checkpoint(path: Path, device: str | torch.device = 'cpu'):
    """Load locally produced files only. Never retry with weights_only=False."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format')
    ByteTokenizer.from_metadata(payload['tokenizer'])
    config = TransformerConfig(**payload['config'])
    check_small_config(config)  # Reject large model settings before construction.
    model = TernaryTransformer(config).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model, payload


def restore_training_state(payload: dict, optimizer: torch.optim.Optimizer,
                           batcher: NextTokenBatcher, device: str | torch.device) -> None:
    optimizer.load_state_dict(payload['optimizer'])
    batcher.generator.set_state(payload['batch_rng'])
    torch.set_rng_state(payload['cpu_rng'])
    if torch.device(device).type == 'cuda' and payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])


@torch.no_grad()
def greedy_generate(model: TernaryTransformer, tokens: Tensor, max_new_tokens: int) -> Tensor:
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1 or max_new_tokens < 0:
        raise ValueError('generation needs one nonempty prompt and nonnegative token count')
    previous_mode = model.training
    model.eval()
    try:
        result = tokens.clone()
        for _ in range(max_new_tokens):
            logits = model(result[:, -model.config.max_seq_len:])[:, -1, :].clone()
            # Specials are reserved but not trained. Generate only real byte IDs.
            logits[:, 256:] = float('-inf')
            result = torch.cat((result, logits.argmax(-1, keepdim=True)), dim=1)
        return result
    finally:
        model.train(previous_mode)
