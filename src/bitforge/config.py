"""Validated architecture settings and allocation-free parameter arithmetic."""

from dataclasses import asdict, dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 64
    max_seq_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    norm_eps: float = 1e-6
    projection_backend: Literal['ternary', 'fp32'] = 'ternary'

    def __post_init__(self) -> None:
        if self.projection_backend not in ('ternary', 'fp32'):
            raise ValueError('projection_backend must be ternary or fp32')
        for name in ('vocab_size', 'max_seq_len', 'd_model', 'n_heads', 'n_layers', 'd_ff'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.d_model % 2:
            raise ValueError('sinusoidal positions require even d_model')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be positive and finite')


def tiny_config() -> TransformerConfig:
    return TransformerConfig()


def target_300m_config() -> TransformerConfig:
    """Return settings only. Never constructs tensors or modules."""
    return TransformerConfig(vocab_size=32000, max_seq_len=2048, d_model=1024,
                             n_heads=16, n_layers=24, d_ff=3328)


def estimate_parameters(config: TransformerConfig) -> dict[str, int]:
    """Unique trainable parameters, including a tied embedding/output weight."""
    d, f, layers = config.d_model, config.d_ff, config.n_layers
    counts = {
        'token_embedding_and_tied_head': config.vocab_size * d,
        'attention_projections': layers * 4 * d * d,
        'feed_forward_projections': layers * 2 * d * f,
        'rmsnorm_scales': (2 * layers + 1) * d,
    }
    counts['total'] = sum(counts.values())
    return counts


def architecture_summary(config: TransformerConfig) -> dict:
    return {'config': asdict(config), 'parameters': estimate_parameters(config)}
