"""Small decoder-only Transformer using the existing ternary QAT projections."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import TransformerConfig
from .layers import make_projection


class RMSNorm(nn.Module):
    """Learned scale, no bias; accumulate low-precision statistics in FP32."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        stats = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = stats * torch.rsqrt(stats.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class SinusoidalPositions(nn.Module):
    """Fixed, non-trainable absolute positions; context starts at position zero."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, config.d_model, 2, dtype=torch.float32)
                                * (-math.log(10000.0) / config.d_model))
        angles = positions * frequencies
        encoding = torch.empty(config.max_seq_len, config.d_model)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:x.size(1)].to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    """Explicit scaled dot-product attention. True mask entries are blocked."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = make_projection(config.d_model, 3 * config.d_model, config.projection_backend)
        self.output = make_projection(config.d_model, config.d_model, config.projection_backend)
        self.register_buffer('future_mask', torch.ones(config.max_seq_len, config.max_seq_len,
                             dtype=torch.bool).triu(1), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.future_mask[:length, :length], float('-inf'))
        # FP32 softmax supports future low-precision use; FP64 stays FP64.
        probabilities = F.softmax(scores, dim=-1,
                                  dtype=torch.float32 if scores.dtype in
                                  (torch.float16, torch.bfloat16) else scores.dtype).to(v.dtype)
        attended = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(attended)


class TernaryFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.up = make_projection(config.d_model, config.d_ff, config.projection_backend)
        self.down = make_projection(config.d_ff, config.d_model, config.projection_backend)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm residual attention followed by pre-norm residual FFN."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = TernaryFeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TernaryTransformer(nn.Module):
    """Unnormalized next-token logits [batch, time, vocabulary], no KV cache."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.positions = SinusoidalPositions(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.size(0) == 0 or not 1 <= tokens.size(1) <= self.config.max_seq_len:
            raise ValueError('tokens must have shape [nonempty batch, time] within context limit')
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError('tokens must contain integer token IDs (int32 or int64)')
        x = self.positions(self.token_embedding(tokens))
        for block in self.blocks:
            x = block(x)
        # Functional projection shares the embedding Parameter; no duplicate head.
        return F.linear(self.final_norm(x), self.token_embedding.weight)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
