"""Reusable QAT linear layer with floating-point master parameters."""

from torch import Tensor, nn
from torch.nn import functional as F

from .quantization import ternary_quantize, least_squares_quantize


class BitLinear(nn.Linear):
    """Linear API with absmean ternary weights in training and evaluation.

    Bias and activations stay floating point. Normalization belongs to the caller.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, ternary_quantize(self.weight), self.bias)


class LeastSquaresBitLinear(BitLinear):
    """Same FP32 masters and identity STE, alternating least-squares forward."""

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, least_squares_quantize(self.weight), self.bias)


def make_projection(in_features: int, out_features: int, backend: str) -> nn.Linear:
    """Identical bias-free parameter layouts and initialization in either backend."""
    if backend == 'ternary':
        return BitLinear(in_features, out_features, bias=False)
    if backend == 'ternary_ls':
        return LeastSquaresBitLinear(in_features, out_features, bias=False)
    if backend == 'fp32':
        return nn.Linear(in_features, out_features, bias=False)
    raise ValueError('unknown projection backend')
