"""Absmean ternary weight quantization with an identity STE."""

import torch
from torch import Tensor


def ternary_codes(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Return floating-point ternary codes and a detached per-tensor scale.

    FP16/BF16 statistics use FP32. No bit packing or integer kernels are used.
    """
    if not weight.is_floating_point():
        raise TypeError("weights must be floating point")
    if weight.numel() == 0:
        raise ValueError("weights must be nonempty")
    stats = weight.detach()
    if stats.dtype in (torch.float16, torch.bfloat16):
        stats = stats.float()
    scale = stats.abs().mean().clamp_min(torch.finfo(stats.dtype).eps)
    codes = (stats / scale).round().clamp(-1, 1)
    return codes.to(weight.dtype), scale


def ternary_quantize(weight: Tensor) -> Tensor:
    """Dequantized ternary forward values; identity surrogate derivative."""
    codes, scale = ternary_codes(weight)
    quantized = (codes * scale).to(weight.dtype)
    # Forward is exactly quantized, including under cancellation-sensitive inputs.
    return quantized.detach() + (weight - weight.detach())


def least_squares_codes(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Four alternating minimization steps for ||W - alpha Q||^2.

    For fixed alpha, nearest ternary rounding minimizes error. For fixed Q,
    alpha=<W,Q>/<Q,Q> is the exact minimizer. Starting from absmean makes
    reconstruction MSE nonincreasing relative to the original quantizer.
    This does not imply improved language-model loss. Statistics are detached.
    """
    codes, scale = ternary_codes(weight)
    stats = weight.detach().float() if weight.dtype in (torch.float16, torch.bfloat16) else weight.detach()
    codes = codes.to(stats.dtype)
    for _ in range(4):
        scale = ((stats * codes).sum() / codes.square().sum().clamp_min(1)).clamp_min(torch.finfo(stats.dtype).eps)
        codes = (stats / scale).round().clamp(-1, 1)
    return codes.to(weight.dtype), scale


def least_squares_quantize(weight: Tensor) -> Tensor:
    codes, scale = least_squares_codes(weight)
    return (codes * scale).to(weight.dtype).detach() + (weight - weight.detach())
