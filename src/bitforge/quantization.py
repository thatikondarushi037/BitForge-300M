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
