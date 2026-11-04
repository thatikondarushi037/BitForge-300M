"""Small, explicit building blocks for ternary weight research."""

from .layers import BitLinear
from .quantization import ternary_codes, ternary_quantize
from .config import TransformerConfig, estimate_parameters, target_300m_config, tiny_config
from .transformer import TernaryTransformer, count_parameters

__all__ = ["BitLinear", "ternary_codes", "ternary_quantize", "TransformerConfig",
           "estimate_parameters", "target_300m_config", "tiny_config",
           "TernaryTransformer", "count_parameters"]
