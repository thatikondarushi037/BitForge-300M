"""Versioned two-bit ternary inference files: JSON and bytes, never pickle.

Eligible tensors are BitLinear.weight only. Export quantizes on the source
device using the existing quantizer; import uses frozen dense nn.Linear so it
never re-quantizes already reconstructed weights. Serialized tensors are FP32.
"""

from dataclasses import asdict, fields, replace
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import torch
from torch import Tensor

from .config import TransformerConfig, estimate_parameters
from .layers import BitLinear
from .quantization import ternary_codes, least_squares_codes
from .tokenizer import ByteTokenizer
from .transformer import TernaryTransformer
from .resources import disk_preflight

MAGIC = b'BFTPACK1'
MAX_HEADER = 1 << 20
MAX_FILE = 768 << 20
MAX_PARAMETERS = 350_000_000
LS_QUANTIZER = "alternating-ls4-fp32-round-even-v1"
ARCHITECTURE = 'bitforge-decoder-v1'
QUANTIZER = 'absmean-fp32-round-even-clamp-v1'


def eligible_weight_names(model: TernaryTransformer) -> set[str]:
    return {name + '.weight' for name, module in model.named_modules() if isinstance(module, BitLinear)}


def pack_ternary(values: Tensor) -> bytes:
    """Row-major, little-endian bit lanes: -1=00, 0=01, +1=10, 11 invalid."""
    values = values.detach().cpu().flatten()
    if not 1 <= values.numel() <= 16_000_000 or not torch.isfinite(values).all():
        raise ValueError('invalid ternary tensor size or values')
    if not ((values == -1) | (values == 0) | (values == 1)).all():
        raise ValueError('expected ternary values')
    codes = (values.to(torch.int8) + 1).to(torch.uint8)
    padding = (-len(codes)) % 4
    codes = torch.cat((codes, torch.ones(padding, dtype=torch.uint8, device='cpu'))).reshape(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return ctypes.string_at(packed.data_ptr(), packed.numel())


def unpack_ternary(data: bytes, count: int) -> Tensor:
    if type(count) is not int or not 1 <= count <= 16_000_000 or len(data) != (count + 3) // 4:
        raise ValueError('invalid packed length')
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    codes = ((packed[:, None] >> torch.tensor([0, 2, 4, 6], device='cpu')) & 3).flatten()
    if (codes == 3).any():
        raise ValueError('reserved two-bit code 11')
    if (codes[count:] != 1).any():
        raise ValueError('noncanonical padding')
    return (codes[:count].to(torch.int8) - 1).to(torch.float32)


def _tensor_bytes(value: Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    return ctypes.string_at(value.data_ptr(), value.numel() * value.element_size())


def _json_bytes(header: dict) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _assemble(header: dict, payload: bytes) -> bytes:
    metadata = _json_bytes(header)
    if len(metadata) > MAX_HEADER:
        raise ValueError('header too large')
    body = MAGIC + struct.pack('<I', len(metadata)) + metadata + payload
    return body + hashlib.sha256(body).digest()


def export_packed(path: Path, model: TernaryTransformer) -> dict:
    if sys.byteorder != 'little':
        raise ValueError('version 1 requires a little-endian host')
    _check_packed_config(model.config)
    if model.config.projection_backend not in ('ternary', 'ternary_ls'):
        raise ValueError('export requires a ternary source model')
    # Check before quantization or payload allocation, leaving 1 GiB free.
    disk_preflight(path, estimate_parameters(model.config)['total'] * 4 + MAX_HEADER)
    eligible = eligible_weight_names(model)
    if len(eligible) != 4 * model.config.n_layers:
        raise ValueError('unsupported projection architecture')
    entries, chunks, offset = [], [], 0
    eligible_count = 0
    for name, value in model.state_dict().items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('version 1 requires finite FP32 state tensors')
        if name in eligible:
            codes, scale = (least_squares_codes if model.config.projection_backend == 'ternary_ls' else ternary_codes)(value)  # Source-device reduction, exactly as BitLinear.
            blob = struct.pack('<f', scale.item()) + pack_ternary(codes)
            encoding = 'ternary2-f32scale'
            eligible_count += value.numel()
        else:
            blob = _tensor_bytes(value)
            encoding = 'raw-f32'
        entries.append({'name': name, 'shape': list(value.shape), 'dtype': 'float32',
                        'encoding': encoding, 'offset': offset, 'nbytes': len(blob),
                        'sha256': hashlib.sha256(blob).hexdigest()})
        chunks.append(blob)
        offset += len(blob)
    header = {'format': 'bitforge-packed-ternary', 'version': 2 if model.config.projection_backend == 'ternary_ls' else 1,
              'architecture': ARCHITECTURE, 'config': asdict(model.config),
              'tokenizer': ByteTokenizer().metadata(), 'quantizer': LS_QUANTIZER if model.config.projection_backend == 'ternary_ls' else QUANTIZER,
              'byte_order': 'little', 'source_quantization_device': str(next(model.parameters()).device),
              'tensors': entries}
    data = _assemble(header, b''.join(chunks))
    if len(data) > MAX_FILE:
        raise ValueError('file exceeds format resource bound')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(data)
    total = sum(value.numel() for value in model.state_dict().values())
    return {'packed_bytes': len(data), 'eligible_parameters': eligible_count,
            'noneligible_parameters': total - eligible_count, 'total_parameters': total,
            'eligible_percent': eligible_count / total * 100,
            'effective_file_bits_per_parameter': len(data) * 8 / total,
            'payload_bytes': offset, 'header_bytes': len(_json_bytes(header)),
            'sha256': hashlib.sha256(data).hexdigest(), 'manifest': header}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'invalid JSON constant: {value}')


def _read_container(path: Path) -> tuple[dict, bytes]:
    with path.open('rb') as source:
        data = source.read(MAX_FILE + 1)
    if len(data) > MAX_FILE or len(data) < 44 or data[:8] != MAGIC:
        raise ValueError('invalid file magic, size, or truncated header')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('file integrity checksum mismatch')
    size = struct.unpack('<I', data[8:12])[0]
    if not 0 < size <= MAX_HEADER or 12 + size > len(data) - 32:
        raise ValueError('invalid header length')
    try:
        header = json.loads(data[12:12+size], object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid JSON manifest') from error
    return header, data[12+size:-32]


def load_packed(path: Path, device: str | torch.device = 'cpu',
                expected_config: TransformerConfig | None = None) -> tuple[TernaryTransformer, dict]:
    """Strict bounded import; all tensors validate before any state is loaded.

    Checksums detect corruption, not authenticity. There is no torch.load,
    pickle, dynamic class import, or fallback for missing tensors in this path.
    """
    if sys.byteorder != 'little':
        raise ValueError('version 1 requires a little-endian host')
    header, payload = _read_container(path)
    keys = {'format', 'version', 'architecture', 'config', 'tokenizer', 'quantizer',
            'byte_order', 'source_quantization_device', 'tensors'}
    if not isinstance(header, dict) or set(header) != keys:
        raise ValueError('invalid manifest fields')
    if (header['format'] != 'bitforge-packed-ternary' or type(header['version']) is not int
            or header['version'] not in (1, 2) or header['architecture'] != ARCHITECTURE
            or header['quantizer'] != (QUANTIZER if header['version'] == 1 else LS_QUANTIZER) or header['byte_order'] != 'little'
            or not isinstance(header['source_quantization_device'], str)):
        raise ValueError('unsupported format, architecture, or quantizer')
    if not isinstance(header['config'], dict) or set(header['config']) != {f.name for f in fields(TransformerConfig)}:
        raise ValueError('invalid architecture config fields')
    config = TransformerConfig(**header['config'])
    _check_packed_config(config)  # Reject oversized configurations before allocation.
    if config.projection_backend != ('ternary' if header['version'] == 1 else 'ternary_ls') or (expected_config is not None and config != expected_config):
        raise ValueError('incompatible source architecture')
    ByteTokenizer.from_metadata(header['tokenizer'])
    with torch.device('meta'):
        template = TernaryTransformer(config).float()
    expected = template.state_dict()
    eligible = eligible_weight_names(template)
    entries = header['tensors']
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('missing or extra state tensors')
    reconstructed, cursor = {}, 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'name', 'shape', 'dtype', 'encoding', 'offset', 'nbytes', 'sha256'}:
            raise ValueError('invalid tensor metadata')
        name = entry['name']
        if not isinstance(name, str) or name not in expected or name in reconstructed:
            raise ValueError('missing, unknown, or duplicate tensor name')
        shape = entry['shape']
        if (not isinstance(shape, list) or any(type(n) is not int for n in shape)
                or shape != list(expected[name].shape) or entry['dtype'] != 'float32'):
            raise ValueError('tensor shape or dtype mismatch')
        count = expected[name].numel()
        packed = name in eligible
        encoding = 'ternary2-f32scale' if packed else 'raw-f32'
        size = 4 + (count + 3) // 4 if packed else 4 * count
        if (entry['encoding'] != encoding or type(entry['offset']) is not int or entry['offset'] != cursor
                or type(entry['nbytes']) is not int or entry['nbytes'] != size
                or cursor + size > len(payload)):
            raise ValueError('encoding, offset, length, or eligibility mismatch')
        blob = payload[cursor:cursor+size]
        if not isinstance(entry['sha256'], str) or hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('tensor integrity checksum mismatch')
        if packed:
            scale = struct.unpack('<f', blob[:4])[0]
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('invalid quantization scale')
            value = unpack_ternary(blob[4:], count) * torch.tensor(scale, dtype=torch.float32, device='cpu')
        else:
            value = torch.frombuffer(bytearray(blob), dtype=torch.float32).clone()
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite tensor payload')
        reconstructed[name] = value.reshape(shape)
        cursor += size
    if cursor != len(payload) or set(reconstructed) != set(expected):
        raise ValueError('unreferenced payload or missing weights')
    # Dense frozen projections prevent the mathematically incorrect second quantization.
    with torch.device('meta'):
        model = TernaryTransformer(replace(config, projection_backend='fp32')).float()
    # Nonpersistent buffers must be regenerated after meta construction.
    from .transformer import SinusoidalPositions
    model.positions = SinusoidalPositions(config)
    for block in model.blocks:
        block.attention.future_mask = torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool).triu(1)
    model.load_state_dict(reconstructed, strict=True, assign=True)
    return model.requires_grad_(False).eval().to(device), header


def _check_packed_config(config: TransformerConfig) -> None:
    if (config.n_layers > 128 or config.d_model > 4096 or config.d_ff > 16384
            or config.vocab_size > 65536 or estimate_parameters(config)['total'] > MAX_PARAMETERS or config.max_seq_len > 4096):
        raise ValueError('packed execution limited to vocabulary 65536, 350M parameters, context 4096')
