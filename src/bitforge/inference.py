"""Local checkpoint loading, bounded sampling, and measured inference."""
from dataclasses import asdict
import gc
from pathlib import Path
import json
import struct
import statistics
import time

import torch

from .config import TransformerConfig, estimate_parameters
from .execution import autocast_context, configure_execution, construct_model, preflight
from .packed import MAGIC, MAX_HEADER, MAX_PARAMETERS, _check_packed_config, load_packed
from .layers import BitLinear
from .quantization import least_squares_quantize, ternary_quantize
from .resources import cpu_memory, disk_preflight, host_preflight
from .tokenizer import ByteTokenizer
from .transformer import count_parameters


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def validate_device(device):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    return device


def inference_preflight(config, device, sequence_length=8):
    device = validate_device(device)
    # Incremental inference budget: decoded FP32 state + possible device staging,
    # with 512 MiB for payload/workspace. No optimizer/training state is needed.
    host = host_preflight(max(512 << 20, estimate_parameters(config)['total'] * 8 + (512 << 20)))
    result = {'host': host}
    if device.type == 'cuda':
        with torch.cuda.device(device):
            result['cuda'] = preflight(config, sequence_length=sequence_length)
        if not result['cuda']['inference_allowed']:
            raise RuntimeError('CUDA inference memory preflight failed')
    return result


def save_inference_checkpoint(path, model, *, training):
    """Portable dense FP32 master or reconstructed state, with explicit config."""
    path = Path(path)
    disk_preflight(path, count_parameters(model) * 4 + (1 << 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'inference_format_version': 1, 'config': asdict(model.config),
               'model': model.state_dict(), 'tokenizer': ByteTokenizer().metadata(),
               'training': json.loads(json.dumps(training, allow_nan=False))}
    with path.open('xb') as output:
        torch.save(payload, output)


def load_local_checkpoint(path, device='cpu', *, sdpa=True):
    """Load packed v1/v2, inference v1, or legacy training v1 checkpoints.

    Dense files must be trusted local artifacts. weights_only=True is mandatory.
    Legacy raw state dictionaries need conversion with their known configuration.
    """
    path = Path(path)
    device = validate_device(device)
    with path.open('rb') as source:
        packed = source.read(len(MAGIC)) == MAGIC
    # Host bound before torch.load: dense states can include optimizer state.
    if path.stat().st_size > 3 << 30:
        raise ValueError('checkpoint exceeds 3 GiB local inference limit')
    host_preflight(max(512 << 20, path.stat().st_size * 2 + (512 << 20)))
    start = time.perf_counter()
    if packed:
        # Header is untrusted here: use it only for bounded allocation preflight.
        # load_packed subsequently verifies the complete container and tensors.
        with path.open('rb') as source:
            source.read(len(MAGIC))
            size_bytes = source.read(4)
            if len(size_bytes) != 4:
                raise ValueError('truncated packed header')
            size = struct.unpack('<I', size_bytes)[0]
            if not 0 < size <= MAX_HEADER:
                raise ValueError('invalid packed header size')
            config = TransformerConfig(**json.loads(source.read(size))['config'])
        _check_packed_config(config)
        inference_preflight(config, device)
        model, header = load_packed(path)
        model = model.to(device)
        metadata = {'checkpoint_type': 'packed', 'source_config': header['config'],
                    'source_quantization_device': header['source_quantization_device'],
                    'training': 'not encoded in packed v1/v2; consult experiment provenance',
                    'integrity': 'container and per-tensor SHA-256 verified'}
    else:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or not (
                payload.get('inference_format_version') == 1 or payload.get('format_version') == 1):
            raise ValueError('unsupported dense checkpoint; expected config-bearing local checkpoint')
        ByteTokenizer.from_metadata(payload['tokenizer'])
        config = TransformerConfig(**payload['config'])
        _check_packed_config(config)
        if estimate_parameters(config)['total'] > MAX_PARAMETERS:
            raise ValueError('checkpoint is too large')
        inference_preflight(config, device)
        # Initialize only nonpersistent buffers after meta construction.
        model = construct_model(config, 'meta')
        from .transformer import SinusoidalPositions
        model.positions = SinusoidalPositions(config)
        for block in model.blocks:
            block.attention.future_mask = torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool).triu(1)
        state = payload['model']
        if not isinstance(state, dict):
            raise ValueError('dense model state must be a tensor dictionary')
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError('missing or extra dense state tensors')
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f'invalid dense tensor: {name}')
        model.load_state_dict(state, strict=True, assign=True)
        metadata = {'checkpoint_type': 'dense', 'source_config': asdict(config),
                    'training': payload.get('training', payload.get('progress', 'legacy training checkpoint')),
                    'integrity': 'weights_only load, strict shapes/dtypes and finite tensors; no embedded checksum'}
        del state, payload, expected
        gc.collect()
        model = model.to(device)
    model = configure_execution(model.eval().requires_grad_(False), sdpa=sdpa)
    synchronize(device)
    metadata.update(parameters=count_parameters(model), device=str(device),
                    checkpoint_bytes=path.stat().st_size,
                    load_seconds=time.perf_counter()-start,
                    runtime='dense floating-point tensors; no native ternary kernels',
                    cpu_memory=cpu_memory())
    return model, metadata


def freeze_quantized_projections(model):
    """Materialize dequantized weights once, on the reference quantizer device.

    Replace one projection at a time; no second full model is constructed.
    The resulting config describes its executable dense representation.
    """
    from dataclasses import replace
    for block in model.blocks:
        for parent, name in ((block.attention, 'qkv'), (block.attention, 'output'),
                             (block.ffn, 'up'), (block.ffn, 'down')):
            layer = getattr(parent, name)
            if not isinstance(layer, BitLinear):
                continue
            quantize = least_squares_quantize if model.config.projection_backend == 'ternary_ls' else ternary_quantize
            with torch.no_grad():
                weight = quantize(layer.weight).detach()
            with torch.device('meta'):
                dense = torch.nn.Linear(layer.in_features, layer.out_features, bias=False)
            dense.weight = torch.nn.Parameter(weight, requires_grad=False)
            setattr(parent, name, dense)
    model.config = replace(model.config, projection_backend='fp32')
    return model


def execution_preflight(model, sequence_length):
    """Recheck context-dependent GPU workspace after weights have been loaded."""
    device = next(model.parameters()).device
    if device.type == 'cuda':
        from .execution import memory_estimate
        estimate = memory_estimate(model.config, sequence_length=sequence_length)
        additional = estimate['inference_budget_bytes'] - estimate['fp32_weights_bytes']
        with torch.cuda.device(device):
            free, _ = torch.cuda.mem_get_info()
        if additional > free * .70:
            raise RuntimeError('context-dependent CUDA inference preflight failed; shorten the prompt/context')


def sample_token(logits, *, temperature=0., top_k=0, top_p=1., generator=None):
    if not 0 <= temperature < float('inf') or not 0 <= top_p <= 1 or top_p == 0:
        raise ValueError('temperature must be finite/nonnegative and top_p in (0,1]')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite sampling logits')
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k:
        threshold = scores.topk(min(top_k, scores.size(-1)), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, float('-inf'))
    if top_p < 1:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float('-inf'))
        scores = torch.full_like(scores, float('-inf')).scatter(-1, indices, ordered)
    return torch.multinomial(scores.softmax(-1), 1, generator=generator)


@torch.inference_mode()
def generate(model, prompt, *, max_new_tokens=32, temperature=0., top_k=0, top_p=1., seed=42, precision='fp32'):
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 512:
        raise ValueError('max_new_tokens must be 0..512')
    # Validate options even when zero tokens are requested.
    if not 0 <= temperature < float('inf') or not 0 < top_p <= 1 or type(top_k) is not int or top_k < 0:
        raise ValueError('invalid sampling options')
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError('prompt must not be empty')
    device = next(model.parameters()).device
    execution_preflight(model, min(model.config.max_seq_len, len(ids)+max_new_tokens))
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        with autocast_context(device, precision):
            # Only byte IDs are defined by the bundled tokenizer, including for
            # the 32K architectural target. This is an execution smoke test there.
            logits = model(tokens[:, -model.config.max_seq_len:])[:, -1, :256].float()
        selected = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=rng)
        tokens = torch.cat((tokens, selected), dim=1)
    synchronize(device)
    elapsed = time.perf_counter()-start
    output = tokens[0].tolist()
    return {'text': tokenizer.decode(output, errors='replace'), 'token_ids': output,
            'generated_tokens': max_new_tokens, 'seconds': elapsed,
            'tokens_per_second': max_new_tokens/elapsed if elapsed else 0.,
            'seed': seed, 'precision': precision, 'byte_only_sampling': True}


@torch.inference_mode()
def benchmark_model(model, *, sequence_length=8, warmup=1, repeats=3, precision='fp32', generation_tokens=4):
    if not 1 <= sequence_length <= model.config.max_seq_len or not 0 <= warmup <= 20 or not 1 <= repeats <= 100:
        raise ValueError('invalid benchmark length/warmup/repeats')
    device = next(model.parameters()).device
    execution_preflight(model, sequence_length)
    tokens = torch.arange(sequence_length, device=device)[None, :] % min(256, model.config.vocab_size)
    model.eval()
    def forward():
        with autocast_context(device, precision):
            return model(tokens)
    for _ in range(warmup):
        forward()
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        logits = forward()
        synchronize(device)
        samples.append(time.perf_counter()-start)
    if list(logits.shape) != [1, sequence_length, model.config.vocab_size] or not torch.isfinite(logits).all():
        raise RuntimeError('invalid forward output')
    forward_peak = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'reserved_bytes': torch.cuda.max_memory_reserved(device)} if device.type == 'cuda' else None
    generated = generate(model, 'The ', max_new_tokens=generation_tokens, precision=precision)
    return {'parameters': count_parameters(model), 'device': str(device), 'precision': precision,
            'projection_backend': model.config.projection_backend, 'sdpa': model.blocks[0].attention.use_sdpa,
            'batch_size': 1, 'sequence_length': sequence_length, 'warmup': warmup, 'repeats': repeats,
            'latency_seconds': samples, 'median_latency_seconds': statistics.median(samples),
            'output_shape': list(logits.shape), 'finite_logits': True,
            'forward_cuda_peak': forward_peak, 'cpu_memory': cpu_memory(), 'generation': generated}
