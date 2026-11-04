"""Split first, then generate next-token windows within each isolated stream."""

import hashlib
import json
from pathlib import Path
import re

import torch
from torch import Tensor

from .tokenizer import ByteTokenizer


def load_corpus(path: Path, validation_fraction: float = 0.2) -> tuple[Tensor, Tensor, dict]:
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest_path = path.with_suffix('.manifest.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if digest != manifest['sha256'] or len(raw) != manifest['bytes']:
            raise ValueError('corpus does not match its manifest')
    text = raw.decode('utf-8')
    cut = int(len(text) * (1 - validation_fraction))
    boundaries = [match.end() for match in re.finditer('\n\n', text) if match.end() < len(text)]
    if boundaries:
        cut = min(boundaries, key=lambda boundary: abs(boundary - cut))
    tokenizer = ByteTokenizer()
    train = torch.tensor(tokenizer.encode(text[:cut]), dtype=torch.long)
    validation = torch.tensor(tokenizer.encode(text[cut:]), dtype=torch.long)
    if min(train.numel(), validation.numel()) < 2:
        raise ValueError('each split needs at least two tokens')
    metadata = {'path': str(path), 'bytes': len(raw), 'sha256': digest,
                'requested_validation_fraction': validation_fraction,
                'split_byte_offset': len(text[:cut].encode('utf-8')),
                'train_tokens': train.numel(), 'validation_tokens': validation.numel(),
                'train_sha256': hashlib.sha256(text[:cut].encode('utf-8')).hexdigest(),
                'validation_sha256': hashlib.sha256(text[cut:].encode('utf-8')).hexdigest()}
    return train, validation, metadata


class NextTokenBatcher:
    """Uniformly sample overlapping windows, with a private CPU RNG."""

    def __init__(self, tokens: Tensor, sequence_length: int, batch_size: int, seed: int) -> None:
        if sequence_length < 1 or batch_size < 1 or tokens.ndim != 1 or len(tokens) <= sequence_length:
            raise ValueError('need a 1D stream, positive batch/length, and at least length+1 tokens')
        self.tokens = tokens.detach().cpu().clone()
        self.sequence_length, self.batch_size = sequence_length, batch_size
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def sample(self, device: str | torch.device = 'cpu') -> tuple[Tensor, Tensor]:
        starts = torch.randint(len(self.tokens) - self.sequence_length, (self.batch_size,),
                               generator=self.generator)
        windows = self.tokens[starts[:, None] + torch.arange(self.sequence_length + 1)]
        return windows[:, :-1].to(device), windows[:, 1:].to(device)


def evaluation_batches(tokens: Tensor, sequence_length: int, batch_size: int):
    """Visit every adjacent-token target once, with a possibly shorter final batch."""
    if sequence_length < 1 or batch_size < 1 or tokens.ndim != 1 or len(tokens) < 2:
        raise ValueError('invalid evaluation stream or batch settings')
    full = (len(tokens) - 1) // sequence_length
    for first in range(0, full, batch_size):
        starts = torch.arange(first, min(first + batch_size, full)) * sequence_length
        windows = tokens[starts[:, None] + torch.arange(sequence_length + 1)]
        yield windows[:, :-1], windows[:, 1:]
    start = full * sequence_length
    if start < len(tokens) - 1:
        yield tokens[start:-1][None, :], tokens[start+1:][None, :]
