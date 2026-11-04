"""Stateless UTF-8 byte tokenizer with an explicitly versioned ID mapping."""


class ByteTokenizer:
    vocab_size = 259
    bos_id, eos_id, pad_id = 256, 257, 258

    def metadata(self) -> dict:
        return {'type': 'utf8-byte', 'version': 1, 'vocab_size': self.vocab_size,
                'bos_id': self.bos_id, 'eos_id': self.eos_id, 'pad_id': self.pad_id}

    @classmethod
    def from_metadata(cls, metadata: dict) -> 'ByteTokenizer':
        tokenizer = cls()
        if metadata != tokenizer.metadata():
            raise ValueError('unsupported tokenizer metadata')
        return tokenizer

    def encode(self, text: str) -> list[int]:
        # No automatic BOS/EOS insertion or Unicode normalization.
        return list(text.encode('utf-8'))

    def decode(self, ids: list[int], *, errors: str = 'strict', skip_special: bool = True) -> str:
        values = []
        for token in ids:
            if type(token) is not int or not 0 <= token < self.vocab_size:
                raise ValueError('invalid token ID')
            if token >= 256:
                if skip_special:
                    continue
                raise ValueError('special tokens have no byte representation')
            values.append(token)
        return bytes(values).decode('utf-8', errors=errors)
