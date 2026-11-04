from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch import nn

from bitforge import BitLinear, TernaryTransformer, tiny_config
from bitforge.packed import (MAGIC, _assemble, _read_container, eligible_weight_names,
                             export_packed, load_packed, pack_ternary, unpack_ternary)
from bitforge.quantization import ternary_quantize


class PackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        torch.manual_seed(42)
        self.config = replace(tiny_config(), vocab_size=259, d_model=16,
                              d_ff=32, n_layers=1, n_heads=2, max_seq_len=8)
        self.model = TernaryTransformer(self.config).eval()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.bft'

    def test_bit_layout_roundtrip_and_padding(self):
        self.assertEqual(pack_ternary(torch.tensor([-1, 0, 1, -1])), bytes([36]))
        for count in range(1, 18):
            values = (torch.arange(count) % 3 - 1).float()
            packed = pack_ternary(values)
            self.assertEqual(len(packed), (count + 3) // 4)
            torch.testing.assert_close(unpack_ternary(packed, count), values, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            unpack_ternary(b'\x00', 1)  # Padding must be code 01, not 00.
        with self.assertRaises(ValueError):
            unpack_ternary(b'\x03', 4)
        with self.assertRaises(ValueError):
            unpack_ternary(b'', 1)
        with self.assertRaises(ValueError):
            pack_ternary(torch.tensor([0., 0.5]))

    def check_model_roundtrip(self, device):
        source = self.model.to(device)
        info = export_packed(self.path, source)
        restored, manifest = load_packed(self.path, device, expected_config=self.config)
        eligible = eligible_weight_names(source)
        self.assertEqual(len(eligible), 4)
        self.assertFalse(any(isinstance(m, BitLinear) for m in restored.modules()))
        self.assertEqual(sum(type(m) is nn.Linear for m in restored.modules()), 4)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        for name, value in source.state_dict().items():
            expected = ternary_quantize(value) if name in eligible else value
            torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
        for length in (1, 5, 8):
            tokens = torch.randint(0, 259, (2, length), device=device)
            with torch.no_grad():
                torch.testing.assert_close(source(tokens), restored(tokens), rtol=1e-5, atol=1e-6)
        self.assertEqual(info['total_parameters'], sum(p.numel() for p in source.parameters()))
        self.assertEqual(manifest['config'], dict(source.config.__dict__))

    def test_model_roundtrip_cpu(self):
        self.check_model_roundtrip('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_model_roundtrip_cuda(self):
        self.check_model_roundtrip('cuda')

    def test_export_deterministic_and_exclusive(self):
        export_packed(self.path, self.model)
        other = self.path.with_name('second.bft')
        export_packed(other, self.model)
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        with self.assertRaises(FileExistsError):
            export_packed(self.path, self.model)

    def test_explicit_fp32_and_cpu_independent_of_defaults(self):
        export_packed(self.path, self.model)
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            restored, _ = load_packed(self.path, 'cpu')
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu'
                                for p in restored.parameters()))
            if torch.cuda.is_available():
                with torch.device('cuda'):
                    restored, _ = load_packed(self.path, 'cpu')
                self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu'
                                    for p in restored.parameters()))
        finally:
            torch.set_default_dtype(old_dtype)

    def test_ineligible_tensors_and_source_modes(self):
        eligible = eligible_weight_names(self.model)
        self.assertTrue(all('attention.qkv.weight' in n or 'attention.output.weight' in n
                            or 'ffn.up.weight' in n or 'ffn.down.weight' in n for n in eligible))
        self.assertNotIn('token_embedding.weight', eligible)
        with self.assertRaises(ValueError):
            export_packed(self.path, TernaryTransformer(replace(self.config, projection_backend='fp32')))
        with self.assertRaises(ValueError):
            export_packed(self.path, self.model.double())

    def test_corruption_and_truncation(self):
        export_packed(self.path, self.model)
        original = self.path.read_bytes()
        for length in (0, 8, 12, len(original) - 1, len(original) - 33):
            self.path.write_bytes(original[:length])
            with self.assertRaises(ValueError):
                load_packed(self.path)
        changed = bytearray(original)
        changed[-40] ^= 1
        self.path.write_bytes(changed)
        with self.assertRaises(ValueError):
            load_packed(self.path)

    def rewrite(self, mutate):
        export_packed(self.path, self.model)
        header, payload = _read_container(self.path)
        header, payload = mutate(header, bytearray(payload))
        self.path.write_bytes(_assemble(header, payload))

    def test_invalid_code_with_valid_checksums(self):
        def mutate(header, payload):
            entry = next(e for e in header['tensors'] if e['encoding'] == 'ternary2-f32scale')
            payload[entry['offset'] + 4] |= 3
            blob = payload[entry['offset']:entry['offset']+entry['nbytes']]
            entry['sha256'] = hashlib.sha256(blob).hexdigest()
            return header, payload
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, 'reserved'):
            load_packed(self.path)

    def test_bad_scale_with_valid_checksums(self):
        def mutate(header, payload):
            entry = next(e for e in header['tensors'] if e['encoding'] == 'ternary2-f32scale')
            payload[entry['offset']:entry['offset']+4] = struct.pack('<f', float('nan'))
            entry['sha256'] = hashlib.sha256(payload[entry['offset']:entry['offset']+entry['nbytes']]).hexdigest()
            return header, payload
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, 'scale'):
            load_packed(self.path)

    def test_tensor_checksum(self):
        def mutate(header, payload):
            payload[0] ^= 1
            return header, payload
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, 'tensor integrity'):
            load_packed(self.path)

    def test_metadata_mismatches(self):
        export_packed(self.path, self.model)
        original = self.path.read_bytes()
        for field, value in (('shape', [1, 1]), ('dtype', 'float16'), ('name', 'unknown.weight'),
                             ('offset', 1), ('nbytes', 1), ('encoding', 'ternary2-f32scale')):
            self.path.write_bytes(original)
            header, payload = _read_container(self.path)
            header['tensors'][0][field] = value
            self.path.write_bytes(_assemble(header, payload))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_packed(self.path)

    def test_missing_duplicate_and_extra_payload(self):
        export_packed(self.path, self.model)
        original = self.path.read_bytes()
        for mode in ('missing', 'duplicate', 'extra'):
            self.path.write_bytes(original)
            header, payload = _read_container(self.path)
            if mode == 'missing':
                header['tensors'].pop()
            elif mode == 'duplicate':
                header['tensors'][-1] = dict(header['tensors'][0])
            else:
                payload += b'\x00'
            self.path.write_bytes(_assemble(header, payload))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                load_packed(self.path)

    def test_incompatible_architecture_and_bounds(self):
        export_packed(self.path, self.model)
        with self.assertRaises(ValueError):
            load_packed(self.path, expected_config=replace(self.config, n_heads=4))
        original = self.path.read_bytes()
        for field, value in (('version', 2), ('architecture', 'unknown'), ('quantizer', 'unknown')):
            self.path.write_bytes(original)
            header, payload = _read_container(self.path)
            header[field] = value
            self.path.write_bytes(_assemble(header, payload))
            with self.assertRaises(ValueError):
                load_packed(self.path)
        self.path.write_bytes(original)
        header, payload = _read_container(self.path)
        header['config']['n_layers'] = 10000
        self.path.write_bytes(_assemble(header, payload))
        with patch('bitforge.packed.TernaryTransformer', side_effect=AssertionError('must not construct')):
            with self.assertRaises(ValueError):
                load_packed(self.path)

    def test_no_pickle_loading_and_duplicate_json(self):
        export_packed(self.path, self.model)
        with patch('torch.load', side_effect=AssertionError('pickle forbidden')):
            load_packed(self.path)
        metadata = b'{"version":1,"version":1}'
        body = MAGIC + struct.pack('<I', len(metadata)) + metadata
        self.path.write_bytes(body + hashlib.sha256(body).digest())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_packed(self.path)


if __name__ == '__main__':
    unittest.main()
