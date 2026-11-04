import copy
from dataclasses import asdict, replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch import nn
from torch.nn import functional as F

from bitforge import BitLinear, TernaryTransformer, count_parameters, tiny_config
from bitforge.benchmark import aggregate_results, state_digest, storage_estimate
from bitforge.data import NextTokenBatcher
from bitforge.layers import make_projection
from bitforge.training import load_checkpoint, save_checkpoint


class ProjectionBackendTests(unittest.TestCase):
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
        self.config = replace(tiny_config(), vocab_size=259)
        torch.manual_seed(42)
        self.ternary = TernaryTransformer(self.config)
        torch.manual_seed(42)
        self.fp32 = TernaryTransformer(replace(self.config, projection_backend='fp32'))

    def test_exact_projection_types_and_architecture(self):
        for model, projection_type in ((self.ternary, BitLinear), (self.fp32, nn.Linear)):
            projections = [module for module in model.modules() if isinstance(module, nn.Linear)]
            self.assertEqual(len(projections), 4 * self.config.n_layers)
            self.assertTrue(all(type(module) is projection_type for module in projections))
            self.assertTrue(all(module.bias is None for module in projections))
            self.assertEqual(count_parameters(model), 82432)
        a, b = dict(self.ternary.named_modules()), dict(self.fp32.named_modules())
        self.assertEqual(a.keys(), b.keys())
        for name in a:
            if not isinstance(a[name], nn.Linear):
                self.assertIs(type(a[name]), type(b[name]))
        for name, tensor in self.ternary.named_buffers():
            torch.testing.assert_close(tensor, dict(self.fp32.named_buffers())[name], rtol=0, atol=0)

    def test_matched_initial_master_weights(self):
        self.assertEqual(state_digest(self.ternary), state_digest(self.fp32))
        self.assertEqual(self.ternary.state_dict().keys(), self.fp32.state_dict().keys())
        for name, value in self.ternary.state_dict().items():
            other = self.fp32.state_dict()[name]
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        tokens = torch.randint(0, 259, (2, 8))
        # With quantization replaced by identity, no hidden architecture difference remains.
        with patch('bitforge.layers.ternary_quantize', side_effect=lambda weight: weight):
            torch.testing.assert_close(self.ternary(tokens), self.fp32(tokens), rtol=0, atol=0)
        self.assertFalse(torch.equal(self.ternary(tokens), self.fp32(tokens)))

    def check_fp32(self, device):
        model = self.fp32.to(device).eval()
        tokens = torch.arange(16, device=device)[None, :]
        logits = model(tokens)
        changed = tokens.clone()
        changed[:, 8:] += 10
        torch.testing.assert_close(logits[:, :8], model(changed)[:, :8], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits.flatten(0, 1), ((tokens + 1) % 259).flatten()).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fp32_causality_and_gradients_cpu(self):
        self.check_fp32('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_fp32_causality_and_gradients_cuda(self):
        self.check_fp32('cuda')

    def test_backend_checkpoint_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fp32.pt'
            optimizer = torch.optim.AdamW(self.fp32.parameters())
            batcher = NextTokenBatcher(torch.arange(100), 8, 2, 42)
            save_checkpoint(path, self.fp32, optimizer, batcher, {'step': 0})
            restored, payload = load_checkpoint(path)
            self.assertEqual(restored.config.projection_backend, 'fp32')
            self.assertFalse(any(isinstance(m, BitLinear) for m in restored.modules()))
            self.assertEqual(state_digest(restored), state_digest(self.fp32))
            # Old Milestone 3 configs omitted this field and must still load as ternary.
            payload['config'].pop('projection_backend')
            legacy = Path(directory) / 'legacy.pt'
            torch.save(payload, legacy)
            old, _ = load_checkpoint(legacy)
            self.assertEqual(old.config.projection_backend, 'ternary')
            self.assertTrue(any(isinstance(m, BitLinear) for m in old.modules()))

    def test_storage_arithmetic_not_compression(self):
        result = storage_estimate(self.ternary)
        self.assertEqual(result['actual_parameter_tensor_bytes'], 329728)
        self.assertEqual(result['projection_parameters'], 65536)
        self.assertEqual(result['hypothetical_packed_total_bytes'], 84000)
        self.assertEqual(result['ternary_projection_modules'], 8)
        self.assertEqual(storage_estimate(self.fp32)['actual_parameter_tensor_bytes'], 329728)
        self.assertEqual(storage_estimate(self.fp32)['ternary_projection_modules'], 0)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            replace(self.config, projection_backend='unknown')
        with self.assertRaises(ValueError):
            make_projection(4, 4, 'unknown')

    def test_paired_aggregation_and_mismatch_rejection(self):
        runs = []
        for seed in (42, 43, 44):
            for backend in ('fp32', 'ternary'):
                value = 2.0 if backend == 'fp32' else 3.0
                runs.append({'seed': seed, 'backend': backend,
                             'initial_state_sha256': str(seed), 'batch_stream_sha256': str(seed),
                             'settings': {}, 'corpus': {}, 'parameters': 82432,
                             'config': asdict(replace(self.config, projection_backend=backend)),
                             'final_train': {'loss': value, 'perplexity': value},
                             'final_validation': {'loss': value, 'perplexity': value},
                             'training': {'seconds': value, 'peak_allocated_bytes': 10, 'peak_reserved_bytes': 20},
                             'inference': {'mean_ms': value, 'peak_allocated_bytes': 5, 'peak_reserved_bytes': 10},
                             'training_checkpoint_bytes': 1000, 'weights_checkpoint_bytes': 400})
        summary = aggregate_results(runs)
        self.assertEqual(summary['by_backend']['fp32']['validation_loss']['mean'], 2)
        self.assertEqual(summary['by_backend']['ternary']['validation_loss']['sample_stdev'], 0)
        self.assertEqual(summary['paired_ternary_minus_fp32']['validation_loss']['mean'], 1)
        for field in ('initial_state_sha256', 'batch_stream_sha256', 'settings', 'corpus', 'parameters'):
            changed = copy.deepcopy(runs)
            changed[0][field] = 'different'
            with self.assertRaises(ValueError):
                aggregate_results(changed)
        with self.assertRaises(ValueError):
            aggregate_results(runs[:-1])


if __name__ == '__main__':
    unittest.main()
