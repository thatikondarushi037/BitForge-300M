import io
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as F

from bitforge import (BitLinear, TernaryTransformer, count_parameters,
                      estimate_parameters, target_300m_config, tiny_config)
from bitforge.transformer import CausalSelfAttention, RMSNorm, SinusoidalPositions


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        cls.previous_threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.use_deterministic_algorithms(cls.previous_deterministic)
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = tiny_config()

    def test_config_validation(self):
        for changes in ({'d_model': 63}, {'n_heads': 3}, {'n_layers': 0},
                        {'vocab_size': True}, {'max_seq_len': -1},
                        {'d_ff': 0}, {'norm_eps': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_positions_and_shapes(self):
        positions = SinusoidalPositions(self.config)
        x = positions(torch.zeros(2, 4, self.config.d_model))
        torch.testing.assert_close(x[0, 0, 0::2], torch.zeros(32))
        torch.testing.assert_close(x[0, 0, 1::2], torch.ones(32))
        self.assertFalse(torch.equal(x[0, 0], x[0, 1]))
        self.assertEqual(sum(p.numel() for p in positions.parameters()), 0)
        model = TernaryTransformer(self.config)
        for batch, length in ((1, 1), (2, 7), (3, 32)):
            logits = model(torch.randint(0, 64, (batch, length)))
            self.assertEqual(logits.shape, (batch, length, 64))
            self.assertTrue(torch.isfinite(logits).all())

    def test_rmsnorm(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 3, 8, requires_grad=True)
        torch.testing.assert_close(norm(x), x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
        self.assertTrue(torch.isfinite(norm(torch.zeros_like(x))).all())
        norm(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(norm.weight.grad).all())

    def test_attention_matches_prefix_reference(self):
        attention = CausalSelfAttention(self.config).eval()
        x = torch.randn(2, 5, 64)
        q, k, v = attention.qkv(x).reshape(2, 5, 3, 4, 16).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # Independently compute each position using only its prefix, no mask.
        outputs = []
        for t in range(5):
            scores = (q[:, :, t:t+1] @ k[:, :, :t+1].transpose(-2, -1)) / 4.0
            outputs.append(scores.softmax(-1) @ v[:, :, :t+1])
        reference = torch.cat(outputs, dim=2).transpose(1, 2).reshape(2, 5, 64)
        torch.testing.assert_close(attention(x), attention.output(reference))

    def check_causality(self, device):
        model = TernaryTransformer(self.config).to(device).eval()
        tokens = torch.randint(0, 64, (2, 16), device=device)
        with torch.no_grad():
            baseline = model(tokens)
            for split in (1, 7, 15):
                changed = tokens.clone()
                changed[:, split:] = (changed[:, split:] + 17) % 64
                perturbed = model(changed)
                torch.testing.assert_close(baseline[:, :split], perturbed[:, :split], rtol=0, atol=0)
                torch.testing.assert_close(baseline[:, :split], model(tokens[:, :split]),
                                           rtol=1e-5, atol=1e-6)
                self.assertFalse(torch.equal(baseline[:, split:], perturbed[:, split:]))

    def test_causal_cpu(self):
        self.check_causality('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_causal_cuda(self):
        self.check_causality('cuda')

    def check_gradients(self, device):
        model = TernaryTransformer(self.config).to(device)
        tokens = torch.randint(0, 64, (2, 9), device=device)
        target = (tokens + 1) % 64
        loss = F.cross_entropy(model(tokens).flatten(0, 1), target.flatten())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, p in model.named_parameters():
            with self.subTest(parameter=name, device=device):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
                self.assertEqual(p.grad.device.type, device)
        before = model.token_embedding.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, model.token_embedding.weight))

    def test_gradients_cpu(self):
        self.check_gradients('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gradients_cuda(self):
        self.check_gradients('cuda')

    def test_parameter_counts_and_projection_types(self):
        for config in (self.config, replace(self.config, d_model=32, n_heads=2, n_layers=1, d_ff=48)):
            model = TernaryTransformer(config)
            self.assertEqual(count_parameters(model), estimate_parameters(config)['total'])
            self.assertEqual(count_parameters(model, trainable_only=True), count_parameters(model))
            self.assertEqual(sum(isinstance(m, BitLinear) for m in model.modules()), 4 * config.n_layers)
            self.assertFalse(any(p.ndim == 1 and 'bias' in name for name, p in model.named_parameters()))
        self.assertEqual(estimate_parameters(self.config)['total'], 69952)
        # A failing constructor proves the target estimator does not construct a model.
        with patch.object(TernaryTransformer, '__init__', side_effect=AssertionError('allocation forbidden')):
            target = estimate_parameters(target_300m_config())
        self.assertEqual(target['total'], 297059328)

    def test_seed_and_state_dict_reproducibility(self):
        model = TernaryTransformer(self.config).eval()
        tokens = torch.randint(0, 64, (2, 8))
        baseline = model(tokens)
        torch.manual_seed(42)
        same = TernaryTransformer(self.config).eval()
        torch.testing.assert_close(baseline, same(tokens), rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = TernaryTransformer(self.config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(baseline, restored(tokens), rtol=0, atol=0)

    def test_invalid_inputs(self):
        model = TernaryTransformer(self.config)
        for tokens in (torch.zeros(2, 33, dtype=torch.long), torch.zeros(2, 0, dtype=torch.long),
                       torch.zeros(0, 4, dtype=torch.long), torch.zeros(4, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(tokens)
        with self.assertRaises(TypeError):
            model(torch.zeros(2, 4))


if __name__ == '__main__':
    unittest.main()
