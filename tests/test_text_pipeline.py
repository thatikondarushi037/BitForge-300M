import os
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.nn import functional as F

from bitforge import TernaryTransformer, tiny_config
from bitforge.data import NextTokenBatcher, evaluation_batches, load_corpus
from bitforge.tokenizer import ByteTokenizer
from bitforge.training import (check_small_config, evaluate, greedy_generate,
                               load_checkpoint, restore_training_state, save_checkpoint)


class TextPipelineTests(unittest.TestCase):
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
        self.config = replace(tiny_config(), vocab_size=259, max_seq_len=8,
                              d_model=16, n_heads=2, n_layers=1, d_ff=32)

    def test_tokenizer_round_trips_and_specials(self):
        tokenizer = ByteTokenizer()
        for text in ('', 'Hello\nworld!', 'café, 東京, 🌱', '\x00\r\n\t'):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.encode(text), list(text.encode('utf-8')))
        self.assertEqual(tokenizer.vocab_size, 259)
        self.assertEqual(tokenizer.decode([256, 65, 257, 258]), 'A')
        with self.assertRaises(ValueError):
            tokenizer.decode([256], skip_special=False)
        for ids in ([-1], [259], [1.0]):
            with self.assertRaises(ValueError):
                tokenizer.decode(ids)
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([255])
        self.assertEqual(tokenizer.decode([255], errors='replace'), '\ufffd')
        self.assertEqual(ByteTokenizer.from_metadata(tokenizer.metadata()).metadata(), tokenizer.metadata())
        with self.assertRaises(ValueError):
            ByteTokenizer.from_metadata({'type': 'unknown'})

    def test_batch_determinism_and_shift(self):
        tokens = torch.arange(100)
        a, b = (NextTokenBatcher(tokens, 8, 4, 42) for _ in range(2))
        for _ in range(3):
            x, y = a.sample()
            bx, by = b.sample()
            torch.testing.assert_close(x, bx, rtol=0, atol=0)
            torch.testing.assert_close(y, by, rtol=0, atol=0)
            self.assertEqual(x.shape, (4, 8))
            torch.testing.assert_close(y, x + 1)
        different, _ = NextTokenBatcher(tokens, 8, 4, 43).sample()
        self.assertFalse(torch.equal(x, different))
        state = a.generator.get_state()
        expected = a.sample()
        a.generator.set_state(state)
        torch.testing.assert_close(a.sample()[0], expected[0], rtol=0, atol=0)
        # The single valid window includes the last stream token as its target.
        x, y = NextTokenBatcher(torch.arange(9), 8, 1, 1).sample()
        self.assertEqual(y[0, -1].item(), 8)

    def test_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_bytes(('A' * 80 + '\n\n' + 'B' * 20).encode())
            train, validation, metadata = load_corpus(path, 0.2)
            self.assertEqual(bytes(train.tolist()) + bytes(validation.tolist()), path.read_bytes())
            self.assertEqual(metadata['split_byte_offset'], len(train))
            self.assertEqual(set(validation.tolist()), {ord('B')})
            self.assertNotIn(ord('B'), train.tolist())
            for stream in (train, validation):
                x, y = NextTokenBatcher(stream, 8, 20, 42).sample()
                self.assertTrue(set(x.flatten().tolist()) <= set(stream.tolist()))
                self.assertTrue(set(y.flatten().tolist()) <= set(stream.tolist()))

    def test_bundled_corpus_manifest(self):
        path = Path(__file__).resolve().parents[1] / 'data/dev_corpus.txt'
        train, validation, metadata = load_corpus(path)
        self.assertEqual(len(train) + len(validation), metadata['bytes'])
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / path.name
            copy.write_bytes(path.read_bytes() + b'changed')
            copy.with_suffix('.manifest.json').write_bytes(path.with_suffix('.manifest.json').read_bytes())
            with self.assertRaises(ValueError):
                load_corpus(copy)

    def test_weighted_evaluation_and_tail(self):
        model = TernaryTransformer(self.config)
        tokens = torch.arange(23)
        batches = list(evaluation_batches(tokens, 8, 2))
        targets = torch.cat([y.flatten() for _, y in batches])
        torch.testing.assert_close(targets, tokens[1:])
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='sum').item()
        result = evaluate(model, tokens, 8, 2)
        self.assertEqual(result['tokens'], 22)
        self.assertAlmostEqual(result['loss'], total / 22, places=6)
        self.assertAlmostEqual(result['perplexity'], math.exp(result['loss']), places=6)
        self.assertTrue(model.training)
        self.assertAlmostEqual(result['loss'], evaluate(model, tokens, 8, 1)['loss'], places=5)

    def check_checkpoint(self, device):
        model = TernaryTransformer(self.config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        batcher = NextTokenBatcher(torch.arange(100), 8, 2, 42)

        def step(m, opt, data):
            x, y = data.sample(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer, batcher)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'local.pt'
            save_checkpoint(path, model, optimizer, batcher, {'step': 1})
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, batcher, {'step': 1})
            expected_loss = step(model, optimizer, batcher)
            restored, payload = load_checkpoint(path, device)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
            restored_batcher = NextTokenBatcher(torch.arange(100), 8, 2, 99)
            restore_training_state(payload, restored_optimizer, restored_batcher, device)
            self.assertEqual(payload['progress']['step'], 1)
            self.assertEqual(restored.config, model.config)
            torch.testing.assert_close(torch.get_rng_state(), payload['cpu_rng'], rtol=0, atol=0)
            loss = step(restored, restored_optimizer, restored_batcher)
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            for actual, expected in zip(restored.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(restored_optimizer.state.values(), optimizer.state.values()):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_checkpoint_resume_cpu(self):
        self.check_checkpoint('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_checkpoint_resume_cuda(self):
        self.check_checkpoint('cuda')

    def test_restricted_checkpoint_loading(self):
        with patch('bitforge.training.torch.load', side_effect=RuntimeError('rejected')) as loader:
            with self.assertRaises(RuntimeError):
                load_checkpoint(Path('untrusted.pt'))
            loader.assert_called_once_with(Path('untrusted.pt'), map_location='cpu', weights_only=True)
        with patch('bitforge.training.torch.load', return_value={'format_version': 999}):
            with self.assertRaises(ValueError):
                load_checkpoint(Path('bad.pt'))
        with self.assertRaises(ValueError):
            check_small_config(replace(self.config, n_layers=10000))

    def test_generation_shape_and_byte_mask(self):
        model = TernaryTransformer(self.config)
        prompt = torch.tensor([[65] * 12])
        def fake_logits(x):
            logits = torch.zeros(*x.shape, 259)
            logits[..., 258] = 100
            logits[..., 66] = 1
            self.assertLessEqual(x.size(1), self.config.max_seq_len)
            return logits
        with patch.object(model, 'forward', side_effect=fake_logits):
            output = greedy_generate(model, prompt, 5)
        self.assertEqual(output.shape, (1, 17))
        torch.testing.assert_close(output[:, :12], prompt)
        torch.testing.assert_close(output[:, 12:], torch.full((1, 5), 66))
        self.assertTrue(model.training)
        torch.testing.assert_close(greedy_generate(model, prompt, 0), prompt)
        real = greedy_generate(model, prompt, 3)
        self.assertEqual(real.shape, (1, 15))
        self.assertTrue((real < 256).all())

    def test_invalid_batches_and_generation(self):
        for length, batch in ((0, 1), (4, 0), (10, 1)):
            with self.assertRaises(ValueError):
                NextTokenBatcher(torch.arange(10), length, batch, 42)
        with self.assertRaises(ValueError):
            greedy_generate(TernaryTransformer(self.config), torch.empty(1, 0, dtype=torch.long), 2)


if __name__ == '__main__':
    unittest.main()
