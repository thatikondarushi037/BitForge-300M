"""Opt-in real 297M integration tests; run separately from tests/ discovery.

    python -m unittest discover -s tests/expensive -v

These construct/execute real tensors and export a temporary ~197 MB checkpoint.
They never train. A memory preflight is required; CUDA absence is a skip.
"""
import gc
from pathlib import Path
import tempfile
import unittest

import torch
from bitforge.config import target_300m_config
from bitforge.execution import construct_model, configure_execution
from bitforge.inference import inference_preflight, generate
from bitforge.packed import export_packed, load_packed
from bitforge.transformer import count_parameters


class FullModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(301)
        cls.config=target_300m_config()
        inference_preflight(cls.config,'cpu',8)
        cls.model=configure_execution(construct_model(cls.config),sdpa=True).eval().requires_grad_(False)
        cls.tokens=torch.arange(8)[None,:]
        with torch.inference_mode():cls.reference=cls.model(cls.tokens)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'model'):del cls.model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    def test_1_real_construction_cpu_forward(self):
        self.assertEqual(count_parameters(self.model),297_059_328)
        self.assertTrue(all(not p.is_meta for p in self.model.parameters()))
        self.assertEqual(list(self.reference.shape),[1,8,32000])
        self.assertTrue(torch.isfinite(self.reference).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_2_real_cuda_forward_generation(self):
        inference_preflight(self.config,'cuda',8)
        self.model.to('cuda')
        try:
            with torch.inference_mode():output=self.model(self.tokens.cuda())
            self.assertEqual(list(output.shape),[1,8,32000])
            self.assertTrue(torch.isfinite(output).all())
            a=generate(self.model,'The ',max_new_tokens=2,seed=42)
            b=generate(self.model,'The ',max_new_tokens=2,seed=42)
            self.assertEqual(a['token_ids'],b['token_ids'])
        finally:
            self.model.cpu()
            torch.cuda.empty_cache()

    def test_3_real_packed_export_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.bft'
            info=export_packed(path,self.model)
            self.assertEqual(info['total_parameters'],297_059_328)
            self.assertEqual(path.stat().st_size,info['packed_bytes'])
            del type(self).model
            gc.collect()
            restored,header=load_packed(path)
            configure_execution(restored,sdpa=True)
            self.assertEqual(count_parameters(restored),297_059_328)
            self.assertEqual(header['config']['n_layers'],24)
            with torch.inference_mode():actual=restored(self.tokens)
            torch.testing.assert_close(actual,self.reference,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
