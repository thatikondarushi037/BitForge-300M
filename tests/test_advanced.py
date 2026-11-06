import tempfile
from pathlib import Path
from dataclasses import replace
import unittest
import torch
from bitforge.config import TransformerConfig, target_300m_config, estimate_parameters
from bitforge.execution import construct_model,configure_execution,memory_estimate,autocast_context
from bitforge.quantization import ternary_quantize,least_squares_quantize,least_squares_codes
from bitforge.packed import export_packed,load_packed,_check_packed_config
from bitforge.transformer import count_parameters

class AdvancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)
    def config(self):
        return TransformerConfig(vocab_size=259,max_seq_len=8,d_model=16,n_heads=2,n_layers=2,d_ff=32,projection_backend='ternary_ls')
    def test_ls_mse_and_ste(self):
        for seed in range(5):
            torch.manual_seed(seed);w=torch.randn(31,47,requires_grad=True)
            improved=least_squares_quantize(w)
            self.assertLessEqual((w-improved).square().mean().item(),(w-ternary_quantize(w)).square().mean().item()+1e-7)
            improved.sum().backward();torch.testing.assert_close(w.grad,torch.ones_like(w))
        codes,scale=least_squares_codes(torch.zeros(5));self.assertTrue(torch.isfinite(scale));self.assertEqual(codes.count_nonzero(),0)
    def test_meta_target_and_budget(self):
        c=target_300m_config();m=construct_model(c,'meta')
        self.assertEqual(count_parameters(m),estimate_parameters(c)['total'])
        self.assertTrue(290_000_000<count_parameters(m)<310_000_000)
        self.assertTrue(all(p.is_meta for p in m.parameters()))
        self.assertGreater(memory_estimate(c)['training_budget_bytes'],8<<30)
        _check_packed_config(c)
    def test_checkpoint_and_sdpa_outputs_gradients(self):
        torch.manual_seed(1);c=self.config();base=construct_model(c)
        optimized=configure_execution(construct_model(c),checkpointing=True,sdpa=True)
        optimized.load_state_dict(base.state_dict());x=torch.randint(259,(2,8))
        a=base(x);b=optimized(x);torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().mean().backward();b.square().mean().backward()
        for p,q in zip(base.parameters(),optimized.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=1e-6)
    def test_improved_packed(self):
        c=self.config();m=construct_model(c).eval();x=torch.randint(259,(2,8))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.bft';export_packed(p,m);restored,header=load_packed(p)
            self.assertEqual(header['version'],2)
            torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
    def test_large_tensor_pack(self):
        from bitforge.packed import pack_ternary,unpack_ternary
        values=(torch.arange(5_000_001)%3-1).float()
        torch.testing.assert_close(unpack_ternary(pack_ternary(values),values.numel()),values,rtol=0,atol=0)
    def test_bf16_rejected_on_cpu(self):
        with self.assertRaises(ValueError):autocast_context('cpu','bf16')
    def test_engineering_corpus_isolation(self):
        from bitforge.data import load_corpus
        import hashlib
        import json
        path=Path(__file__).resolve().parents[1]/'data/engineering_corpus.txt'
        raw=path.read_bytes()
        manifest=json.loads(path.with_suffix('.manifest.json').read_text())
        self.assertEqual(manifest['license'],'CC0-1.0')
        self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest['sha256'])
        train,val,metadata=load_corpus(path)
        cut=metadata['split_byte_offset']
        self.assertTrue(raw[:cut].endswith(b'\n\n'))
        self.assertEqual(len(train)+len(val),len(raw))
        a=set(raw[:cut].decode().strip().split('\n\n'))
        b=set(raw[cut:].decode().strip().split('\n\n'))
        self.assertFalse(a & b)
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_bf16_checkpoint_backward(self):
        m=configure_execution(construct_model(self.config(),'cuda'),checkpointing=True,sdpa=True)
        with autocast_context('cuda','bf16'):
            loss=m(torch.randint(259,(2,8),device='cuda')).float().square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))

if __name__=='__main__':unittest.main()
