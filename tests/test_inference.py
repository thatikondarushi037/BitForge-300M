import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import torch
from bitforge.__main__ import main
from bitforge.config import TransformerConfig
from bitforge.execution import construct_model, preflight
from bitforge.inference import (load_local_checkpoint, save_inference_checkpoint,
                               generate, sample_token, benchmark_model, freeze_quantized_projections)
from bitforge.packed import export_packed
from bitforge.resources import cpu_memory, disk_preflight, host_preflight
from bitforge.training import save_checkpoint
from bitforge.data import NextTokenBatcher


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.directory=Path(self.tmp.name)
        self.config=TransformerConfig(vocab_size=259,max_seq_len=16,d_model=16,n_heads=2,n_layers=1,d_ff=32)
        torch.manual_seed(10)
        self.model=construct_model(self.config).eval().requires_grad_(False)
        self.dense=self.directory/'m.pt';self.packed=self.directory/'m.bft'
        save_inference_checkpoint(self.dense,self.model,training='untrained test fixture')
        export_packed(self.packed,self.model)

    def tearDown(self):self.tmp.cleanup()

    def call(self,*args):
        out=io.StringIO();err=io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            code=main(list(args))
        return code,out.getvalue(),err.getvalue()

    def test_dense_packed_numerical_roundtrip(self):
        x=torch.arange(8)[None,:]
        for path in (self.dense,self.packed):
            model,info=load_local_checkpoint(path)
            self.assertEqual(info['parameters'],sum(p.numel() for p in self.model.parameters()))
            with torch.inference_mode():torch.testing.assert_close(self.model(x),model(x),rtol=1e-5,atol=1e-6)

    def test_legacy_training_checkpoint(self):
        model=construct_model(self.config)
        opt=torch.optim.AdamW(model.parameters())
        batcher=NextTokenBatcher(torch.arange(64)%259,8,1,42)
        path=self.directory/'legacy.pt'
        save_checkpoint(path,model,opt,batcher,{'step':0})
        restored,info=load_local_checkpoint(path)
        self.assertEqual(info['training'],{'step':0})
        self.assertEqual(restored.config,self.config)

    def test_seeded_sampling_and_filters(self):
        a=generate(self.model,'Test',max_new_tokens=8,temperature=.8,top_k=12,top_p=.8,seed=3)
        b=generate(self.model,'Test',max_new_tokens=8,temperature=.8,top_k=12,top_p=.8,seed=3)
        self.assertEqual(a['token_ids'],b['token_ids'])
        logits=torch.tensor([[1.,2.,3.]])
        for seed in range(5):
            g=torch.Generator().manual_seed(seed)
            self.assertEqual(sample_token(logits,temperature=1,top_k=1,generator=g).item(),2)
            self.assertEqual(sample_token(logits,temperature=1,top_p=.01,generator=g).item(),2)
        self.assertEqual(sample_token(logits).item(),2)

    def test_invalid_sampling(self):
        for kw in ({'temperature':-1},{'temperature':float('nan')},{'top_k':-1},{'top_p':0},{'top_p':1.1},{'max_new_tokens':513}):
            with self.assertRaises(ValueError):generate(self.model,'X',**kw)
        with self.assertRaises(ValueError):generate(self.model,'')

    def test_cli_all_commands_offline(self):
        with patch.object(socket,'socket',side_effect=AssertionError('network forbidden')):
            for cmd in ('model-info','generate','demo','benchmark','verify-checkpoint'):
                extra=['--max-new-tokens','2'] if cmd in ('generate','demo') else []
                if cmd=='benchmark':extra=['--warmup','0','--repeats','1','--generation-tokens','1']
                code,out,err=self.call(cmd,'--checkpoint',str(self.packed),'--json',*extra)
                self.assertEqual(code,0,err)
                self.assertIsInstance(json.loads(out),dict)

    def test_cli_reference_verification(self):
        code,out,err=self.call('verify-checkpoint','--checkpoint',str(self.packed),'--reference',str(self.dense))
        self.assertEqual(code,0,err)
        self.assertTrue(json.loads(out)['verified'])

    def test_cli_missing_and_corrupted(self):
        for path in (self.directory/'absent',self.directory/'bad'):
            if path.name=='bad':path.write_bytes(b'not a checkpoint')
            code,out,err=self.call('model-info','--checkpoint',str(path))
            self.assertEqual(code,2);self.assertIn('bitforge:',err)
        raw=bytearray(self.packed.read_bytes());raw[-1]^=1;self.packed.write_bytes(raw)
        code,out,err=self.call('verify-checkpoint','--checkpoint',str(self.packed))
        self.assertEqual(code,2);self.assertIn('checksum',err)

    def test_cli_readable_and_options(self):
        code,out,err=self.call('generate','--checkpoint',str(self.dense),'--max-new-tokens','1','--verbose')
        self.assertEqual(code,0,err);self.assertIn('parameters',out)
        code,out,err=self.call('model-info','--checkpoint',str(self.dense),'--threads','0')
        self.assertEqual(code,2)
        code,out,err=self.call('verify-checkpoint','--checkpoint',str(self.dense),'--sequence-length','17')
        self.assertEqual(code,2);self.assertIn('invalid verification context',err)

    def test_bad_dense_shapes_and_unsupported_format(self):
        payload=torch.load(self.dense,weights_only=True)
        payload['model']['final_norm.weight']=torch.zeros(2)
        bad=self.directory/'badshape.pt';torch.save(payload,bad)
        with self.assertRaises(ValueError):load_local_checkpoint(bad)
        payload['model']['final_norm.weight']='invalid'
        other=self.directory/'badtype.pt';torch.save(payload,other)
        with self.assertRaises(ValueError):load_local_checkpoint(other)
        raw=self.directory/'raw.pt';torch.save(self.model.state_dict(),raw)
        with self.assertRaises(ValueError):load_local_checkpoint(raw)

    def test_benchmark_schema(self):
        result=benchmark_model(self.model,sequence_length=8,repeats=2,warmup=1,generation_tokens=2)
        self.assertEqual(result['output_shape'],[1,8,259])
        self.assertEqual(len(result['latency_seconds']),2)
        self.assertGreater(result['median_latency_seconds'],0)
        self.assertGreater(result['generation']['tokens_per_second'],0)
        self.assertIsNone(result['forward_cuda_peak'])

    def test_resource_guards(self):
        self.assertGreater(cpu_memory()['rss_bytes'],0)
        self.assertGreater(host_preflight(1)['available_bytes'],0)
        with self.assertRaises(RuntimeError):host_preflight(1<<60)
        with self.assertRaises(RuntimeError):disk_preflight(self.directory,1<<60)
        with self.assertRaises(FileExistsError):save_inference_checkpoint(self.dense,self.model,training='test')

    def test_metadata_str_subclass_is_portable(self):
        from torch.torch_version import TorchVersion
        path=self.directory/'metadata.pt'
        save_inference_checkpoint(path,self.model,training={'torch':TorchVersion('2.7.1')})
        payload=torch.load(path,weights_only=True)
        self.assertIs(type(payload['training']['torch']),str)
        model=construct_model(self.config)
        opt=torch.optim.AdamW(model.parameters())
        batcher=NextTokenBatcher(torch.arange(64)%259,8,1,42)
        legacy=self.directory/'progress.pt'
        save_checkpoint(legacy,model,opt,batcher,{'torch':TorchVersion('2.7.1')})
        self.assertIs(type(torch.load(legacy,weights_only=True)['progress']['torch']),str)

    def test_frozen_reference_keeps_quantized_logits(self):
        x=torch.arange(8)[None,:]
        with torch.inference_mode():expected=self.model(x)
        freeze_quantized_projections(self.model)
        self.assertEqual(self.model.config.projection_backend,'fp32')
        with torch.inference_mode():torch.testing.assert_close(self.model(x),expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_loading_sampling_and_benchmark(self):
        for path in (self.dense,self.packed):
            model,_=load_local_checkpoint(path,'cuda')
            a=generate(model,'Test',max_new_tokens=3,temperature=.9,seed=3)
            b=generate(model,'Test',max_new_tokens=3,temperature=.9,seed=3)
            self.assertEqual(a['token_ids'],b['token_ids'])
            result=benchmark_model(model,repeats=2,precision='bf16',generation_tokens=1)
            self.assertGreater(result['forward_cuda_peak']['allocated_bytes'],0)
        check=preflight(self.config,sequence_length=8)
        self.assertTrue(check['inference_allowed'])


if __name__=='__main__':unittest.main()
