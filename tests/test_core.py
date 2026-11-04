import unittest
import torch
from torch.nn import functional as F
from bitforge import BitLinear, ternary_codes, ternary_quantize

class CoreTests(unittest.TestCase):
    def test_codes(self):
        w = torch.tensor([-2., -.1, 0., .1, 2.])
        codes, scale = ternary_codes(w)
        torch.testing.assert_close(scale, torch.tensor(.84))
        torch.testing.assert_close(codes, torch.tensor([-1., 0., 0., 0., 1.]))
        torch.testing.assert_close(ternary_quantize(w), codes * scale)

    def test_zero_dtype_and_gradient(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                w = torch.zeros(3, 4, dtype=dtype, requires_grad=True)
                q = ternary_quantize(w)
                self.assertEqual(q.dtype, dtype)
                self.assertTrue(torch.isfinite(q).all())
                self.assertEqual(torch.count_nonzero(q), 0)
                q.sum().backward()
                torch.testing.assert_close(w.grad, torch.ones_like(w))

    def test_identity_ste(self):
        w = torch.tensor([-100., -.2, 0., .2, 100.], requires_grad=True)
        upstream = torch.arange(1., 6.)
        ternary_quantize(w).backward(upstream)
        torch.testing.assert_close(w.grad, upstream)

    def test_layer(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                torch.manual_seed(7)
                layer = BitLinear(4, 3, bias=bias)
                x = torch.randn(2, 5, 4, requires_grad=True)
                y = layer(x)
                self.assertEqual(y.shape, (2, 5, 3))
                torch.testing.assert_close(y, F.linear(x, ternary_quantize(layer.weight), layer.bias))
                upstream = torch.randn_like(y)
                y.backward(upstream)
                torch.testing.assert_close(layer.weight.grad, upstream.flatten(0, 1).T @ x.detach().flatten(0, 1))
                torch.testing.assert_close(x.grad, upstream @ ternary_quantize(layer.weight).detach())
                if bias:
                    torch.testing.assert_close(layer.bias.grad, upstream.sum((0, 1)))
                before = layer.weight.detach().clone()
                torch.optim.SGD(layer.parameters(), lr=.01).step()
                self.assertFalse(torch.equal(before, layer.weight))
                layer.eval()
                torch.testing.assert_close(layer(x.detach()), F.linear(x.detach(), ternary_quantize(layer.weight), layer.bias))

    def test_invalid(self):
        with self.assertRaises(TypeError):
            ternary_quantize(torch.tensor([1]))
        with self.assertRaises(ValueError):
            ternary_quantize(torch.empty(0))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        layer = BitLinear(16, 8).cuda()
        x = torch.randn(4, 16, device='cuda', requires_grad=True)
        layer(x).square().mean().backward()
        self.assertTrue(layer.weight.grad.is_cuda)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())

if __name__ == '__main__':
    unittest.main()
