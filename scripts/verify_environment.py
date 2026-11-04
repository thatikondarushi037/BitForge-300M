"""Verify the isolated Windows CUDA environment without training a model."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    root = Path(__file__).resolve().parents[1]
    expected = root / '.venv' / 'Scripts' / 'python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f'Run explicitly with {expected}')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('An isolated virtual environment is required')

    import torch
    import bitforge

    if not Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('PyTorch must be installed inside .venv')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    gpu = torch.cuda.get_device_name(0)
    if 'RTX 4060' not in gpu:
        raise RuntimeError(f'Expected RTX 4060, detected {gpu}')
    layer = bitforge.BitLinear(16, 8, device='cuda')
    x = torch.randn(4, 16, device='cuda', requires_grad=True)
    y = layer(x)
    y.square().mean().backward()
    for grad in (x.grad, layer.weight.grad, layer.bias.grad):
        if grad is None or not grad.is_cuda or not torch.isfinite(grad).all():
            raise RuntimeError('CUDA backward propagation failed')
    torch.cuda.synchronize()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(root / 'tests'))
    )
    if not result.wasSuccessful() or result.skipped or result.testsRun < 6:
        raise RuntimeError('Expected the full discovered suite, including six core tests, with zero skips')
    report = {
        'python_executable': sys.executable,
        'python': sys.version,
        'isolated': True,
        'torch_path': torch.__file__,
        'project_path': bitforge.__file__,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': gpu,
        'cuda_forward_shape': list(y.shape),
        'cuda_backward_finite': True,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skips': len(result.skipped),
        'dependencies': dict(sorted(
            (d.metadata['Name'], d.version)
            for d in importlib.metadata.distributions()
        )),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
