# Milestone 1 validation — 2026-11-04

## Completed

Modular src-layout package, isolated Python 3.13.7 environment, absmean ternary
weight quantization, identity STE, reusable BitLinear, unit tests, bounded GPU
experiment, and architecture/math documentation. No 300M allocation and no commits
or pushes were performed. The starting directory was empty and was not a Git repository.

## Environment limitation

`.venv` was created successfully, but CUDA PyTorch installation from the official
cu130 index failed: WinError 10051 (network unreachable). The final pip message
was "No matching distribution found" after connection failures; compatibility
was not established by that failure. No cached torch wheel was available.
The isolated environment therefore does **not** yet have PyTorch installed.

Validation used the existing user installation via `py -3.13`, not `.venv`:
Python 3.13.7, PyTorch 2.7.1+cu118, CUDA runtime 11.8, driver 617.42,
NVIDIA GeForce RTX 4060 Laptop GPU with 8,188 MiB reported memory.

Reproduce this validation using the existing installation:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
py -3.13 -m unittest discover -s tests -v
py -3.13 experiments/train_tiny.py --device cuda
py -3.13 -m compileall -q src tests experiments
```

## Results

All six unittest methods passed, including dtype/bias subtests, in 4.152 seconds;
zero skips. Checks cover known quantization values, zero tensors, FP32/FP16/BF16
preservation, identity surrogate gradients, forward shapes/values, analytical
input/weight/bias gradients, optimizer updates, evaluation behavior, invalid
inputs, and CUDA forward/backward. Python compilation also passed.

The 2,632-parameter GPU experiment completed 100 AdamW steps with seed 42:

| Measurement | Result |
| --- | ---: |
| Initial training MSE | 1.3697137833 |
| Final training MSE | 0.0484611690 |
| Elapsed seconds | 0.5473 |
| Peak PyTorch allocated bytes | 68,268,032 |
| Peak PyTorch reserved bytes | 69,206,016 |

This synthetic training-set regression demonstrates optimization and gradient
flow only; it says nothing about language quality or generalization. Memory
numbers include no comparison baseline and establish no compression benefit.
The raw run is saved to `artifacts/tiny_training.json`.

## Next steps (not performed)

1. Rerun `scripts/setup.ps1` with package-network access; validate inside `.venv`
   and retain its generated dependency lock before changing the environment.
2. Design small causal Transformer blocks and validate masking and language loss
   before considering 300M sizing or training-memory budgets.
3. Design a versioned packed ternary checkpoint format with scales, round-trip
   tests, and measured disk sizes, followed by offline inference validation.

Milestone 1 stops here. Isolated CUDA dependency installation remains incomplete;
the implementation and GPU validation succeeded using existing local software.
