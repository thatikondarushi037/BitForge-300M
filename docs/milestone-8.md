# Milestone 8 - memory-efficient execution

Changed/new files across Phase 1: `src/bitforge/{config,layers,quantization,transformer,packed,execution}.py`; `tests/test_advanced.py`; `scripts/prepare_engineering_corpus.py`; `experiments/{phase1_advanced,benchmark_advanced_loading}.py`; `data/engineering_corpus.{txt,manifest.json}`; `README.md`; `docs/milestone-{6,7,8}.md`; and measured JSON artifacts under `artifacts/phase1-final/`. Generated checkpoint binaries are local ignored artifacts.

The final runner additionally checks the execution benchmark's batch-8 preflight before allocating its model. That guard was added after the reported run; a separate successful post-run check is recorded in `execution-preflight-verification.json`. It does not retroactively change the measurement history. Initial batch-4 training was preflighted before execution.

Execution controls add optional BF16 CUDA autocast, non-reentrant per-block activation checkpointing, and PyTorch causal scaled-dot-product attention (SDPA). FP32 master weights and RMSNorm statistics remain FP32. BF16 requires a supporting GPU; FP16/scaler execution is not implemented. SDPA chooses a supported backend; no specific FlashAttention kernel is claimed.

Measured RTX 4060 Laptop CUDA results on the 1.378M improved-QAT model, batch 8, context 64, seed 77: one warm-up update followed by 10 timed updates. Peaks are reset after warm-up and include live optimizer state. Each variant starts with matched initialization/batches. Reserved memory includes allocator caching.

| Precision | SDPA | Checkpoint | Step ms | Peak allocated MiB | Peak reserved MiB |
|---|---|---|---:|---:|---:|
| fp32 | False | False | 51.94 | 120.77 | 138.00 |
| fp32 | True | False | 50.48 | 108.04 | 120.00 |
| bf16 | True | False | 55.12 | 97.43 | 100.00 |
| bf16 | True | True | 95.46 | 91.50 | 94.00 |

Packed improved model: 540,722 bytes versus 5,519,475 dense master bytes, **10.2076x** compression. Dense reconstructed file: 5,521,015 bytes. Maximum CUDA logit reconstruction error: 9.53674316e-07; checked with rtol 1e-5, atol 2e-6. CPU unit-test logits reconstruct bitwise. Version 1 remains readable; version 2 explicitly identifies the improved quantizer. Both store two-bit ternary codes and FP32 scales, with unquantized embedding/norm tensors and integrity checks.

The loader validates bounded dimensions and parameter counts before model construction. Limits now permit the 297M target (350M total, 16M per packed tensor, 768 MiB file); full-size export/load has not been executed. Tests cover a 5,000,001-element packed tensor, target configuration acceptance, malformed metadata and historical checkpoints. Meta construction plus state assignment avoids an additional initialized dense parameter copy. The container and decoded weights are still held in host RAM; streaming/mmap loading is not implemented.

Isolated loading benchmark, local warm filesystem, one load per path; inference has 5 warm-ups and 100 timed forwards at batch 1/context 64:

| Path | Load ms | Inference ms | Load peak allocated MiB | Inference allocated/reserved MiB |
|---|---:|---:|---:|---:|
| dense reconstructed | 41.16 | 5.861 | 5.32 | 13.96 / 26.00 |
| packed | 2558.04 | 5.355 | 13.44 | 13.96 / 26.00 |

Packed loading is slower here because of checksum validation and unpacking. Both paths reconstruct dense FP32 weights and use floating-point GEMM; inference memory is identical. Small latency differences are timing noise, not evidence of ternary acceleration. Native low-bit execution would require a verified custom kernel; none was added. PyTorch's floating-point SDPA/autocast optimizations do not supply a ternary matrix multiplication kernel.

Validation: **57 tests passed**, including all 49 baseline tests and 8 advanced tests: quantization MSE/STE, meta counts/budget, checkpoint/SDPA outputs and gradients, improved packed reconstruction, larger tensor packing, BF16 device rejection, corpus split isolation, and CUDA BF16 checkpoint backward. Full pretraining, 300M CUDA execution and full-size packed benchmarks remain unperformed due to the conservative hardware budget. Total intended experiment training: 900 QAT updates plus 44 execution-benchmark updates; interrupted preliminary attempts are not included in reported final metrics. No additional dependencies were introduced.

Reproduce from the repository root (existing Python 3.13 / PyTorch 2.7.1+cu118 environment):
```powershell
python scripts/prepare_engineering_corpus.py
python -m unittest discover -s tests -v
python scripts/estimate_architecture.py
python experiments/phase1_advanced.py --output-dir artifacts/phase1-reproduction --steps 100
python experiments/benchmark_advanced_loading.py artifacts/phase1-reproduction
```
Use a new output directory for each run; files are created exclusively. No downloads occur. CUDA and BF16 support are required for the experiment. Results: `artifacts/phase1-final/results.json` and `loading.json`. Baseline `3a1e760` and Milestones 1-5 artifacts were preserved. No commit or push was performed.
