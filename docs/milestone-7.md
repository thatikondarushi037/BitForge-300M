# Milestone 7 - approximately 300M architecture

The configurable BitNet-inspired decoder has **297,059,328 parameters**, verified both arithmetically and by constructing the entire module tree on the meta device. Settings: vocabulary 32,000, width 1,024, 24 blocks, 16 heads, FFN 3,328, context 2,048. It uses pre-RMSNorm residual blocks, ternary QAT projections, GELU, fixed sinusoidal positions and tied floating-point embedding/output weights. This is an adaptation of the existing architecture, not a reproduction of a published BitNet architecture. The 32K vocabulary is a configuration target; no 32K tokenizer was trained (current experiments use bytes).

Parameter breakdown: embedding/head 32,768,000; attention 100,663,296; FFN 163,577,856; RMSNorm 50,176. Configuration validation checks positive dimensions, head divisibility, even position width, backend and epsilon. Meta construction avoids allocating parameter storage; direct destination construction avoids a duplicate host model.

Memory planning explicitly reports estimates, not measured peaks. FP32 weights alone: 1,188,237,312 bytes. FP32 AdamW parameters, gradients and moments: 16 bytes/parameter, plus conservative eager attention/activation estimates, quantization workspace, logits and 1 GiB reserve.

| Context, batch 1 | Estimated training GiB | Estimated inference GiB | Training permitted |
|---|---:|---:|---|
| 2048 | 22.678 | 19.359 | False |
| 64 | 5.676 | 2.356 | False |

The runtime preflight allowed at most 70% of currently free CUDA memory (5,215,826,739 bytes in this run), below both training estimates. No 300M parameters were allocated on CUDA; no full-model forward/backward or pretraining was attempted. The successful 1.378M fallback preflight led to the bounded training documented in Milestone 6. Memory estimates deliberately do not assume BF16/checkpoint savings; memory depends on context, batch, kernels and allocator. Architecture verification is complete; meaningful 300M language training is not. No language ability is claimed for this untrained architecture.

Reproduce from the repository root (existing Python 3.13 / PyTorch 2.7.1+cu118 environment):
```powershell
python scripts/prepare_engineering_corpus.py
python -m unittest discover -s tests -v
python scripts/estimate_architecture.py
python experiments/phase1_advanced.py --output-dir artifacts/phase1-reproduction --steps 100
python experiments/benchmark_advanced_loading.py artifacts/phase1-reproduction
```
Use a new output directory for each run; files are created exclusively. No downloads occur. CUDA and BF16 support are required for the experiment. Results: `artifacts/phase1-final/results.json` and `loading.json`. Baseline `3a1e760` and Milestones 1-5 artifacts were preserved. No commit or push was performed.
