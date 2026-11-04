# BitForge-300M

Memory-efficient, BitNet-inspired Transformer research. The eventual target is
300 million parameters. Milestone 1 provides quantization foundations;
**Milestone 2 adds a modular, tiny decoder-only Transformer**.
Milestone 3 adds an offline byte tokenizer, local corpus, checkpointed text
training, validation perplexity, and greedy generation.
Milestone 4 compares matched FP32 and ternary projection backends over three
paired seeds, with separate training/inference measurements.
Milestone 5 adds a versioned two-bit checkpoint format with strict integrity
checks and frozen dense reconstruction for offline inference.
No 300M model is constructed, trained, or downloaded.

## Layout

```text
src/bitforge/quantization.py  absmean quantization and identity STE
src/bitforge/layers.py        reusable BitLinear
src/bitforge/config.py        validated configs and parameter arithmetic
src/bitforge/transformer.py   RMSNorm, positions, attention, FFN, decoder
src/bitforge/tokenizer.py     versioned UTF-8 byte tokenizer (259 IDs)
src/bitforge/data.py          isolated splits and deterministic next-token batches
src/bitforge/training.py      evaluation, local checkpoints, greedy generation
src/bitforge/benchmark.py     pairing audits, storage arithmetic, statistics
src/bitforge/packed.py        two-bit export/import, strict metadata and checksums
tests/test_core.py           CPU and CUDA unit tests (standard-library unittest)
tests/test_transformer.py    architecture, causality, gradient, count tests
tests/test_text_pipeline.py  tokenizer, data, checkpoint, generation tests
tests/test_projection_backends.py matched initialization and backend tests
tests/test_packed.py          packing, reconstruction, corruption rejection
experiments/train_tiny.py    offline synthetic teacher regression
experiments/train_next_token.py bounded synthetic next-token smoke test
experiments/train_text.py    configurable offline text-training and resume CLI
experiments/benchmark_projections.py fresh-process paired CUDA benchmark
experiments/benchmark_packed.py actual sizes, equivalence, loading/inference measurements
scripts/generate_text.py     offline greedy text generation from a checkpoint
scripts/export_packed.py     local ternary checkpoint to versioned .bft file
scripts/generate_packed.py   offline packed import and dense FP32 generation
data/                       original CC0 prose, provenance, and SHA-256 manifest
scripts/estimate_architecture.py allocation-free tiny and 300M estimates
scripts/setup.ps1           isolated Windows CUDA setup
scripts/verify_environment.py isolated CUDA and full-suite verification
requirements-lock.txt       resolved Windows/Python 3.13 CUDA dependencies
docs/environment.md         environment stabilization report
docs/milestone-1.md          validation and environment limitations
docs/milestone-2.md          architecture, counts, measured validation
docs/milestone-3.md          text pipeline, checkpoints, measured real-text run
docs/milestone-4.md          controlled FP32/ternary comparison and limitations
docs/milestone-4-results.json complete machine-readable benchmark snapshot
docs/milestone-5.md          format specification and measured real compression
docs/milestone-5-results.json packed-checkpoint measurements
```

## Setup and execution (PowerShell)

Python 3.13 is the reference interpreter. From this directory, with network access:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe scripts\verify_environment.py
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\.venv\Scripts\python.exe experiments/train_next_token.py --device cuda --steps 100 --seed 42 --output "artifacts/next-token-$runTag.json"
.\.venv\Scripts\python.exe scripts/estimate_architecture.py
```

Activation is optional. Setup installs the checked-in dependency snapshot,
including PyTorch 2.7.1+cu118 from the official PyTorch cu118 index, installs the
project, and verifies CUDA plus all discovered tests. Python is constrained to 3.13;
other Python/PyTorch minor versions require a separate validation pass.
The dependency snapshot targets Windows x64 and is version-pinned, without a
complete set of package hashes. Setup never rewrites it or uses global PyTorch.
For offline setup with the recovered local wheels, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Wheelhouse .venv\wheelhouse
```

The wheelhouse is local and ignored by Git; a fresh machine must download the
locked packages first. See [the environment report](docs/environment.md) for
exact network installation commands and cache provenance.
See [PyTorch installation guidance](https://pytorch.org/get-started/locally/).
The NVIDIA driver must support the selected runtime; a separate CUDA toolkit is
not needed for wheel-based execution. Offline replay requires pre-downloaded
compatible wheels and dependencies. The experiment itself uses no network or
external datasets. `--device cpu` explicitly selects CPU; CUDA errors never
silently fall back to CPU.

## Controlled FP32 versus ternary comparison

`TransformerConfig(projection_backend='fp32')` selects ordinary `nn.Linear`
attention/FFN projections; the default `'ternary'` preserves BitLinear. All
other architecture settings and parameter layouts are shared. Old checkpoint
configs without this field still load as ternary. The text-training CLI also
accepts `--projection-backend fp32` and prevents changing backend on resume.

```powershell
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\.venv\Scripts\python.exe experiments\benchmark_projections.py --output-dir "artifacts/paired-$runTag" --seeds 42 43 44 --steps 300
```

See [Milestone 4](docs/milestone-4.md) for exact seed-level results and timing
scope. Both backends retain FP32 master weights and dense execution; measured
master-checkpoint sizes are identical in that benchmark. This tiny-corpus
comparison is non-generalizable.

Milestone 5 separately packs the trained ternary weights: **87,324 bytes**
versus **335,225 bytes** for the same model's original weights-only file.
The importer reconstructs frozen FP32 Linear weights and bypasses repeated
quantization. This is on-disk compression, with no packed GPU execution or
measured VRAM reduction against the equivalent dense frozen control. See
[Milestone 5](docs/milestone-5.md) for exact commands, safety limits, and timings.

## Mathematical foundations

For a floating-point master weight tensor W, use one scale for the entire tensor:

```text
s = max(mean(abs(W)), machine_epsilon)
T = clamp(round(W / s), -1, +1)
Q = s * T
Y = X Q^T + b
```

T contains only -1, 0, +1. Round uses PyTorch's ties-to-even convention.
Zero tensors map to zero. FP16/BF16 statistics are accumulated in FP32.
Inputs must be finite, nonempty floating-point tensors.

Rounding has an unusable derivative for training. The straight-through estimator
uses Q in the forward pass and the surrogate derivative dQ/dW = 1 in backward,
including saturated weights. The scale is detached. This is a deliberate biased
gradient approximation, not the derivative of the discrete function. The optimizer
updates floating-point master weights; each forward pass quantizes them again.

`BitLinear` follows `nn.Linear`, supporting arbitrary leading input dimensions,
optional bias, device/dtype placement, and ordinary state dictionaries. Bias and
activations stay floating point. There is no activation quantization or internal
normalization. Quantization runs in both train and eval modes. This is a
BitNet-inspired primitive, not a reproduction of the complete BitNet architecture.

The Transformer composes these projections into fused Q/K/V, attention output,
and GELU feed-forward layers. It uses fixed sinusoidal positions, pre-RMSNorm
residual blocks, and tied full-precision token embeddings/output weights. The
tiny model has 69,952 parameters; the separate 297,059,328-parameter target is
estimated without instantiating it. See [Milestone 2](docs/milestone-2.md).

## Memory and reproducibility

Current execution uses floating-point tensors and dense PyTorch GEMMs. Master
weights, gradients, optimizer state, and dequantized temporary weights consume
memory. There is no demonstrated training-memory reduction or inference speedup.
Three symbols have an ideal information content of log2(3), about 1.585 bits;
straightforward packing uses 2 bits per eligible weight plus scales and metadata.
Milestone 5 measures packed-file compression and validates round-trip weights,
logits, and offline inference. Imported weights are expanded to frozen FP32;
ordinary training state dictionaries still contain full-precision master weights.

The tiny experiment has 2,632 parameters, fixed synthetic data, explicit seeds,
deterministic PyTorch algorithms, and bounded steps. It reports initial/final MSE,
elapsed wall time, and CUDA peak allocated/reserved bytes in `artifacts/`.
Those allocator figures exclude some driver and library allocations and are not
total GPU usage. Timing includes initialization effects and is not a benchmark.
Exact results across different software/hardware versions are not guaranteed.
