# Milestone 4: controlled FP32 versus ternary benchmark

Completed 2026-11-04 in the existing isolated Python 3.13.7 environment,
PyTorch 2.7.1+cu118, CUDA runtime 11.8, NVIDIA driver 617.42, and RTX 4060
Laptop GPU (8,188 MiB reported). No dependencies were added. No 300M model
was instantiated or trained. This is a small assistant-generated-corpus
development experiment, not a general language-model or hardware benchmark.

Full per-run measurements, hashes, samples, timing repetitions, environment,
configuration, and aggregates are in [milestone-4-results.json](milestone-4-results.json).
It is an exact snapshot of `artifacts/milestone-4-paired/results.json`.

## Controlled architecture and initialization

`TransformerConfig.projection_backend` selects `'ternary'` (default) or `'fp32'`.
The shared factory creates the existing `BitLinear` or an exact `nn.Linear`,
respectively, with identical dimensions and `bias=False`. Only the fused QKV,
attention output, FFN up, and FFN down projections vary. There are eight such
modules. The existing Transformer classes are retained for compatibility,
including the historical name `TernaryTransformer` in FP32 mode.

Both models have **82,432 trainable parameters**: two blocks, model width 64,
four heads of width 16, GELU FFN width 128, context 64, and vocabulary 259.
They share sinusoidal positions, pre-RMSNorm/residual layout, full-precision
token embeddings and tied output head, full-precision normalization scales,
causal masking, and no dropout. All tensors and optimizer state use ordinary
floating-point execution; master parameters are FP32 in both modes.

The CPU RNG is reset to the same seed before each model is constructed.
BitLinear inherits nn.Linear initialization unchanged. Each paired run's
initial state-dictionary SHA-256 is identical. Tests additionally compare every
master parameter and buffer, module paths, parameter counts, and output logits
when ternary quantization is temporarily replaced by identity in the test.

Matching FP32 master weights does **not** match effective forward weights:
ternary projections apply the existing absmean quantizer and identity STE;
FP32 projections use master weights directly. Effective initial weight variance
and subsequent optimization trajectories therefore differ. This is the intended
experimental difference. No backend-specific scaling, initialization change,
learning-rate tuning, or other hyperparameter adjustment was made.

The quantization implementation is unchanged, including one per-tensor scale
for each fused QKV matrix. Existing checkpoint configs that omit the new field
default to ternary. New checkpoints record their backend. Standard text training
now accepts `--projection-backend fp32`; its own resume path rejects backend
changes. The benchmark uses the existing checkpoint utility for full training
state and also writes a weights-only state dictionary for size comparison.

## Shared data and optimization budget

The existing CC0 corpus is unchanged: **5,807 UTF-8 bytes**, original local
assistant-generated workshop prose. Its SHA-256 remains
`9a37fcfa0fc55b65c7090240126bc83200379af993417d0eb8b198140c4864ce`.
The unmodified loader preserves the split at byte 4,751: 4,751 training and
1,056 validation tokens. No window crosses the boundary. Tokenizer vocabulary
and reserved special IDs are unchanged.

Each seed/backend trains for **300 steps**, batch size **16**, sequence length
**64**, AdamW learning rate **0.003**, weight decay **0.01**, and gradient clip
norm **1.0**. Python and PyTorch are seeded, deterministic algorithms are enabled,
TF32 is disabled, cuBLAS workspace is configured before torch import, and CPU
intra-op threads are limited to two. No mixed precision or compiled kernels are
used. A private seeded CPU generator produces training windows.

The benchmark hashes the **actual input and target byte batches** used during
training. Paired runs must agree on initialization hash, batch-stream hash,
corpus/split metadata, training/measurement settings, parameter count, and all
architecture fields except the backend; aggregation fails on a mismatch.
All three pairs passed these audits. Initial hashes differ across seeds, as
expected. Seed 42 ternary final losses also reproduce the Milestone 3 results.

Both splits are evaluated with the unchanged token-weighted procedure: each
adjacent target is scored once, including short tail chunks, with context reset
per chunk. Training evaluation scores 4,750 targets; validation scores 1,055.
Perplexity is the exponential of mean byte-token cross-entropy. These are
exhaustive **post-training** evaluations, not a final minibatch's training loss.

## Seed-level quality results

| Seed | Backend | Train CE | Train PPL | Validation CE | Validation PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | FP32 | 2.751557 | 15.666999 | 2.748405 | 15.617700 |
| 42 | Ternary | 2.979366 | 19.675348 | 2.980681 | 19.701231 |
| 43 | FP32 | 2.516915 | 12.390314 | 2.530120 | 12.555010 |
| 43 | Ternary | 2.978380 | 19.655955 | 2.982478 | 19.736671 |
| 44 | FP32 | 2.977587 | 19.640367 | 2.982066 | 19.728527 |
| 44 | Ternary | 2.978724 | 19.662702 | 2.983268 | 19.752256 |

Mean +/- **sample standard deviation across three seeds**:

| Metric | FP32 | Ternary |
| --- | ---: | ---: |
| Train CE | 2.748686 +/- 0.230349 | 2.978823 +/- 0.000501 |
| Train PPL | 15.899227 +/- 3.630601 | 19.664668 +/- 0.009845 |
| Validation CE | 2.753530 +/- 0.226017 | 2.982142 +/- 0.001326 |
| Validation PPL | 15.967079 +/- 3.599498 | 19.730053 +/- 0.026148 |
| Training seconds | 4.602696 +/- 0.340968 | 5.311104 +/- 1.396764 |
| Inference forward milliseconds | 3.662254 +/- 0.181138 | 6.108771 +/- 1.924617 |

PPL means are arithmetic means of per-seed PPL, not the exponential of the
across-seed mean loss. These standard deviations are not confidence intervals.
The paired ternary-minus-FP32 validation PPL difference is **3.762974 +/-
3.589717**. FP32 has lower validation PPL for all three seeds, but the difference
at seed 44 is only 0.023729. Three seeds on one tiny story cannot establish a
general quality ranking. Low variation among collapsed ternary samples is not
evidence of useful learning.

## Wall-clock and inference methodology

Each seed/backend uses a **fresh child process**, preventing earlier runs from
retaining CUDA allocator state. Pair order alternates: FP32/ternary for seed 42,
ternary/FP32 for seed 43, FP32/ternary for seed 44. With three pairs, order is
not perfectly balanced. No runs or outliers were discarded.

There are five untimed training forward/backward warmups, with no optimizer
updates or batch-generator advance. Gradients are cleared, unused cache is
released, CUDA is synchronized, and memory peaks are reset before training.
Training timing covers the 300 optimization steps, including CPU batch
generation, byte-stream hashing, host-to-device transfer, loss finite checks,
backward, clipping, AdamW, and scalar loss collection. First-step optimizer
state allocation is included. Construction, initial/final exhaustive evaluation,
checkpoint saving, process startup, and generation are excluded. This timing
scope differs from Milestone 3's broader elapsed-time scope.

For inference, optimizer state, gradients, final training tensors, and loss are
deleted. Unused cache is released. An already-resident CUDA input of shape
**[1, 64]** is used. There are 20 warmup forwards followed by **five repetitions
of 100 forwards each**, under `torch.inference_mode()`. CUDA is synchronized
immediately before and after each timed repetition. The reported latency is
the mean per-forward wall-clock time. Individual repetition means and variation
are retained in JSON.

This workload computes vocabulary logits for **all 64 positions**. It is not
single-token KV-cached latency, greedy-generation latency, GPU-event kernel time,
or throughput. Input transfer and token decoding are outside inference timing;
CPU dispatch overhead is included. Generation happens afterward and is excluded
from both timed phases and their memory measurements.

| Seed | Backend | Training seconds | Forward milliseconds |
| --- | --- | ---: | ---: |
| 42 | FP32 | 4.756807 | 3.548099 |
| 42 | Ternary | 6.462875 | 7.326483 |
| 43 | FP32 | 4.839403 | 3.567550 |
| 43 | Ternary | 5.712981 | 7.109894 |
| 44 | FP32 | 4.211879 | 3.871112 |
| 44 | Ternary | 3.757457 | 3.889936 |

Timing variation is substantial: ternary training was faster for seed 44, and
its forward latency fell sharply from earlier runs. GPU clocks, temperature,
power policy, operating-system scheduling, and machine activity were not locked
or isolated. A mid-run `nvidia-smi` snapshot reported 65 C and 9.14 W; it is not
a time-series trace or evidence of a particular cause. Fresh processes isolate
allocator lifetimes, not hardware conditions. These observations are preserved
instead of selecting favorable runs. On average ternary took longer here, but
this protocol does not establish a stable general speed ratio.

## Actual memory and checkpoint sizes

All three seeds produced the following identical backend-specific memory peaks
and file sizes; their across-seed sample standard deviations are zero.

| Measurement | FP32 | Ternary |
| --- | ---: | ---: |
| Unique trainable parameters | 82,432 | 82,432 |
| Raw parameter tensor bytes | 329,728 | 329,728 |
| Training peak allocated bytes | 81,571,328 (77.79 MiB) | 81,833,472 (78.04 MiB) |
| Training peak reserved bytes | 102,760,448 (98 MiB) | 102,760,448 (98 MiB) |
| Inference peak allocated bytes | 67,710,976 (64.57 MiB) | 67,760,640 (64.62 MiB) |
| Inference peak reserved bytes | 69,206,016 (66 MiB) | 69,206,016 (66 MiB) |
| Actual weights-only `weights.pt` | 335,225 bytes | 335,225 bytes |
| Actual full training `checkpoint.pt` | 1,018,610 bytes | 1,018,610 bytes |

Training memory includes optimizer state, activations, gradients, live model
and library allocations visible to PyTorch. Inference memory excludes optimizer
and gradients. Their common allocated baselines are 67,464,704 bytes before
training and 67,465,216 bytes before inference warmup; library/context overhead
dominates this very small model. JSON retains these baselines, so peaks should
not be mistaken for parameter storage. Reserved peaks include cached blocks;
driver allocations and other applications' total VRAM are not fully counted.

The ternary backend uses **262,144 more allocated bytes during training** and
**49,664 more during inference** in these runs. It does not reduce measured
VRAM or file size. Both checkpoints store FP32 master parameters; full training
checkpoints also contain AdamW moments, configuration, tokenizer metadata,
progress, and RNG state. The weights-only files use identical serialization
structure and contain no optimizer state. Neither is a packed ternary export.

## Theoretical storage, explicitly unimplemented

There are 65,536 projection weights and 16,896 full-precision embedding/norm
parameters. Hypothetically storing each projection weight in two bits would
require 16,384 bytes, plus 32 bytes for eight FP32 per-tensor scales. Keeping
the remaining parameters FP32 adds 67,584 bytes: **84,000 raw bytes total**.
This arithmetic omits metadata, file headers, alignment, optimizer state, and
the FP32 master weights needed by the present training implementation.

No such packer, packed checkpoint, unpacking path, or packed/integer kernel
exists here. No actual artifact has that size. Ordinary dequantized ternary
tensors still use dense floating-point operations and additional quantization
work. The hypothetical number cannot be used to claim checkpoint compression,
faster inference, or lower training/inference VRAM.

## Generated sample quality

All models use the identical prompt `The workshop ` and greedy generation of
120 bytes, with special IDs suppressed and no KV cache. Full samples are in
JSON. Representative results:

- FP32 seed 42 repeats `te tete te te ...`.
- FP32 seed 43 repeats `te the the te ...`.
- FP32 seed 44 produces 120 spaces.
- Every ternary seed produces 120 spaces.

All samples are poor and repetitive. Lower byte perplexity does not demonstrate
coherent language ability. The shared corpus, vocabulary, short budget, and
context restrictions make this comparison non-generalizable.

## Reproduction and verification

Exact executed benchmark command:

```powershell
.\.venv\Scripts\python.exe experiments\benchmark_projections.py --output-dir artifacts\milestone-4-paired --seeds 42 43 44 --steps 300
```

The directory already exists and is intentionally protected. To repeat in a
new directory, from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pip check
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\.venv\Scripts\python.exe experiments\benchmark_projections.py --output-dir "artifacts/paired-$runTag" --seeds 42 43 44 --steps 300
.\.venv\Scripts\python.exe scripts\generate_text.py --checkpoint artifacts\milestone-4-paired\seed-43-fp32\checkpoint.pt --device cpu --max-new-tokens 120
```

The benchmark exposes common batch size, context, learning rate, warmup counts,
latency repetition count, and iteration count. It uses fresh processes for each
run, exclusively creates outputs, checks pair integrity, and writes per-run
metrics plus `results.json` with sample standard deviations and paired deltas.
The fixed model dimensions remain the small tested configuration. It requires
CUDA and the project `.venv` interpreter, with no silent global/CPU fallback.

The complete environment verifier passed **35 tests in 4.488 seconds**, zero
failures/errors/skips. Eight new tests check backend implementations, common
architecture, exact initial weights, identity-quantization output equivalence,
FP32 CPU/CUDA causal behavior and finite gradients, checkpoint backend/legacy
defaults, storage arithmetic, invalid modes, and aggregate pairing failures.
All 27 prior tests are unchanged. An actual Milestone 3 checkpoint also loaded
and generated offline after the backend addition. No dependency changes were
needed.

## Files, preservation, and next steps

Modified `src/bitforge/config.py`, `src/bitforge/layers.py`,
`src/bitforge/transformer.py`, `experiments/train_text.py`, and README.
Created `src/bitforge/benchmark.py`, `experiments/benchmark_projections.py`,
`tests/test_projection_backends.py`, this report, and
`docs/milestone-4-results.json`. Six new run directories each contain metrics,
full training checkpoint, and weights-only checkpoint; their parent contains
the aggregate JSON. Previous artifacts/checkpoints and the corpus are preserved.
Existing quantization math, tokenizer, batcher, evaluator, and checkpoint
utilities are unchanged. Artifacts remain ignored by Git; the documentation
snapshot keeps machine-readable results available separately.

Recommended future work, not performed:

1. Use a larger independently sourced/licensed corpus and more paired seeds to
   investigate loss plateaus and collapse; predeclare a common hyperparameter
   grid rather than tuning only one backend.
2. Repeat timing in blocked, balanced sessions with stable machine/power
   conditions. Profile dense projection and quantization costs separately
   before attributing the observed wall-clock differences to kernels.
3. If storage or speed is the objective, implement and correctness-test a real
   packed representation and execution path, then measure exported file sizes,
   end-to-end latency, and memory again. No benefit should be inferred from the
   theoretical bit count alone.

Final checks verified all 23 prior files outside the four intentionally modified
code files against pre-change SHA-256 snapshots, including every previous
checkpoint/artifact and the complete corpus. All ten added/modified source and
documentation files passed separate no-index whitespace checks. Compilation,
`pip check`, aggregate recomputation, pair audits, and all recorded checkpoint
file-size checks passed. The documentation result snapshot matches the artifact
JSON content. An actual FP32 benchmark checkpoint also generated offline on CPU.

Git metadata is still absent; `git diff --check` returned exit 129. No Git
initialization, commit, or push was performed. Milestone 4 stops here; the 300M
architecture remains unallocated.
