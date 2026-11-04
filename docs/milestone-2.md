# Milestone 2: modular ternary decoder — 2026-11-04

Implemented and verified inside the isolated Python 3.13.7 environment with
PyTorch 2.7.1+cu118, CUDA runtime 11.8, and the RTX 4060 Laptop GPU (8 GB).
No additional dependencies were installed. No 300M model was constructed,
allocated, or trained. Milestone 3 has not started.

## Architecture

`TransformerConfig` is immutable and validates positive integer dimensions,
an even model width for sinusoidal positions, divisibility by head count, and
positive finite normalization epsilon. `TernaryTransformer` accepts integer
tokens `[batch, time]` and returns unnormalized logits `[batch, time, vocab]`.
Inputs must have a nonempty batch and 1 through `max_seq_len` tokens. Embedding
lookup enforces token vocabulary bounds.

The model performs these operations:

1. Full-precision learned token embedding, initialized with Normal(0, 0.02).
2. Add fixed absolute sinusoidal positions starting at zero. Channel `2i` uses
   `sin(position / 10000^(2i/d))`; channel `2i+1` uses cosine. Positions are
   non-trainable buffers regenerated from configuration, with no parameter cost.
3. Repeat `n_layers` pre-normalized residual blocks:
   `x = x + Attention(RMSNorm(x))`, then
   `x = x + Down(GELU(Up(RMSNorm(x))))`.
4. Final RMSNorm, then a full-precision vocabulary projection tied directly
   to the token embedding weight with `F.linear`. No independent head parameter.

RMSNorm computes `x / sqrt(mean(x^2) + 1e-6)` with a learned scale and no bias.
FP16/BF16 statistics and softmax accumulate in FP32; validation and training here
use FP32 throughout. Mixed-precision training has not been validated.

Attention projects fused Q/K/V from width `d` to `3d`, reshapes to multiple
heads, and uses `softmax(Q K^T / sqrt(head_dim)) V`. The strict upper triangle
is filled with negative infinity **before softmax**. Each token can attend to
itself and its prefix. Each head uses the complete prefix; there is no grouped
or multi-query attention. The heads are concatenated and projected to `d`.

The existing `BitLinear` and quantization code are unchanged. Fused Q/K/V,
attention output, FFN up, and FFN down use bias-free `BitLinear`. The fused Q/K/V
matrix has a single per-tensor quantization scale, as defined in Milestone 1.
The FFN is two projections with GELU, not a gated FFN. All master parameters,
gradients, optimizer state, embeddings, RMSNorm scales, residual streams,
attention scores, and activations remain floating point. The tied output head
uses the original embedding values without ternary quantization. There is no
dropout or bias in this architecture. `BitLinear` keeps its existing initialization.

## Configurations and exact counts

| Setting | Tiny model (constructed) | 300M target (arithmetic only) |
| --- | ---: | ---: |
| Vocabulary | 64 | 32,000 |
| Maximum context | 32 | 2,048 |
| Blocks | 2 | 24 |
| Width | 64 | 1,024 |
| Attention heads | 4 | 16 |
| Head width | 16 | 64 |
| FFN width | 128 | 3,328 |
| Embedding and tied head | 4,096 | 32,768,000 |
| Attention projections | 32,768 | 100,663,296 |
| FFN projections | 32,768 | 163,577,856 |
| RMSNorm scales | 320 | 50,176 |
| **Unique trainable parameters** | **69,952** | **297,059,328** |

For vocabulary `V`, width `d`, FFN width `f`, and layer count `L`, the total is:

```text
V*d + L*(4*d*d + 2*d*f + 2*d) + d
```

There are no positional parameters, projection biases, or separate vocabulary
head parameters. `count_parameters` counts unique model Parameters, and
`estimate_parameters` returns an arithmetic breakdown. Tests compare both on
two small configurations. `target_300m_config` only creates a configuration;
`scripts/estimate_architecture.py` never constructs a model or calls CUDA.
The target count is approximately 300M, not exactly 300,000,000. These counts
are not a GPU memory budget for training the target architecture.

## Verification results

The complete suite passed **17 tests in 3.430 seconds**, with **zero skips,
zero failures, and zero errors**. This includes the six unchanged Milestone 1
tests and eleven new architecture tests. Coverage includes:

- Configuration and input validation; lengths 1, intermediate, and full context.
- Sinusoidal positions, RMSNorm reference values, finite outputs, and shapes.
- Attention comparison to an independent per-position prefix computation.
- Future-token perturbations at three boundaries on both CPU and CUDA. Earlier
  logits remain bitwise identical for same-shaped inputs in evaluation mode.
  Prefix-only execution also matches the corresponding full-sequence outputs
  within FP32 tolerance, allowing different matrix shapes to round differently.
- Finite, nonzero gradients for every trainable parameter on CPU and CUDA,
  and a real optimizer update.
- Parameter arithmetic, reuse of BitLinear, seeded initialization, and an
  in-memory state-dictionary save/load round trip.

The initial verification exposed a cuBLAS determinism setting being applied
after the verifier's first CUDA operation. The verifier now sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch. It also accepts the
expanded test suite instead of requiring exactly six tests. Final verification
passed. `pip check` reports no broken requirements.

## Measured synthetic next-token smoke test

`experiments/train_next_token.py` always selects the tiny configuration and
limits training to at most 1,000 steps. The measured run used seed 42, FP32,
100 AdamW steps, learning rate 0.003, weight decay 0.01, and gradient clipping
at norm 1.0. The fixed batch contains 16 sequences of length 16. Targets are
the next tokens in cyclic sequences modulo 64, shifted by one position from
the inputs. Validation uses different sequence starting offsets.

| Measurement | First CUDA run |
| --- | ---: |
| Initial training cross-entropy | 4.1715865135 |
| Final training cross-entropy | 0.0135443406 |
| Initial training token accuracy | 1.953125% |
| Final training token accuracy | 100% |
| Validation cross-entropy | 0.6447979212 |
| Validation token accuracy | 73.046875% |
| Elapsed seconds | 1.9793755 |
| Peak PyTorch allocated bytes | 70,614,016 (67.34 MiB) |
| Peak PyTorch reserved bytes | 71,303,168 (68 MiB) |

Raw results are in `artifacts/milestone-2-next-token.json` and
`artifacts/milestone-2-next-token-repeat.json`. The second seeded run reproduced
all loss and accuracy values exactly and took 1.7625329 seconds. Timing and
allocator peaks cover initial evaluation, training, final evaluation, and
validation; they exclude model/optimizer construction time. CUDA work is
synchronized for timing. Memory peaks include live model/data allocations but
exclude some CUDA/driver allocations and other applications' GPU use.

This is optimization on a simple synthetic transition task. Validation contains
the same successor rule and vocabulary, with changed offsets; it is not evidence
of language quality, unseen-transition generalization, or real-data scaling.
The validation gap remains visible. Timing is not a throughput benchmark.

## Exact verification commands (Windows PowerShell)

From the project root in VS Code's terminal, without global Python fallback:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe scripts\estimate_architecture.py
.\.venv\Scripts\python.exe -m pip check
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\.venv\Scripts\python.exe experiments\train_next_token.py --device cuda --steps 100 --seed 42 --output "artifacts/next-token-$runTag.json"
```

The smoke script sets deterministic algorithms and seeds Python and PyTorch;
TF32 matrix multiplication is disabled. It fails on nonfinite loss/gradients
or a final loss that did not decrease. CUDA errors do not silently select CPU.
`--device cpu` explicitly selects a CPU smoke run. Output creation is exclusive:
the script rejects existing files and preserves previous results. Reproducibility
across other hardware/software stacks is not guaranteed.

## Files and preservation

Created `src/bitforge/config.py`, `src/bitforge/transformer.py`,
`tests/test_transformer.py`, `scripts/estimate_architecture.py`,
`experiments/train_next_token.py`, this report, and two new result artifacts.
Modified `src/bitforge/__init__.py` exports, `scripts/verify_environment.py`,
and README. Dependency declarations and lock snapshot are unchanged.

The original BitLinear, quantization implementation, core tests, and both
Milestone 1 result JSON files were verified unchanged using SHA-256 snapshots.
Milestone 1's experiment script was not modified or rerun. Artifacts remain
ignored by the existing `.gitignore`; copy them separately if archiving results.

## Git status and origin

`git rev-parse --show-toplevel` fails. There is no `.git` file/directory here or
in any parent through `C:\`. No source remote was found in project documentation
or metadata. The Milestone 1 report states that this project began in an empty
directory. Available evidence does not establish a copied upstream repository
or recover its history. If a separate original checkout exists, compare it
before choosing how to retain its history.

For a new standalone repository, recommended commands for the user to run are:

```powershell
Set-Location 'C:\Users\Rushi Thatikonda\Desktop\BitForge-300M'
git init
git status --short
```

Git was not initialized, and no commit or push was performed. `git diff --check`
returned exit 129 because it cannot assess a working tree without metadata.
Separate `git diff --no-index --check` whitespace checks passed for all nine
added or modified source/documentation files. Python compilation of `src`,
`tests`, `experiments`, and `scripts` also passed.

## Limits and stopping point

Attention explicitly materializes quadratic `[batch, heads, time, time]` scores.
There is no KV cache, tokenizer, padding mask, variable-length batching support,
generation API, dropout, packed checkpoint format, integer kernel, distributed
training, or validated mixed-precision training. Every supplied token is a real
context token, and context starts at zero. State dictionaries require the same
configuration, which must be stored separately. Model checkpoints are not saved
by the synthetic smoke script.

Ordinary floating-point ternary tensors execute dense PyTorch operations. No
packed ternary inference, compression, training-memory reduction, or performance
gain is claimed. The 300M configuration remains unallocated and its feasibility
on an 8 GB GPU has not been tested. Milestone 2 is complete; further work requires
Milestone 3 approval.
