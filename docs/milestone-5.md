# Milestone 5: packed ternary inference checkpoints

Completed 2026-11-04 using isolated Python 3.13.7, PyTorch 2.7.1+cu118,
CUDA runtime 11.8, and the RTX 4060 Laptop GPU. This milestone exports an
existing trained model; **no training or 300M allocation occurred**. Training,
quantization, model architecture, and existing checkpoint utilities are unchanged.
No dependencies were added.

The measured source is the **ternary seed-42 model from Milestone 4**:
`artifacts/milestone-4-paired/seed-42-ternary/checkpoint.pt`. Its original
FP32 master-weight state dictionary is `weights.pt` in the same directory.
The benchmark verifies every master weight against that weights-only file
before exporting, so this compares two representations of the same trained
model, not separately trained FP32 and ternary models.

## Actual compression and eligibility

| Measurement | Value |
| --- | ---: |
| Original FP32 weights-only checkpoint | **335,225 bytes** |
| Packed checkpoint | **87,324 bytes** |
| Original / packed compression ratio | **3.8388644588x** |
| File-size reduction | **73.95063017%** |
| Complete model parameters | 82,432 |
| Eligible projection parameters | **65,536 (79.50310559%)** |
| Non-eligible FP32 parameters | 16,896 (20.49689441%) |
| Complete packed-file bits per model parameter | **8.47476708** |

This is a real on-disk size measurement, including metadata and checksums.
The model as a whole does not occupy two bits per parameter.

All eight `BitLinear.weight` tensors are eligible. For each of two blocks:

| Name suffix under `blocks.<index>` | Shape | Parameters per block |
| --- | --- | ---: |
| `attention.qkv.weight` | [192, 64] | 12,288 |
| `attention.output.weight` | [64, 64] | 4,096 |
| `ffn.up.weight` | [128, 64] | 8,192 |
| `ffn.down.weight` | [64, 128] | 8,192 |

The 16,576 token-embedding parameters and 320 RMSNorm scales remain exact FP32.
The output head shares the embedding parameter and is not stored twice. There
are no projection biases in this architecture; biases would not be eligible.
Fixed sinusoidal positions and causal masks are nonpersistent buffers rebuilt
from the validated architecture configuration, not trainable weights.

## Quantization and reconstruction semantics

Export calls the existing `ternary_codes` on the **source device**, without
changing its formula. For each FP32 eligible tensor W:

```text
s = max(mean(abs(W)), finfo(float32).eps)
T = clamp(round(W / s), -1, +1)    # round to nearest, ties to even
Q = float32(T * s)
```

The existing BitLinear forward uses `Q.detach() + (W - W.detach())` before
`F.linear`, with identity surrogate gradients during training. Export stores
T and the exact FP32 scale s; it does not retain W. The model must have finite
FP32 state, and the source backend must be ternary. Scale reductions can round
differently on CPU versus CUDA, so the measured export runs on CUDA to match
the source CUDA forward. The source device is recorded in the manifest. Cross-
device scale recomputation is not used during import.

Import reconstructs Q exactly as FP32 values and uses **frozen nn.Linear
projections**. It does not put Q back into BitLinear. Re-quantizing Q would
compute a new absmean scale, generally changing the intended forward weights.
The source config stays in the manifest; the returned model's runtime backend
is `fp32` because the weights are already quantized and dequantized. This is
dense inference, not the separately trained FP32 model from Milestone 4.
All imported parameters have gradients disabled. Full-precision tensors are
restored exactly, including the shared embedding/output weight.

The export is lossless with respect to the **quantized forward weights**, but
lossy with respect to FP32 master weights. It is inference-only: it contains
no optimizer, master-weight recovery data, or resumable training state. Resume
training from the preserved original checkpoint instead.

## Version-1 binary format

The `.bft` file is a single binary container. It contains no pickle, arbitrary
Python objects, dynamic class names, executable deserialization, or external
tensor-file references.

| Section | Encoding / measured bytes |
| --- | --- |
| Magic | 8 ASCII bytes: `BFTPACK1` |
| JSON header length | 4-byte unsigned little-endian integer |
| Header | 3,280 bytes of canonical UTF-8 JSON |
| Tensor payload | 84,000 bytes |
| File integrity footer | 32 raw SHA-256 bytes over every preceding byte |
| **Total** | **87,324 bytes** |

Header JSON uses sorted keys, compact separators, and finite JSON values.
Fields include format name, integer version 1, architecture identifier
`bitforge-decoder-v1`, full source configuration, exact tokenizer metadata,
quantizer identifier `absmean-fp32-round-even-clamp-v1`, little byte order,
source quantization device, and the ordered tensor manifest. The manifest
contains 14 entries: eight packed projection weights and six raw FP32 tensors.
Each entry records name, shape, dtype, encoding, payload-relative byte offset,
byte length, and SHA-256 of its blob.

For `ternary2-f32scale`, each blob begins with the 4-byte little-endian FP32
scale followed by the packed values. Flattening is contiguous row-major:

| Two-bit code | Value |
| --- | ---: |
| `00` | -1 |
| `01` | 0 |
| `10` | +1 |
| `11` | Invalid; always rejected |

Values 0 through 3 occupy byte bit lanes [1:0], [3:2], [5:4], and [7:6],
respectively. Example `[-1, 0, +1, -1]` becomes `0x24`. A final partial byte
uses canonical `01` zero-value padding in unused lanes. The shape determines
the true count; padding is validated. Packed tensor length is
`4 + ceil(numel / 4)` including its scale.

For `raw-f32`, a blob is exactly `4 * numel` little-endian bytes. Version 1
supports FP32 tensors only and rejects other dtype declarations. Payload sizes:

- Ternary codes: 16,384 bytes.
- Eight FP32 scales: 32 bytes.
- Non-eligible FP32 tensors: 67,584 bytes.
- Metadata and framing/checksum overhead: 3,324 bytes.

The measured packed file SHA-256 is
`eec8a7c12400fbbbc447da4cceca4f26670c35b4a4d7a27adc6bd2d2bcbbfac4`.
Both benchmark exports and a separate CLI export produced identical bytes.

## Strict loading and integrity

The packed importer never calls `torch.load`. It bounds input to 16 MiB and
header JSON to 1 MiB, verifies magic/length and the whole-file digest, rejects
duplicate JSON keys, and checks exact schema/version/architecture/quantizer.
It enforces the existing 2M-parameter/context-256 small-model guard before
constructing a template on CPU. CPU FP32 reconstruction is explicit, even if
the caller has changed the global default dtype or device.

Names must match the expected state exactly, with no duplicates, missing
tensors, or extras. Shape/dtype/eligibility/encoding must match. Offsets must
be contiguous with no gaps, overlaps, unreferenced bytes, or truncated blobs.
Blob checksums, positive finite scales, valid codes, canonical padding, and
finite raw values are checked. Only after every tensor validates is the
complete state loaded with `strict=True` into the frozen runtime model.
GPU transfer happens afterward. There is no random-weight substitution,
missing-weight fallback, or partial successful load.

Callers can pass `expected_config` to reject a different source architecture,
including changes such as head count that may not change tensor shapes. The
benchmark does this. The self-contained generation CLI accepts the file's
validated supported configuration.

SHA-256 checksums detect corruption; they do not authenticate the producer.
Someone who can modify the data and recompute all digests can create a
different valid model. No signature or trusted-producer mechanism is provided.
Resource bounds and the no-pickle format reduce loading risk, but are not a
general security guarantee against all malformed inputs. Legacy source and
control `.pt` files are locally produced and read only with `weights_only=True`;
there is no unsafe fallback.

## Reconstruction and logits results

On the trained CUDA source, every eligible reconstruction matched the intended
quantized weight with **rtol=0, atol=0**. Every non-eligible tensor matched its
original FP32 value exactly. Probes used batch size 2 and lengths **1, 8, 32,
and 64**. Every measured maximum absolute logit error was **0.0**.

The enforced logit tolerance is `rtol=1e-5, atol=1e-6`: a small allowance for
ordinary FP32 GEMM rounding if execution paths differ. The measured same-device
comparison needed no allowance. Both original and reconstructed models had
validation loss **2.9806811292** and byte perplexity **19.7012311596**, over
the same 1,055 held-out targets from the unchanged corpus/split/evaluator.
CPU and CUDA unit tests also check quantized weights and logit equivalence.

Offline greedy generation from `The workshop ` produced 120 spaces, as did
the original model and dense control. The separate packed CLI ran successfully
on CPU. Compression preserves the source model's poor sample quality; it does
not improve language ability.

## Loading, inference memory, and latency measurements

Primary results are in `artifacts/milestone-5-packed-final/results.json`,
with a reviewable snapshot in [milestone-5-results.json](milestone-5-results.json).
The preparation process exports and verifies weights/logits, then exits fully.
Each of these three modes runs in its own fresh worker process:

1. **Master BitLinear:** the existing 335,225-byte master-weight state dictionary,
   loaded into the original ternary architecture; quantization runs each forward.
2. **Dense frozen control:** a new 335,225-byte FP32 state dictionary of Q,
   loaded into frozen ordinary Linear projections. This is the same effective
   model as the packed import, without packed disk storage.
3. **Packed:** the 87,324-byte `.bft` file, validated and reconstructed into
   those same dense frozen FP32 projections.

Loading is measured over five repetitions after an untimed load warms the
OS file cache and loader. CPU loading includes reading, validation, model
construction, and state restoration. CUDA loading includes those operations
plus host-to-device transfer and synchronization. Imports/process startup and
garbage collection between repetitions are excluded. These are **warm-cache
load times**, not cold disk I/O or total command startup latency.

Inference uses a pre-resident CUDA [1, 64] input, 20 warmup forwards, then five
repetitions of 100 forwards in `torch.inference_mode()`. CUDA is synchronized
before/after each timed repetition. Each forward produces all-position logits;
it is not KV-cached token latency or generation throughput. Inputs/decoding,
loading, and generation are excluded. All models have gradients disabled and
no optimizer state. Memory peaks reset after warmup and apply only to this
inference phase. CPU reconstruction memory and loading peaks are not reported.

Mean +/- sample standard deviation across the five measurements/repetition means:

| Mode | CPU load ms | CUDA load ms | Forward ms |
| --- | ---: | ---: | ---: |
| Master BitLinear | 2.909940 +/- 0.034021 | 5.068380 +/- 1.498935 | 3.474768 +/- 0.128932 |
| Dense frozen control | 3.117440 +/- 0.253107 | 4.355880 +/- 0.316877 | 2.599125 +/- 0.097090 |
| Packed import | **9.634940 +/- 2.491747** | **12.847540 +/- 1.777228** | **2.794287 +/- 0.340777** |

| Inference memory | Master BitLinear | Dense frozen | Packed import |
| --- | ---: | ---: | ---: |
| Raw FP32 parameter bytes | 329,728 | 329,728 | 329,728 |
| Allocated baseline bytes | 33,910,784 | 33,910,784 | 33,910,784 |
| Peak allocated bytes | 34,206,208 | **34,156,544** | **34,156,544** |
| Peak allocated MiB | 32.6216 | 32.5742 | 32.5742 |
| Peak reserved bytes | 35,651,584 | 35,651,584 | 35,651,584 |
| Peak reserved MiB | 34 | 34 | 34 |

Packed import has **identical measured inference VRAM** to the dense frozen
control. The small difference from original BitLinear reflects removing
repeated quantization temporaries, not two-bit GPU storage. The frozen paths
use dense FP32 kernels and hold full FP32 weights in VRAM. Any difference in
latency from BitLinear also includes avoiding runtime quantization; it cannot
be attributed to file compression. The packed path did not outperform its
unpacked control here, and its validated loading was slower despite a smaller
file. CPU decoding, checksums, copies, and template/runtime construction add
overhead in this implementation.

This is one model and one session, with a fixed mode order. Repeated timing
samples are not independent training seeds or confidence intervals. Hardware
clocks, thermals, scheduling, and machine load were not controlled. PyTorch
allocator measurements exclude some driver allocations and other applications'
VRAM. The inference-only fresh-process setup differs from Milestone 4's
post-training process, so memory/latency values should not be compared across
milestones as evidence of a packing benefit.

## Exact commands and reproducibility

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pip check
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$packedFile = "artifacts/model-$runTag.bft"
.\.venv\Scripts\python.exe scripts\export_packed.py --checkpoint artifacts\milestone-4-paired\seed-42-ternary\checkpoint.pt --output $packedFile --device cuda
.\.venv\Scripts\python.exe scripts\generate_packed.py --checkpoint $packedFile --device cpu --prompt 'The workshop ' --max-new-tokens 120
.\.venv\Scripts\python.exe experiments\benchmark_packed.py --output-dir "artifacts/packed-$runTag"
```

The exact primary measurement command was:

```powershell
.\.venv\Scripts\python.exe experiments\benchmark_packed.py --output-dir artifacts\milestone-5-packed-final
```

That directory is already populated. Output files/directories must be new;
exclusive creation protects existing results. The first development measurement
is preserved separately in `artifacts/milestone-5-packed/`; final numbers above
come from the explicitly guarded FP32/CPU loader in `milestone-5-packed-final`.
No favorable timing samples were substituted between runs.

The final directory includes `model.bft`, a deterministic CLI reproduction
`cli-export.bft`, `dense-quantized-weights.pt`, `export.json`, three per-mode
measurement JSONs, and aggregate `results.json`. Loading/generation need no
network or corpus download. The benchmark's corpus is the unchanged local
5,807-byte CC0 fixture; no hyperparameters or model weights were trained here.

## Tests, files, and remaining limitations

The full isolated environment verifier passed **49 tests in 3.354 seconds**,
with zero failures, errors, or skips. The 14 new tests cover bit layout and
odd-length padding, deterministic/exclusive export, CPU/CUDA model equivalence,
eligibility and unsupported dtypes/backends, corruption/truncation, invalid
codes and bad scales even with recomputed checksums, per-tensor checksum
failure, metadata/shape/dtype/offset mismatches, missing/duplicate weights and
extra payload, incompatible configuration/format/resource bounds, duplicate
JSON keys, absence of pickle loading, and independence from global default
dtype/device. The previous 35 tests remain unchanged.

Created `src/bitforge/packed.py`, `tests/test_packed.py`,
`scripts/export_packed.py`, `scripts/generate_packed.py`,
`experiments/benchmark_packed.py`, this report, and
`docs/milestone-5-results.json`. README is updated. Existing source modules,
training scripts, checkpoint helpers, dependencies, corpus, tests, and all
previous results/checkpoints are preserved. Generated artifacts remain ignored
by the existing `.gitignore`; archive `.bft` files separately if needed.

Limitations: FP32-only little-endian version 1; bounded small known architecture;
lossy master-weight export; no training resume from packed files; no producer
authentication; no streaming/zero-copy loading; CPU unpacking and duplicate
template/runtime allocation during validation; no packed GPU kernels or
two-bit VRAM representation. General performance claims require broader,
balanced measurements. Real packed execution would be a separate project.

Final SHA-256 checks confirmed all 52 previous source, test, dependency, corpus,
checkpoint, and artifact files are unchanged. All eight added/modified source
and documentation files passed separate no-index whitespace checks. Compilation,
`pip check`, result-snapshot checks, reproducible file hashes, exact validation
and sample comparisons, and the identical dense-control memory check passed.
`git diff --check` returned exit 129 because metadata is absent. Git remains
uninitialized, with no commit or push. Milestone 5 stops here.
