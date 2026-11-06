# Milestone 6 - QAT and corpus experiment

The offline corpus contains 700 original procedural engineering records, 263,537 UTF-8 bytes (45.4 times the 5,807-byte baseline). The original templates and generated corpus are dedicated under CC0-1.0. No external passages or downloads were used. Provenance and license are recorded in `data/engineering_corpus.manifest.json`; generator: `scripts/prepare_engineering_corpus.py`, seed 606.

Corpus SHA-256: `e9d99fa6fe88b266989f61793b3bedd1c61a8dd398b5cb45ac2df1a58f1a6942`.

Split before tokenization/windowing at paragraph boundary byte 210,835: 210,835 training byte tokens and 52,702 validation tokens (52,701 evaluated targets). Split hashes are in results JSON. No windows cross the boundary. Paragraphs are distinct, but templates and phrases overlap heavily: this is a synthetic pipeline experiment, not an independent natural-language benchmark.

The model has **1,378,176 unique trainable parameters**: width 192, 3 blocks, 6 heads, FFN 768, tied 259-byte-token embedding/head, context 64. All three backends retain identical master parameter layouts and seeded initialization. AdamW uses LR 0.001, weight decay 0.01, gradient clipping 1, batch 4. Seeds 42, 43, 44 are paired for initialization and sampling; each run performs 100 updates (25,600 training targets), with complete validation before and after.

Original BitLinear remains the absmean baseline. The new LeastSquaresBitLinear minimizes ||W-alpha Q||^2 by four alternating updates: alpha=<W,Q>/<Q,Q> and Q=clip(round(W/alpha),-1,1). Each coordinate update minimizes the objective with the other fixed, making reconstruction error nonincreasing from absmean initialization, up to rounding. Zero denominators/scales are guarded. The identity straight-through gradient remains a surrogate; smaller weight reconstruction error does not guarantee lower validation loss.

| Backend | Mean validation CE +/- sample SD | Mean perplexity | Mean run seconds | Peak allocated MiB |
|---|---:|---:|---:|---:|
| fp32 | 3.033367 +/- 0.004957 | 20.7672 | 3.768 | 94.56 |
| ternary | 3.033389 +/- 0.004412 | 20.7676 | 4.583 | 98.04 |
| ternary_ls | 3.033338 +/- 0.004913 | 20.7666 | 9.159 | 98.04 |

Paired improved-minus-original CE: [-0.0004739227703165483, -0.00025883322818343046, 0.0005806135576191807]. No consistent QAT quality improvement is demonstrated. All 900 updates had finite losses and gradients; gradient norms and loss ranges are recorded per run. Times include initial/final evaluation and have no repeated timing confidence intervals. Some initial runs overlapped the short unit-test run; treat timing as indicative. These are very short runs on repetitive synthetic data, with only three seeds.

Reproduce from the repository root (existing Python 3.13 / PyTorch 2.7.1+cu118 environment):
```powershell
python scripts/prepare_engineering_corpus.py
python -m unittest discover -s tests -v
python scripts/estimate_architecture.py
python experiments/phase1_advanced.py --output-dir artifacts/phase1-reproduction --steps 100
python experiments/benchmark_advanced_loading.py artifacts/phase1-reproduction
```
Use a new output directory for each run; files are created exclusively. No downloads occur. CUDA and BF16 support are required for the experiment. Results: `artifacts/phase1-final/results.json` and `loading.json`. Baseline `3a1e760` and Milestones 1-5 artifacts were preserved. No commit or push was performed.
