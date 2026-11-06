# GitHub source release record

This records the authorized commit/push preparation after the engineering
phase. No additional model features or training were introduced in this release
session. The user explicitly authorized committing and pushing to `origin/main`.
No tags, force-push or history rewriting are part of this operation.

## Fresh verification

- Fast suite: **71 passed, 0 failed, 0 skipped**.
- Real full-model suite: **3 passed, 0 failed, 0 skipped**.
- CLI subprocess acceptance: **13 passed, 0 failed**.
- Full-model, portable checkpoint and packed file sizes/SHA-256 matched the
  recorded local release checksums. Corpus hash matched the provenance manifest.
- Paired QAT runs, memory optimization results and generation measurements were
  retained. README measurements match the corresponding final run artifacts.
- Gates A (architecture), B (checkpoint), C (offline application), D (research)
  and E (release readiness) passed local re-verification.
- Source/JSON/CSV/text only: no checkpoint binaries, environments, caches,
  credentials or unrelated files belong to the inventory below.
- The engineering corpus uses `-text` in `.gitattributes`, preserving its exact
  SHA-256 bytes on Windows checkouts just like the original corpus.

Fresh receipts: [tests](../artifacts/github-release-tests.json),
[CLI checks](../artifacts/github-release-cli.json). Prior reports and interrupted
attempt evidence remain unchanged, apart from a historical-status note in the
technical report. Checkpoint binaries remain ignored locally; a fresh clone
must reproduce them using the documented commands.

This document describes the reviewed commit contents. The actual commit hash,
remote push result and final working-tree status are verified after the push
and returned in the release completion report; a document cannot embed its own
commit hash without changing that hash.

## Exact release file inventory

68 files changed or added relative to baseline `3a1e760`:

```text
.gitattributes
.gitignore
README.md
artifacts/github-release-cli.json
artifacts/github-release-tests.json
artifacts/phase1-final/execution-preflight-verification.json
artifacts/phase1-final/loading.json
artifacts/phase1-final/results.json
artifacts/release-attempts.json
artifacts/release-cli.json
artifacts/release-demo-final/results.json
artifacts/release-demo-final/resume-verification.json
artifacts/release-demo-storage/results.json
artifacts/release-full-final/quantizer-devices.json
artifacts/release-full-final/results.json
artifacts/release-full-storage/results.json
artifacts/release-full/results.json
artifacts/release-full/verification.json
artifacts/release-generation.json
artifacts/release-summary-final/benchmarks.csv
artifacts/release-summary-final/benchmarks.json
artifacts/release-summary-final/checksums.json
artifacts/release-summary-final/generation.csv
artifacts/release-summary-final/review.json
artifacts/release-summary-final/summary.json
artifacts/release-summary/benchmarks.csv
artifacts/release-summary/benchmarks.json
artifacts/release-summary/checksums.json
artifacts/release-summary/summary.json
artifacts/release-tests-final.json
artifacts/release-tests.json
data/engineering_corpus.manifest.json
data/engineering_corpus.txt
docs/final-technical-report.md
docs/github-release.md
docs/milestone-6.md
docs/milestone-7.md
docs/milestone-8.md
docs/milestones-1-8-readme.md
docs/portfolio-demo.md
docs/release-checklist.md
docs/reproducibility.md
docs/windows-setup.md
experiments/benchmark_advanced_loading.py
experiments/benchmark_generation.py
experiments/benchmark_release_storage.py
experiments/inspect_quantizer_devices.py
experiments/phase1_advanced.py
experiments/release_demo.py
experiments/release_full_model.py
experiments/verify_full_checkpoints.py
scripts/prepare_engineering_corpus.py
scripts/run_release_tests.py
scripts/summarize_release.py
scripts/verify_release_cli.py
src/bitforge/__main__.py
src/bitforge/config.py
src/bitforge/execution.py
src/bitforge/inference.py
src/bitforge/layers.py
src/bitforge/packed.py
src/bitforge/quantization.py
src/bitforge/resources.py
src/bitforge/training.py
src/bitforge/transformer.py
tests/expensive/test_full_model.py
tests/test_advanced.py
tests/test_inference.py
```
