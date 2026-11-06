# Release candidate checklist

The user authorized committing and pushing the verified source release on
`main`. The exact file inventory and fresh checks are in [github-release.md](github-release.md).
Checkpoint distribution, standalone packaging, and release tags are outside this release.

- [x] Preserve Milestones 1–8 implementation and experimental evidence.
- [x] Verify original 57-test suite before changes.
- [x] Construct real 297,059,328-parameter model, CPU and CUDA finite forward.
- [x] Perform inference memory and disk preflight; no full-model training.
- [x] Export/reload full packed checkpoint and measure real compression/error.
- [x] Compare packed vs identical dense reconstruction with repeated forwards.
- [x] Keep FP32, BF16, QAT, frozen dense, and model sizes separately labeled.
- [x] Exercise offline CLI, sampling, errors, and old checkpoint compatibility.
- [x] Produce a trained small-model demo with measured validation and provenance.
- [x] Separate opt-in expensive tests from ordinary tests; record actual skips.
- [x] Document setup, reproduction, benchmark conditions, and video commands.
- [x] Keep checkpoint binaries ignored; no credentials or binary weights tracked.
- [x] User review and explicit instruction to perform the safe GitHub source release.
- [x] Explicit approval to commit and push to `origin/main`; no tags or force-push.

Recheck in a future release session:

```powershell
git diff --check
git status --short
git ls-files '*.pt' '*.pth' '*.bft' '*.safetensors' '*.env' '*.pem'
python scripts/run_release_tests.py --include-full --output artifacts/new-release-tests.json
```

The current acceptance-gate results and exact test totals are in the final
technical report and `artifacts/release-tests-final.json`. A fresh offline machine
must recreate local binaries; the Git repository deliberately carries evidence
and code rather than multi-gigabyte checkpoints. Packaging a wheel and a public
model-download channel are not verified by these local acceptance gates.
