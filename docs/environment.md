# Environment stabilization — 2026-11-04

The isolated environment is verified. No architecture development or training
was performed. Existing experimental JSON files were preserved and their SHA-256
hashes checked against snapshots taken before edits.

## Interpreter identification

The earlier successful GPU tests used
`C:\Program Files\Python313\python.exe` (the interpreter selected by `py -3.13`),
Python 3.13.7. Its PyTorch 2.7.1+cu118 comes from
`C:\Users\Rushi Thatikonda\AppData\Roaming\Python\Python313\site-packages\torch`.
This agrees with the Milestone 1 report and experiment artifacts. The default
`python` on PATH instead points to `C:\Python314\python.exe`; it was not used
for project installation or validation.

`.venv\pyvenv.cfg` identifies Python 3.13.7 from
`C:\Program Files\Python313`, with `include-system-site-packages = false`.
Initially this environment contained only pip 25.2. All installation and final
checks used `.venv\Scripts\python.exe` explicitly, without enabling user or
system packages or copying an installed torch directory.

## Offline installation and provenance

Package-network access is blocked: a direct request from `.venv` to the official
CUDA index failed with `URLError`, WinError 10051 (network unreachable). The pip
installation attempt returned "No matching distribution found"; that message
alone does not establish wheel incompatibility.

`pip cache list` showed no torch wheel, but inspection of ZIP payloads in pip's
HTTP cache found a complete 2,817,175,599-byte compatible wheel. The cache search
also covered this workspace and Downloads. Other cached torch builds included
2.10.0 and 2.9.1 for CPython 3.13, CUDA 12.1/2.5.1 for CPython 3.10, and CUDA
13.0/2.14.1 for CPython 3.14. The matching Milestone 1 build was selected.

- Source: `https://download.pytorch.org/whl/cu118/torch-2.7.1%2Bcu118-cp313-cp313-win_amd64.whl`
- Cache key: `648de3dd72885ea5e5f1de9a178ac844f58e2c4b6de50ff3eff45ed9`
- Payload: `%LOCALAPPDATA%\pip\cache\http-v2\6\4\8\d\e\<cache-key>.body`
- SHA-256: `3122e59a5fe4e9ee991e7ad4e7002afa549b2873e421759df6454f20f53a6c74`
- Wheel tag: `cp313-cp313-win_amd64`

The SHA-224 of the source URL matches pip's cache key. The computed payload
SHA-256 matches the cached response's `x-amz-meta-checksum-sha256`. The
[official wheel index](https://download.pytorch.org/whl/cu118/torch/) lists this
filename, and [official previous-version instructions](https://pytorch.org/get-started/previous-versions/)
document PyTorch 2.7.1 with CUDA 11.8. The checksum comparison uses local cache
metadata; no separate remote checksum was retrieved.

Compatible cached dependency wheels were recovered into `.venv\wheelhouse` and
installed with `--no-index --find-links .venv\wheelhouse`. Recovery metadata is
in `.venv\cache-recovery.json` (NumPy was recovered subsequently). These local
files remain ignored. NumPy was added as a development extra to remove the
PyTorch initialization warning; the core package still requires only torch.

## Verified results

| Item | Result |
| --- | --- |
| Python | CPython 3.13.7, Windows AMD64, MSC v.1944 |
| Interpreter | `.venv\Scripts\python.exe` |
| PyTorch | 2.7.1+cu118, installed under `.venv\Lib\site-packages` |
| CUDA runtime | 11.8 |
| NVIDIA driver | 617.42 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported |
| CUDA availability | True |
| Forward | BitLinear 16 → 8, CUDA input 4 × 16, output 4 × 8 |
| Backward | Finite CUDA input, weight, and bias gradients |
| Unit tests | 6 passed, 0 failures, 0 errors, 0 skips; final run 2.391 s |
| Dependency consistency | `pip check`: No broken requirements found |
| Setup replay | Offline `setup.ps1` completed successfully |

`nvidia-smi` reports driver CUDA capability 13.4; this is distinct from the
installed torch wheel's CUDA runtime 11.8. Full dependency versions and absolute
module/interpreter paths are in [environment-verification.json](environment-verification.json).
The final setup replay emitted no missing-NumPy warning.

## Reproduction in Windows PowerShell

From the project directory, using the recovered local wheels:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Wheelhouse .venv\wheelhouse
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

On a machine with package-network access, the exact installation command is:

```powershell
Set-Location 'C:\Users\Rushi Thatikonda\Desktop\BitForge-300M'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The script creates `.venv` with `py -3.13` only if absent, checks isolated Python
3.13, installs the pinned requirements using `https://pypi.org/simple` and the
official `https://download.pytorch.org/whl/cu118` index, installs `.[dev]` without
dependency resolution or build isolation, then runs pip check and CUDA/tests.
It stops on failure. `ExecutionPolicy Bypass` applies to this child process;
direct script execution was blocked by the current PowerShell execution policy.

`requirements-lock.txt` is a resolved version snapshot for Windows x64/Python
3.13, including build tools. Offline dependency installation and editable build
succeeded before it was written. It is not a hash-complete, cross-platform lock,
and a clean network reinstall has not been tested here. Setup does not overwrite
the snapshot. A fresh machine needs the declared Python version and network
access or a complete compatible wheelhouse. Project constraints intentionally
limit Python to 3.13 and torch to the validated 2.7 minor series.

## File changes and Git check

Updated `pyproject.toml`, `scripts/setup.ps1`, and README setup instructions.
Added `requirements-lock.txt`, `scripts/verify_environment.py`, this report,
and `docs/environment-verification.json`. The verifier requires the explicit
venv interpreter, checks torch's installation path and RTX 4060 CUDA gradients,
and rejects skipped or missing tests. No core source or existing tests changed.

`git diff --check` was run but failed because this directory has no `.git`
metadata (also noted in the original milestone report). No repository was
initialized. Separate no-index comparisons against pre-edit snapshots and
whitespace checks cover the changed files; Git history/status is unavailable.
No commit or push was performed.
