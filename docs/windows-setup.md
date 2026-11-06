# Windows PowerShell setup

Verified hardware/software: Windows 11 x64, Ryzen 9, 32 GB RAM, RTX 4060 Laptop
8 GB, Python 3.13.7, PyTorch 2.7.1+cu118. No extra runtime dependencies beyond
PyTorch were added in the release phase. CUDA BF16 tests require compatible
hardware/drivers. CPU inference works without CUDA.

From an existing checkout and the prepared environment:

```powershell
Set-Location 'C:\Users\Rushi Thatikonda\Desktop\BitForge-300M'
.\.venv\Scripts\Activate.ps1
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -m bitforge --help
```

If activation is restricted, call `.\.venv\Scripts\python.exe` directly; no
execution-policy change is needed. The editable-install command uses local
source and requires setuptools already present.

For a new environment, prepare CPython 3.13 and the pinned PyTorch wheel and
its dependencies separately. Installing PyTorch is a large download; the
project never downloads it or datasets automatically. An offline wheelhouse
prepared for Windows x64 / CPython 3.13 can be used:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links C:\wheelhouse -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
```

`requirements-lock.txt` is the original environment's version snapshot, not a
cross-platform or hash-complete dependency lock. CUDA versions and binary
wheels must match the prepared environment. Do not try to build PyTorch from
source on this laptop as part of setup.

Model files are local generated artifacts, deliberately ignored by Git. A fresh
clone contains source, datasets, tests, and measured JSON/CSV evidence; it does
not include checkpoint binaries. Recreate the demo and full-size files using
[reproducibility.md](reproducibility.md). No pretrained weights are needed.

Run the ordinary suite first. The opt-in full suite constructs a real 297M
model and writes a temporary ~197 MB packed checkpoint. The release runner
writes ~1.39 GB of persistent model files; the optional frozen dense benchmark
adds ~1.19 GB. Keep several GiB of RAM and disk available and close other GPU
workloads. Preflight guards refuse unsafe budgets rather than automatically
scaling or starting a training job.
