param([string]$Wheelhouse)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if (-not (Test-Path .venv\Scripts\python.exe)) {
        py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    $python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
    & $python -c "import sys; assert sys.version_info[:2] == (3, 13); assert sys.prefix != sys.base_prefix"
    if ($LASTEXITCODE -ne 0) { throw 'Isolated Python 3.13 is required' }
    if ($Wheelhouse) {
        & $python -m pip install --no-index --find-links $Wheelhouse -r requirements-lock.txt
    } else {
        & $python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-lock.txt
    }
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed' }
    & $python -m pip install --no-deps --no-build-isolation -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed' }
    & $python scripts\verify_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'CUDA or six-test verification failed' }
} finally {
    Pop-Location
}
