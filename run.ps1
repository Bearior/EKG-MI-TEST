param(
    [int]$Seed = 42,
    [int]$Bootstrap = 1000
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& ./.venv/Scripts/python.exe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Could not provision pip in the environment.' }
& ./.venv/Scripts/python.exe -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& ./.venv/Scripts/python.exe -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& ./.venv/Scripts/python.exe -m mi_lab.download --data-dir data
if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed.' }
& ./.venv/Scripts/python.exe -m mi_lab.experiment --seed $Seed --bootstrap $Bootstrap
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed.' }
