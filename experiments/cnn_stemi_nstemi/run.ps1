$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repo
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Create the repository .venv first; see README.md.' }
& $python -m mi_lab.cnn_experiment --epochs 15 --batch-size 64
if ($LASTEXITCODE -ne 0) { throw 'CNN training failed.' }
