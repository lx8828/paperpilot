$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_spike_v2.py --n-hard 50 --n-unans 10
Write-Host "spike exit=$LASTEXITCODE"
