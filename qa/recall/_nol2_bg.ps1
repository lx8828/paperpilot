$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_noL2.py --n-hard 25 --n-normal 25
Write-Host "noL2 exit=$LASTEXITCODE"
