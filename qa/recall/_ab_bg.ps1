$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_answer_unknown.py
Write-Host "ab exit=$LASTEXITCODE"
