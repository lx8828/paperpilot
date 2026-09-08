$ErrorActionPreference = 'Continue'
# _multiq_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_run_multiq.py *> qa/recall/multiq_run.log
Write-Host "multiq all done exit=$LASTEXITCODE"
