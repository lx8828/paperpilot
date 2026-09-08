$ErrorActionPreference = 'Continue'
# _l2hard_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_l2hard.py *> qa/recall/l2hard_run.log
Write-Host "l2hard all done exit=$LASTEXITCODE"
