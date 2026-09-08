$ErrorActionPreference = 'Continue'
# _l2target_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_l2target.py --n-hard 35 --n-normal 45 *> qa/recall/l2target_run.log
Write-Host "l2target all done exit=$LASTEXITCODE"
