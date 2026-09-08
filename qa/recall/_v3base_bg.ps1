$ErrorActionPreference = 'Continue'
# _v3base_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_v3base.py --n 100 *> qa/recall/v3base_run.log
Write-Host "v3base all done exit=$LASTEXITCODE"
