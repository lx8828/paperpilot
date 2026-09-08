$ErrorActionPreference = 'Continue'
# _v3regress_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_v3regress.py --n 250 *> qa/recall/v3regress_run.log
Write-Host "v3regress all done exit=$LASTEXITCODE"
