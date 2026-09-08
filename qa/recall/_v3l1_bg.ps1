$ErrorActionPreference = 'Continue'
# _v3l1_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_ab_v3l1.py *> qa/recall/v3l1_run.log
Write-Host "v3l1 all done exit=$LASTEXITCODE"
