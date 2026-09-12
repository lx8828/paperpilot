$ErrorActionPreference = 'Continue'
# _v3c14_bg.ps1 在 <root>/qa/recall/，上两级为项目根
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
uv run python qa/recall/_v3c14_regress.py *> qa/recall/v3c14_run.log
Write-Host "v3c14 all done exit=$LASTEXITCODE"
