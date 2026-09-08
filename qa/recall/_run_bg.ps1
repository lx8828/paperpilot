$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
# 阶段1：预览（只用已有缓存论文，快）
uv run python cli/run_retrieval_eval.py --set qa/recall/recall_set_v1.json --skip-missing --out qa/recall/RECALL_PREVIEW_20260908.md
Write-Host "preview exit=$LASTEXITCODE"
# 阶段2：预编码缺失论文（一次性建 cvec）
uv run python qa/recall/_prewarm.py
Write-Host "prewarm exit=$LASTEXITCODE"
# 阶段3：全量基准
uv run python cli/run_retrieval_eval.py --set qa/recall/recall_set_v1.json --out qa/recall/RECALL_BASELINE_20260908.md
Write-Host "baseline exit=$LASTEXITCODE"
