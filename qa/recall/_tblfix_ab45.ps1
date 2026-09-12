# P1+P3 配对 A/B 第 4、5 轮（零编码）
cd f:/paperpilot
foreach ($r in 4,5) {
  $env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_new"
  $env:PP_OUT = "qa/recall/tblfix${r}_new_20260912.json"
  uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix${r}_new.log

  Remove-Item Env:\PAPERPILOT_CHUNK_VIEW_DIR -ErrorAction SilentlyContinue
  $env:PAPERPILOT_TABLE_V1 = "1"
  $env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_v1"
  $env:PP_OUT = "qa/recall/tblfix${r}_v1_20260912.json"
  uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix${r}_v1.log

  Remove-Item Env:\PAPERPILOT_TABLE_V1 -ErrorAction SilentlyContinue
  Remove-Item Env:\PAPERPILOT_CHUNK_VIEW_DIR -ErrorAction SilentlyContinue
}
