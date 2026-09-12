# P1+P3 配对 A/B 第 6、7、8 轮（零编码；交替臂序以减轻上游漂移的耦合）
cd f:/paperpilot
foreach ($r in 6,7,8) {
  if ($r % 2 -eq 0) {
    # 偶数轮：先 new
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
  } else {
    # 奇数轮：先 v1
    $env:PAPERPILOT_TABLE_V1 = "1"
    $env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_v1"
    $env:PP_OUT = "qa/recall/tblfix${r}_v1_20260912.json"
    uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix${r}_v1.log
    Remove-Item Env:\PAPERPILOT_TABLE_V1 -ErrorAction SilentlyContinue

    $env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_new"
    $env:PP_OUT = "qa/recall/tblfix${r}_new_20260912.json"
    uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix${r}_new.log
    Remove-Item Env:\PAPERPILOT_CHUNK_VIEW_DIR -ErrorAction SilentlyContinue
  }
}
