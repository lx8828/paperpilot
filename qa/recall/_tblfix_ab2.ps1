# P1+P3 配对 A/B 第 2 轮（两臂各用独立 cvec 目录，避免来回覆盖重建）
cd f:/paperpilot

# 先跑 new 臂：缓存已从主目录复制过来（new 变体）→ 应零编码
$env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_new"
$env:PP_OUT = "qa/recall/tblfix2_new_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix2_new.log

# 再跑 v1 臂：独立目录 → 需要重建一次（之后各轮都不再重建）
Remove-Item Env:\PAPERPILOT_CHUNK_VIEW_DIR -ErrorAction SilentlyContinue
$env:PAPERPILOT_TABLE_V1 = "1"
$env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_v1"
$env:PP_OUT = "qa/recall/tblfix2_v1_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix2_v1.log
