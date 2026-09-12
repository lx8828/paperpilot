# P1+P3 配对 A/B 第 3 轮（两臂缓存均已就绪 → 全程零编码）
cd f:/paperpilot

# 顺序与第 2 轮相反（v1 先跑），缓解与上游漂移的耦合
$env:PAPERPILOT_TABLE_V1 = "1"
$env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_v1"
$env:PP_OUT = "qa/recall/tblfix3_v1_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix3_v1.log

Remove-Item Env:\PAPERPILOT_TABLE_V1 -ErrorAction SilentlyContinue
Remove-Item Env:\PAPERPILOT_CHUNK_VIEW_DIR -ErrorAction SilentlyContinue
$env:PAPERPILOT_CHUNK_VIEW_DIR = "assets/artifacts/out_views_new"
$env:PP_OUT = "qa/recall/tblfix3_new_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix3_new.log
