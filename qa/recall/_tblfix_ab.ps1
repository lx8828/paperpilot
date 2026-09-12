# P1+P3 同日配对 A/B：v1（旧表表示 + 旧 prompt）vs new（P1+P3）
# 两臂都用配额 0、α 默认（= 生产等价配置）；表通道开。
cd f:/paperpilot

$env:PAPERPILOT_TABLE_V1 = "1"
$env:PP_OUT = "qa/recall/tblfix_v1_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix_v1.log

Remove-Item Env:\PAPERPILOT_TABLE_V1 -ErrorAction SilentlyContinue
$env:PP_OUT = "qa/recall/tblfix_new_20260912.json"
uv run python -u qa/recall/_qasper_tbl_exp.py *> qa/recall/_tblfix_new.log
