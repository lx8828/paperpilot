# 逐题验证 14 道翻绿题的溯源（是否引用到 xtbl-* 表格块）
cd f:/paperpilot
uv run python -u qa/recall/_verify_tbl_cites.py *> qa/recall/_verify_cites.log
