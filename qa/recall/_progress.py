"""临时：查看 MinerU hard 自测进度。"""
import json
from pathlib import Path

p = Path("qa/recall/mineru_hard_result.json")
r = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
print(f"已完成 {len(r)}/30")
for x in r:
    print(f"  {x['qid']:16} {x['pid']:14} score={x['score']} {x['level']}/{x['action']}")
