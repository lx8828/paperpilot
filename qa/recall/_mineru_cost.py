"""临时：MinerU hard 自测的 token 花费统计。"""
import json
from pathlib import Path

r = json.loads(Path("qa/recall/mineru_hard_result.json").read_text(encoding="utf-8"))
p = sum(x.get("prompt", 0) for x in r)
c = sum(x.get("completion", 0) for x in r)
print(f"题 {len(r)} | prompt {p:,} | completion {c:,}")
print(f"  高峰档(2/8元/M): {p/1e6*2 + c/1e6*8:.2f} 元")
print(f"  空闲档(1/4元/M): {p/1e6 + c/1e6*4:.2f} 元")
print(f"  题均 prompt {p/len(r):.0f} | calls {sum(x.get('calls',0) for x in r)/len(r):.1f}")
lvl = {}
for x in r:
    lvl[x["level"]] = lvl.get(x["level"], 0) + 1
act = {}
for x in r:
    act[str(x["action"])] = act.get(str(x["action"]), 0) + 1
print("level:", lvl, "| action:", act)
