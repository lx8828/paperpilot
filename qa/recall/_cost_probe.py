"""临时：估算评测成本——读历史 run 的实际 token/calls，算 250 题单臂/双臂的用量。"""
import json
import os

FILES = {
    "ab_v3regress(250, A+V)": "qa/recall/ab_v3regress_result.json",
    "ab_v3base(100, A+V)": "qa/recall/ab_v3base_result.json",
    "ab_nol3j(100, 单臂)": "qa/recall/ab_nol3j_result.json",
    "ab_v3c14(100, 单臂)": "qa/recall/ab_v3c14_result.json",
}

for name, f in FILES.items():
    if not os.path.exists(f):
        continue
    r = json.load(open(f, encoding="utf-8"))
    print("=" * 74)
    print(f"{name}  n={len(r)}")
    # 统计每个臂的 prompt/calls
    keys = [k for k in ("A", "V") if k in r[0]]
    for k in keys:
        ptsum = sum((x[k] or {}).get("prompt", 0) or 0 for x in r)
        csum = sum((x[k] or {}).get("calls", 0) or 0 for x in r)
        print(f"  {k}: prompt总 {ptsum:,} | 题均 {ptsum/len(r):.0f} | calls总 {csum:,} | 题均 {csum/len(r):.1f}")
    if "prompt" in r[0]:
        ptsum = sum(x.get("prompt", 0) or 0 for x in r)
        csum = sum(x.get("calls", 0) or 0 for x in r)
        print(f"  (单臂) prompt总 {ptsum:,} | 题均 {ptsum/len(r):.0f} | calls总 {csum:,} | 题均 {csum/len(r):.1f}")

print("=" * 74)
print("模型配置（.env / llm.py）:")
for p in (".env", "src/paperpilot/tools/llm.py"):
    if os.path.exists(p):
        txt = open(p, encoding="utf-8", errors="replace").read()
        for line in txt.splitlines():
            if "MODEL" in line.upper() and ("=" in line) and ("PAPERPILOT" in line or "model" in line.lower()):
                print("  ", line.strip()[:120])
