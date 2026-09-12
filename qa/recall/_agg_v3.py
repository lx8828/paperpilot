"""临时：聚合 v3 系列端到端对拍成绩（只读 json）。"""
import json
import os
from collections import Counter

FILES = [
    "qa/recall/ab_v3regress_result.json",
    "qa/recall/ab_v3c14_result.json",
    "qa/recall/ab_v3l1_result.json",
    "qa/recall/ab_nol3j_result.json",
]


def rate(r, k):
    v = r[0].get(k)
    if isinstance(v, bool):
        p = sum(1 for x in r if x.get(k))
    elif isinstance(v, dict):
        p = sum(1 for x in r if (x.get(k) or {}).get("pass"))
    elif isinstance(v, (int, float)):
        p = sum(1 for x in r if (x.get(k) or 0) >= 4)
        avg = sum(x.get(k) or 0 for x in r) / len(r)
        return f"{p/len(r)*100:5.1f}% (n={len(r)})  avg={avg:.2f}"
    else:
        return None
    return f"{p/len(r)*100:5.1f}% (n={len(r)})"


for f in FILES:
    if not os.path.exists(f):
        print(f, "missing")
        continue
    r = json.load(open(f, encoding="utf-8"))
    if not isinstance(r, list) or not r:
        print(f, "empty")
        continue
    print("=" * 78)
    print(os.path.basename(f), "n =", len(r))
    print("  keys:", list(r[0].keys()))
    for k in r[0].keys():
        if k in ("qid", "pid", "grp", "level", "route", "action", "calls"):
            continue
        out = rate(r, k)
        if out:
            print(f"  {k:14} {out}")
    if "grp" in r[0]:
        for g in sorted({x.get("grp") for x in r}):
            sub = [x for x in r if x.get("grp") == g]
            line = f"  [{g}] n={len(sub)}"
            for k in ("A", "V", "official_pass", "oldV_pass", "pass"):
                if k in r[0]:
                    o = rate(sub, k)
                    if o:
                        line += f"  {k}={o.split(' ')[0]}"
            print(line)
    if "level" in r[0]:
        print("  level:", dict(Counter(x.get("level", "") for x in r)))
    if "action" in r[0]:
        print("  action:", dict(Counter(str(x.get("action", "")) for x in r)))
