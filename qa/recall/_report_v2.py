import json
from collections import Counter

d = json.load(open("qa/recall/spike_v2_result.json", encoding="utf-8"))
hd = [r for r in d if r["kind"] == "hard"]
na = [r for r in d if r["kind"] == "unans"]


def stats(grp, name):
    p1 = sum(1 for r in grp if (r["v1"]["score"] or 0) >= 4)
    p2 = sum(1 for r in grp if (r["v2"]["score"] or 0) >= 4)
    rescue = sum(1 for r in grp if (r["v1"]["score"] or 0) < 4 and (r["v2"]["score"] or 0) >= 4)
    regr = sum(1 for r in grp if (r["v1"]["score"] or 0) >= 4 and (r["v2"]["score"] or 0) < 4)
    both = sum(1 for r in grp if (r["v1"]["score"] or 0) >= 4 and (r["v2"]["score"] or 0) >= 4)
    s1 = [r["v1"]["score"] for r in grp]
    s2 = [r["v2"]["score"] for r in grp]
    a1 = sum(1 for r in grp if (r["v1"]["score"] or 0) > 0)
    a2 = sum(1 for r in grp if (r["v2"]["score"] or 0) > 0)
    c1 = round(sum(r["v1"]["calls"] for r in grp) / len(grp), 2)
    c2 = round(sum(r["v2"]["calls"] for r in grp) / len(grp), 2)
    print(f"{name} n={len(grp)}: V1 pass {p1} | V2 pass {p2} | 救回 {rescue} | 回归 {regr} | 同过 {both}")
    print(f"   V1 有分(>0) {a1} 均分 {round(sum(x for x in s1 if x)/max(a1,1),2)} | "
          f"V2 有分 {a2} 均分 {round(sum(x for x in s2 if x)/max(a2,1),2)}")
    print(f"   每题 LLM 调用 V1={c1} V2={c2}")
    print(f"   V2 status: {dict(Counter(r['v2']['state'] for r in grp))}")
    # V1 unknown 中被 V2 救回
    unk = [r for r in grp if r["v1"]["state"] == "unknown"]
    print(f"   V1 unknown 子集 n={len(unk)}，其中 V2 pass={sum(1 for r in unk if (r['v2']['score'] or 0)>=4)}")


stats(hd, "HARD")
stats(na, "UNANS")
