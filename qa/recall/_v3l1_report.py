import json
recs = json.load(open("qa/recall/ab_v3l1_result.json", encoding="utf-8"))
n = len(recs)
ps = sum(1 for r in recs if r["pass"])
pt = sum(r["prompt"] for r in recs) / n
ca = sum(r["calls"] for r in recs) / n
print(f"V1(v3+L1) n={n}  pass {ps}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}")
print(f"对照(同100题): A(现状v2)=60  V0(纯两级)=64")
from collections import Counter
print("V1 终点层:", dict(Counter(r["level"] for r in recs)))
print()
buck = {
    "A过V0挂(v3丢的8, L1要捞)": [r for r in recs if r["prevA"] and not r["prevV0"]],
    "A挂V0过(V0救的12)": [r for r in recs if not r["prevA"] and r["prevV0"]],
    "双过": [r for r in recs if r["prevA"] and r["prevV0"]],
    "双挂": [r for r in recs if not r["prevA"] and not r["prevV0"]],
}
for name, sub in buck.items():
    if not sub:
        continue
    s = sum(1 for r in sub if r["pass"])
    print(f"  {name}: n={len(sub)} V1 pass {s}")
saved = [r for r in recs if (r["prevA"] and not r["prevV0"]) and r["pass"]]
print(f"\nV1 捞回(A过V0挂但现在过) {len(saved)}:", [r["qid"][:10] + "/" + r["level"] for r in saved])
hurt = [r for r in recs if (not r["prevA"] and r["prevV0"]) and not r["pass"]]
print(f"V1 弄丢(V0过现在挂) {len(hurt)}:", [r["qid"][:10] + "/" + r["level"] for r in hurt])
print()
print("A过V0挂 8 题逐题 V1：")
for r in recs:
    if r["prevA"] and not r["prevV0"]:
        print("  {} {} A过 V0挂 V1={}{}({}c)".format(
            r["qid"][:12], r["grp"], "过" if r["pass"] else "挂", r["score"], r["calls"]))
