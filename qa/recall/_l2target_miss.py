import json
from collections import Counter
recs = json.load(open("qa/recall/ab_l2target_result.json", encoding="utf-8"))

miss = [r for r in recs if r["l2_target"] and not r["A"]["pass"] and r["B"]["pass"]]
win = [r for r in recs if r["l2_target"] and r["A"]["pass"] and not r["B"]["pass"]]
both = [r for r in recs if r["l2_target"] and r["A"]["pass"] and r["B"]["pass"]]
print("L2目标群: A漏B过(净亏)", len(miss), "| A过B漏(净赢)", len(win), "| 双过", len(both))
print("漏题 A 终点 level:", Counter(r["A"]["level"] for r in miss))
print("漏题 B 终点 level:", Counter(r["B"]["level"] for r in miss))
print()
print("漏题（A挂B过）明细：")
for r in sorted(miss, key=lambda x: x["grp"]):
    print("  {} {} A={}({},{},{}) B={}({},{},{})".format(
        r["qid"][:12], r["grp"],
        r["A"]["score"], r["A"]["level"], r["A"]["calls"], r["A"]["prompt"],
        r["B"]["score"], r["B"]["level"], r["B"]["calls"], r["B"]["prompt"]))
print()
print("双过题里 A 终点 level:", Counter(r["A"]["level"] for r in both))
print("双过题 token 比（A/B）：", round(sum(r["A"]["prompt"] for r in both) /
      max(1, sum(r["B"]["prompt"] for r in both)), 2))
print()
print("净赢题（A过B漏，L2 独有的价值）明细：")
for r in sorted(win, key=lambda x: x["grp"]):
    print("  {} {} A={}({}) B={}({})".format(
        r["qid"][:12], r["grp"], r["A"]["score"], r["A"]["level"],
        r["B"]["score"], r["B"]["level"]))
