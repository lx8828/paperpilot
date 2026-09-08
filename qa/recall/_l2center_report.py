import json
recs = json.load(open("qa/recall/ab_l2center_result.json", encoding="utf-8"))
n = len(recs)
ps = sum(1 for r in recs if r["pass"])
pt = sum(r["prompt"] for r in recs) / n
ca = sum(r["calls"] for r in recs) / n
nc = sum(1 for r in recs if (r.get("n_centers") or 0) > 2)
print(f"multicent(多候选圆心) n={n}  pass {ps}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}  "
      f"| 候选圆心>2 的题 {nc}")
print(f"对照: base(现状)=40  strict(严判够)=44  B(直L3)=44")
print()
buck = {
    "净亏(A挂B过)": [r for r in recs if not r["prevA"] and r["prevB"]],
    "净赢(A过B挂)": [r for r in recs if r["prevA"] and not r["prevB"]],
    "双过": [r for r in recs if r["prevA"] and r["prevB"]],
    "双挂": [r for r in recs if not r["prevA"] and not r["prevB"]],
}
for name, sub in buck.items():
    if not sub:
        continue
    s = sum(1 for r in sub if r["pass"])
    print(f"  {name}: n={len(sub)} mc pass {s} "
          f"(历史 A pass {sum(1 for r in sub if r['prevA'])} / B pass {sum(1 for r in sub if r['prevB'])})")
saved = [r for r in recs if not r["prevA"] and r["prevB"] and r["pass"]]
hurt = [r for r in recs if r["prevA"] and not r["pass"]]
print()
print(f"净亏救回 {len(saved)} 题:", [r["qid"][:10] + "/" + r["level"] for r in saved])
print(f"误伤(曾A过) {len(hurt)} 题:", [r["qid"][:10] + "/" + r["level"] for r in hurt])
print()
print("净亏 10 题逐题：")
for r in recs:
    if not r["prevA"] and r["prevB"]:
        print("  {} {} mc={}{}({}c,{}) prevA={} prevB={}".format(
            r["qid"][:12], r["grp"], "过" if r["pass"] else "挂", r["score"],
            r["calls"], r["n_centers"], "过" if r["prevA"] else "挂",
            "过" if r["prevB"] else "挂"))
