import json
import re
recs = json.load(open("qa/recall/ab_multiq_result.json", encoding="utf-8"))
n = len(recs)
ps = sum(1 for r in recs if r["pass"])
print(f"多方法对比题 V0 n={n}  pass {ps} ({ps/n:.0%})  | 参照全链 64%")
print(f"score 分布:", {s: sum(1 for r in recs if r['score'] == s) for s in sorted(set(r['score'] for r in recs))})

grps = {
    "多对象列举(what models/baselines compared)": r"(what|which) (models|baselines|methods|algorithms|approaches|solutions|work|attention) |compared against|compared to|compares? to$",
    "两两差异(difference between A and B)": r"difference|differ",
    "量化对比(how much/by how much/faster/higher)": r"how much|by how much|how (faster|higher|bigger|better|large)|improvement|better (performance|than)",
    "优劣判断(which works better)": r"which (works|one|of|language|attention|method)|works better",
}
for name, pat in grps.items():
    sub = [r for r in recs if re.search(pat, r["question"], re.I)]
    if sub:
        s = sum(1 for r in sub if r["pass"])
        print(f"  {name}: n={len(sub)} pass {s} ({s/len(sub):.0%})")

print()
print("=" * 80)
print("失败题（pass=False）逐条：")
fails = [r for r in recs if not r["pass"]]
for r in fails:
    print("-" * 80)
    print(f"score={r['score']} | Q: {r['question'][:160]}")
    print(f"  gold: {(r['gold'] or '')[:200]}")
    print(f"  ans:  {(r['answer'] or '')[:300]}")
