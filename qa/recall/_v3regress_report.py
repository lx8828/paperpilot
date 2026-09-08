import json
recs = json.load(open("qa/recall/ab_v3regress_result.json", encoding="utf-8"))
n = len(recs)
a = {"pass": sum(1 for r in recs if r["A"]["pass"]),
     "prompt": sum(r["A"]["prompt"] for r in recs) / n,
     "calls": sum(r["A"]["calls"] for r in recs) / n}
v = {"pass": sum(1 for r in recs if r["V"]["pass"]),
     "prompt": sum(r["V"]["prompt"] for r in recs) / n,
     "calls": sum(r["V"]["calls"] for r in recs) / n}
off = sum(1 for r in recs if r["official_pass"])
print(f"定稿回归 n={n}")
print(f"A(现状v2): pass {a['pass']} ({a['pass']/n:.1%})  prompt {a['prompt']:.0f}  calls {a['calls']:.1f}")
print(f"V0(v3):   pass {v['pass']} ({v['pass']/n:.1%})  prompt {v['prompt']:.0f}  calls {v['calls']:.1f}")
print(f"差 V0-A: {v['pass']-a['pass']} pt")
print(f"官方口径参照(该子集当时): {off} ({off/n:.1%})")
aw = [r for r in recs if r["A"]["pass"] and not r["V"]["pass"]]
vw = [r for r in recs if not r["A"]["pass"] and r["V"]["pass"]]
print(f"A过V挂 {len(aw)} / A挂V过 {len(vw)}")
print()
print("A过V挂（v3 掉的，需看模式）：")
for r in aw:
    print("  {} A={}({}) V={}({})".format(r["qid"][:12], r["A"]["score"], r["A"]["level"],
          r["V"]["score"], r["V"]["level"]))
print("A挂V过（v3 救的）：")
for r in vw:
    print("  {} A={}({}) V={}({})".format(r["qid"][:12], r["A"]["score"], r["A"]["level"],
          r["V"]["score"], r["V"]["level"]))
# 与官方一致的稳定性（裁判复现度）
agree = sum(1 for r in recs if r["A"]["pass"] == r["official_pass"])
print(f"\n本run A pass 与官方 status 一致率: {agree}/{n} ({agree/n:.0%}) —— 反映裁判/模型漂移")
