import json
recs = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))
n = len(recs)
print(f"全链样本 n={n}（hard {sum(1 for r in recs if r['grp']=='hard')} / "
      f"normal {sum(1 for r in recs if r['grp']=='normal')}）")


def agg(recs, key):
    return {"pass": sum(1 for r in recs if r[key]["pass"]),
            "prompt": sum(r[key]["prompt"] for r in recs) / max(len(recs), 1),
            "calls": sum(r[key]["calls"] for r in recs) / max(len(recs), 1)}


for name, key in [("A(现状 v2 漏斗)", "A"), ("V(v3 纯两级)", "V")]:
    a = agg(recs, key)
    print(f"{name}: pass {a['pass']}/{n}  prompt/题 {a['prompt']:.0f}  calls/题 {a['calls']:.1f}")

# 交叉
aw = [r for r in recs if r["A"]["pass"] and not r["V"]["pass"]]
vw = [r for r in recs if not r["A"]["pass"] and r["V"]["pass"]]
both = [r for r in recs if r["A"]["pass"] and r["V"]["pass"]]
none = [r for r in recs if not r["A"]["pass"] and not r["V"]["pass"]]
print(f"交叉：A过V挂 {len(aw)} / A挂V过 {len(vw)} / 双过 {len(both)} / 双挂 {len(none)}")
print("A过V挂（v3 掉的）：")
for r in aw:
    print("  {} {} A={}({}) V={}({})".format(r["qid"][:12], r["grp"],
        "过" if r["A"]["pass"] else "挂", r["A"]["score"] if r["A"]["pass"] else r["A"]["level"],
        "过" if r["V"]["pass"] else "挂", r["V"]["score"] if r["V"]["pass"] else r["V"]["level"]))
print("A挂V过（v3 救的）：")
for r in vw:
    print("  {} {} A={}({}) V={}({})".format(r["qid"][:12], r["grp"],
        "过" if r["A"]["pass"] else "挂", r["A"]["score"] if r["A"]["pass"] else r["A"]["level"],
        "过" if r["V"]["pass"] else "挂", r["V"]["score"] if r["V"]["pass"] else r["V"]["level"]))
# 按 level 切片
for name, key in [("A", "A"), ("V", "V")]:
    from collections import Counter
    lv = Counter(r[key]["level"] for r in recs)
    print(f"{name} 终点层分布:", dict(lv))
# easy(实际 L0/L1 直答) 占比
l0a = sum(1 for r in recs if r["A"]["level"].startswith("L0"))
l1a = sum(1 for r in recs if r["A"]["level"].startswith("L1"))
print(f"A 中 L0/L1 直答题: {l0a}/{l1a}（这些在 V 里走了 L3 吗）")
for lv in ("L0", "L1"):
    sub = [r for r in recs if r["A"]["level"].startswith(lv)]
    if sub:
        vv = agg(sub, "V")
        print(f"  A-{lv} 直答 {len(sub)} 题 → V pass {vv['pass']}/{len(sub)}  "
              f"prompt {vv['prompt']:.0f}  calls {vv['calls']:.1f}")
