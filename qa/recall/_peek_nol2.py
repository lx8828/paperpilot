import json

d = json.load(open("qa/recall/ab_noL2_result.json", encoding="utf-8"))
for g in ("hard", "normal", "all"):
    grp = [r for r in d if g == "all" or r["group"] == g]
    pa = sum(1 for r in grp if r["A"]["pass"])
    pb = sum(1 for r in grp if r["B"]["pass"])
    ca = round(sum(r["A"]["calls"] for r in grp) / max(len(grp), 1), 2)
    cb = round(sum(r["B"]["calls"] for r in grp) / max(len(grp), 1), 2)
    # 层级分布（现状 A 中 level 落在 L2 的题数）
    la2 = sum(1 for r in grp if r["A"]["level"] == "L2")
    print(f"{g} n={len(grp)}: A(现状) pass {pa} | B(砍L2) pass {pb} | Δ={pb-pa} | "
          f"A中L2收口 {la2} | 调用/题 A={ca} B={cb}")
# 交叉表：A vs B 四种组合
print("\n交叉（A→B）:")
for a in (True, False):
    for b in (True, False):
        n = sum(1 for r in d if r["A"]["pass"] == a and r["B"]["pass"] == b)
        print(f"  A {'pass' if a else 'fail'} → B {'pass' if b else 'fail'}: {n}")
