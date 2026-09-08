import json
ab = json.load(open("qa/recall/ab_noL2_result.json", encoding="utf-8"))
items = {it["qid"]: it for it in json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))["items"]}

# A 臂 level==L2 = judge_l1 不够但有 target → 真实进了 L2 扩窗并作答的题
l2 = [r for r in ab if r["A"]["level"] == "L2"]
print(f"A 臂最终在 L2 作答的题: {len(l2)}")
for r in sorted(l2, key=lambda x: items[x["qid"]]["group"]):
    A, B = r["A"], r["B"]
    mark = "L2赢/L3输" if (A["pass"] and not B["pass"]) else \
           ("L2输/L3赢" if (not A["pass"] and B["pass"]) else
            ("双过" if (A["pass"] and B["pass"]) else "双挂"))
    print(f"  {items[r['qid']]['group']:6} {r['qid'][:12]} A={A['score']}({'过' if A['pass'] else '挂'},{A['level']},{A['calls']}c) "
          f"B={B['score']}({'过' if B['pass'] else '挂'},{B['level']},{B['calls']}c)  {mark}")

print()
for g in ("hard", "normal"):
    sub = [r for r in l2 if items[r["qid"]]["group"] == g]
    if not sub:
        continue
    a_pass = sum(1 for r in sub if r["A"]["pass"])
    b_pass = sum(1 for r in sub if r["B"]["pass"])
    l2w = sum(1 for r in sub if r["A"]["pass"] and not r["B"]["pass"])
    l2l = sum(1 for r in sub if not r["A"]["pass"] and r["B"]["pass"])
    ac = round(sum(r["A"]["calls"] for r in sub) / len(sub), 1)
    bc = round(sum(r["B"]["calls"] for r in sub) / len(sub), 1)
    print(f"{g}: n={len(sub)}  A(L2作答) pass {a_pass}  vs  B(砍L2直L3) pass {b_pass}  "
          f"| L2净赢 {l2w} / L2净亏 {l2l}  | 调用/题 A={ac} B={bc}")
