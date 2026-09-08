"""从 ab_l2target_result.json 出分层报告（含 L2 目标题群明细）。"""
from __future__ import annotations
import io
import json
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

recs = json.load(open("qa/recall/ab_l2target_result.json", encoding="utf-8"))


def line(r):
    return "  {} {} A={}{}({}) {}c/{}tok B={}{}({}) {}c/{}tok  tgt={} used={}".format(
        r["qid"][:12], r["grp"][:1],
        "过" if r["A"]["pass"] else "挂", r["A"]["score"], r["A"]["level"],
        r["A"]["calls"], r["A"]["prompt"],
        "过" if r["B"]["pass"] else "挂", r["B"]["score"], r["B"]["level"],
        r["B"]["calls"], r["B"]["prompt"],
        int(r["l2_target"]), int(r["l2_used"]))


def grp(pred, name):
    sub = [r for r in recs if pred(r)]
    if not sub:
        print(f"{name}: n=0")
        return []
    a = sum(1 for r in sub if r["A"]["pass"])
    b = sum(1 for r in sub if r["B"]["pass"])
    l2w = [r for r in sub if r["A"]["pass"] and not r["B"]["pass"]]
    l2l = [r for r in sub if not r["A"]["pass"] and r["B"]["pass"]]
    at = sum(r["A"]["prompt"] for r in sub) / len(sub)
    bt = sum(r["B"]["prompt"] for r in sub) / len(sub)
    ac = sum(r["A"]["calls"] for r in sub) / len(sub)
    bc = sum(r["B"]["calls"] for r in sub) / len(sub)
    print(f"{name}: n={len(sub)}  A(扩窗)pass {a}  B(直L3)pass {b}  "
          f"净赢{len(l2w)}/净亏{len(l2l)}  | prompt/题 A={at:.0f} B={bt:.0f} "
          f"| calls/题 A={ac:.1f} B={bc:.1f}")
    return sub


print(f"总样本 {len(recs)}: hard {sum(1 for r in recs if r['grp']=='hard')} / "
      f"normal {sum(1 for r in recs if r['grp']=='normal')}  "
      f"| err A={sum(1 for r in recs if r['A']['level']=='err')} "
      f"B={sum(1 for r in recs if r['B']['level']=='err')}")
err = [r for r in recs if r["A"]["level"] == "err" or r["B"]["level"] == "err"]
for r in err:
    print("  ERR:", line(r))
eff = [r for r in recs if r["A"]["level"] != "err" and r["B"]["level"] != "err"]
print(f"有效样本（剔除双 err）: {len(eff)}")
print("-" * 70)
grp(lambda r: True, "全体(有效)")
tgt = grp(lambda r: r["l2_target"], "L2目标题群(judge_l1不够+有target)")
used = grp(lambda r: r["l2_used"], "真走L2(route含L2)")
grp(lambda r: r["grp"] == "hard", "hard 子集")
grp(lambda r: r["grp"] == "normal", "normal 子集")
if tgt:
    print("-" * 70)
    print("L2 目标题群逐题（A 扩窗 vs B 直L3）：")
    for r in tgt:
        print(line(r))
if used:
    print("-" * 70)
    print("真走 L2 的题逐题：")
    for r in used:
        print(line(r))
print("-" * 70)
print("judge_l1 enough A/B 不一致:", sum(1 for r in eff if r["A"]["j1_enough"] != r["B"]["j1_enough"]))
print("B 目标题里 A/B 不一致:", sum(1 for r in eff if r["l2_target"] and r["A"]["j1_enough"] != r["B"]["j1_enough"]))
