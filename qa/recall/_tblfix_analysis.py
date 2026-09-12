"""P1+P3 同日配对 A/B 的归因分析。"""
import json
from collections import Counter
from pathlib import Path

R = Path("qa/recall")
v1 = {x["qid"]: x for x in json.loads((R / "tblfix_v1_20260912.json").read_text(encoding="utf-8"))}
new = {x["qid"]: x for x in json.loads((R / "tblfix_new_20260912.json").read_text(encoding="utf-8"))}


def tbl_hit(rec):
    """该题的上下文里是否有表块（xtbl-*，排除公式）。"""
    ids = list(rec.get("entry_ids") or [])
    return any(str(c).startswith("xtbl-") for c in ids)


def cited_tbl(rec):
    out = []
    for c in rec.get("cites") or []:
        ref = str(c.get("ref") or "")
        cid = ref.split("#")[-1] if "#" in ref else ref
        if str(cid).startswith("xtbl-"):
            out.append(cid)
    return out


p1 = sum(1 for x in v1.values() if x["pass"])
p2 = sum(1 for x in new.values() if x["pass"])
s1 = sum(x["score"] or 0 for x in v1.values())
s2 = sum(x["score"] or 0 for x in new.values())
print(f"v1  pass {p1}/36（总分 {s1}） | new pass {p2}/36（总分 {s2}）")
gain = [q for q in new if new[q]["pass"] and not v1[q]["pass"]]
loss = [q for q in new if v1[q]["pass"] and not new[q]["pass"]]
print(f"翻绿 {len(gain)} / 翻红 {len(loss)}（churn {len(gain)+len(loss)}）\n")

print("| 方向 | qid | 论文 | 分数 v1→new | v1 有表 | new 有表 | v1 引用表 | new 引用表 |")
print("|---|---|---|---|---|---|---|---|")
for q in gain + loss:
    a, b = v1[q], new[q]
    print(f"| {'翻绿' if q in gain else '翻红'} | {q[:8]} | {a['pid']} | {a['score']}→{b['score']} "
          f"| {'Y' if tbl_hit(a) else '-'} | {'Y' if tbl_hit(b) else '-'} "
          f"| {len(cited_tbl(a))} | {len(cited_tbl(b))} |")

print()
print("分层：按【new 臂上下文里有没有表块】看 pass")
for lab, cond in (("有表块", lambda r: tbl_hit(r)), ("无表块", lambda r: not tbl_hit(r))):
    qs = [q for q in new if cond(new[q])]
    if not qs:
        continue
    a = sum(1 for q in qs if v1[q]["pass"])
    b = sum(1 for q in qs if new[q]["pass"])
    print(f"  {lab}: n={len(qs)} | v1 {a} → new {b} ({b-a:+d})")

print()
print("表块被引用率（引用次数>0 的题占比）")
for lab, D in (("v1", v1), ("new", new)):
    qs = [q for q in D if tbl_hit(D[q])]
    c = sum(1 for q in qs if cited_tbl(D[q]))
    print(f"  {lab}: 有表块的题 {len(qs)}，其中引用了表块 {c}")

print()
print("分数变化分布（new - v1）:", dict(sorted(Counter(
    (new[q]["score"] or 0) - (v1[q]["score"] or 0) for q in v1).items())))
print()
print("重点题 75b69eef：")
q = next((k for k in v1 if k.startswith("75b69eef")), None)
if q:
    for lab, D in (("v1 ", v1), ("new", new)):
        r = D[q]
        print(f"  {lab} score={r['score']} | 引用表块 {cited_tbl(r)} | facts {r.get('n_facts')} 条")
        print(f"       {(r.get('answer') or '')[:220]}")
