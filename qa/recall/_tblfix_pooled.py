"""P1+P3 配对 A/B：多轮合并分析（自动发现轮次）。

扫描 qa/recall/tblfix*_v1_20260912.json 与 tblfix*_new_20260912.json 成对读取。
"""
import json
from collections import Counter
from pathlib import Path

R = Path("qa/recall")
pairs = {}
for f in sorted(R.glob("tblfix*_20260912.json")):
    stem = f.stem                              # tblfix2_v1_20260912
    pre, arm, _ = stem.rsplit("_", 2)          # pre=tblfix2, arm=v1|new
    if arm not in ("v1", "new"):
        continue
    pairs.setdefault(pre, {})[arm] = {x["qid"]: x for x in json.loads(f.read_text(encoding="utf-8"))}
data = [(pre, d["v1"], d["new"]) for pre, d in sorted(pairs.items())
        if "v1" in d and "new" in d and len(d["v1"]) == 36 and len(d["new"]) == 36]
print(f"完整轮次 {len(data)}: {[x[0] for x in data]}")


def has_tbl(rec):
    return [c for c in (rec.get("entry_ids") or []) if str(c).startswith("xtbl-")]


def cited_tbl(rec):
    return [c for c in (rec.get("cites") or [])
            if str(c.get("ref", "")).split("#")[-1].startswith("xtbl-")]


for lab, v, n in data:
    print(f"  {lab}: v1 {sum(1 for x in v.values() if x['pass'])}/36 | "
          f"new {sum(1 for x in n.values() if x['pass'])}/36")
pv = sum(sum(1 for x in v.values() if x["pass"]) for _, v, _ in data) / len(data)
pn = sum(sum(1 for x in n.values() if x["pass"]) for _, _, n in data) / len(data)
print(f"\n**{len(data)} 轮均值：v1 {pv:.2f} | new {pn:.2f} | 差 {pn-pv:+.2f}**")

qids = set(data[0][1]) & set(data[0][2])
w = l = 0
for q in qids:
    a = sum(1 for _, v, _ in data if v[q]["pass"]) / len(data)
    b = sum(1 for _, _, n in data if n[q]["pass"]) / len(data)
    w += b > a
    l += b < a
print(f"逐题配对（{len(qids)} 题 × {len(data)} 轮）：new 更好 {w} | 相同 {len(qids)-w-l} | 更差 {l}")

for lab, idx in (("v1", 1), ("new", 2)):
    tot = c = 0
    for _, v, n in data:
        for rec in (v if idx == 1 else n).values():
            if has_tbl(rec):
                tot += 1
                c += bool(cited_tbl(rec))
    print(f"  表块引用率 {lab}: {c}/{tot} = {c/max(tot,1):.0%}")

for lab, want in (("有表块", True), ("无表块", False)):
    av = an = cnt = 0
    for _, v, n in data:
        for q in v:
            if bool(has_tbl(v[q])) == want:
                cnt += 1
                av += bool(v[q]["pass"])
                an += bool(n[q]["pass"])
    if cnt:
        print(f"  分层 {lab}: n={cnt} | v1 {av} -> new {an} ({an-av:+d})")
