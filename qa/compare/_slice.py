"""按 题型/问题长度/论文长度 切片 B2 vs B1/B0 的分差，验证"题简单/文短→无优势"。

只读分析，不改任何产物。
"""
from __future__ import annotations
import json

B1 = json.load(open("qa/compare/run_20260907_191606_B1.json", encoding="utf-8"))
B0 = json.load(open("qa/compare/run_20260907_191606_B0.json", encoding="utf-8"))
B2 = json.load(open("qa/compare/run_20260907_191606_B2.json", encoding="utf-8"))
PAPERS = {p["pid"]: p for p in json.load(open("qa/compare/papers_compare.json", encoding="utf-8"))}

cols = {r["qid"]: r for r in B2}
base1 = {r["qid"]: r for r in B1}
base0 = {r["qid"]: r for r in B0}


def score(r):
    return r.get("score", 0) if r.get("status") != "overflow" else 0


rows = []
for qid, r2 in cols.items():
    r1, r0 = base1[qid], base0[qid]
    q = r2["question"]
    pid = r2["pid"]
    rows.append({
        "qid": qid, "qtype": r2["qtype"], "unans": bool(r2["unanswerable"]),
        "qlen": len(q), "full_chars": PAPERS.get(pid, {}).get("full_chars", 0),
        "s2": score(r2), "s1": score(r1), "s0": score(r0),
        "question": q[:60],
    })

for r in rows:
    r["d21"] = r["s2"] - r["s1"]
    r["d20"] = r["s2"] - r["s0"]


def dump(title, filt):
    sel = [r for r in rows if filt(r)]
    n = len(sel)
    if not n:
        print(f"{title}: n=0")
        return
    md21 = sum(r["d21"] for r in sel) / n
    md20 = sum(r["d20"] for r in sel) / n
    win = sum(1 for r in sel if r["d21"] > 0)
    lose = sum(1 for r in sel if r["d21"] < 0)
    # B2 pass 而 B1 非 pass
    p21 = sum(1 for r in sel if r["s2"] >= 4 and r["s1"] < 4)
    p12 = sum(1 for r in sel if r["s1"] >= 4 and r["s2"] < 4)
    print(f"{title}: n={n} | Δavg(B2-B1)={md21:+.2f} (B2-B0)={md20:+.2f} "
          f"| B2胜/负 B1={win}/{lose} | B2pass&!B1pass={p21} | B1pass&!B2pass={p12}")


print("== 总体 ==")
dump("全部", lambda r: True)

print("\n== 按题型 ==")
for qt in ("extractive", "free_form", "yes_no", "unanswerable"):
    dump(qt, lambda r, qt=qt: r["qtype"] == qt)

print("\n== 按是否无答案 ==")
dump("unanswerable", lambda r: r["unans"])
dump("有答案", lambda r: not r["unans"])

print("\n== 按问题长度 ==")
dump("问题 ≤80 字", lambda r: r["qlen"] <= 80)
dump("问题 81-160 字", lambda r: 80 < r["qlen"] <= 160)
dump("问题 >160 字", lambda r: r["qlen"] > 160)

print("\n== 按论文全文长度 ==")
dump("论文 ≤20K 字符", lambda r: r["full_chars"] <= 20000)
dump("论文 20-30K", lambda r: 20000 < r["full_chars"] <= 30000)
dump("论文 >30K 字符", lambda r: r["full_chars"] > 30000)

print("\n== B2 输给 B1 的题（看是否集中在简单定位题）==")
for r in sorted([r for r in rows if r["d21"] < 0], key=lambda r: r["d21"]):
    print(f"  {r['qtype']:12} qlen={r['qlen']:3} B2={r['s2']} B1={r['s1']} B0={r['s0']} | {r['question']}")

print("\n== B2 赢 B1 的题 ==")
for r in sorted([r for r in rows if r["d21"] > 0], key=lambda r: -r["d21"]):
    print(f"  {r['qtype']:12} qlen={r['qlen']:3} B2={r['s2']} B1={r['s1']} B0={r['s0']} | {r['question']}")
