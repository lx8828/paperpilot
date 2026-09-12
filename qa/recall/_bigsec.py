"""临时：量化"大节论文"——某单节块数很大的论文占比（决定 cap 是否需自适应）。"""
from __future__ import annotations
import json
import sys
from collections import Counter

sys.path.insert(0, "src")
from paperpilot.agents.document_cache import ordered_chunks, top_section  # noqa: E402

d = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in d["items"]})

rows = []
for pid in pids:
    chunks = ordered_chunks(f"qasper_{pid}.qpdf")
    secs = Counter(top_section(c) or "(no-sec)" for c in chunks)
    nsec = len(secs)
    maxsec = max(secs.values())            # 最大节块数
    # 若 cap=1：该篇丢掉的"同节第二块"最大损失；若答案在同节多块则可能缺
    rows.append({"pid": pid, "n_chunk": len(chunks), "n_sec": nsec,
                 "max_sec": maxsec, "top2_sec": sorted(secs.values(), reverse=True)[:3]})

n = len(rows)
print(f"论文 {n} 篇")
def pct(pred, label):
    c = sum(1 for r in rows if pred(r))
    print(f"  {label}: {c}/{n} ({c/n:.0%})")

print("存在'大节'（单节块数 ≥5）的论文:")
pct(lambda r: r["max_sec"] >= 5, "≥5块")
pct(lambda r: r["max_sec"] >= 8, "≥8块")
pct(lambda r: r["max_sec"] >= 12, "≥12块")
print()
print("cap=1 在'大节论文'上的潜在伤害（该篇节内第2..块合计被砍数量，若答案分散即缺料）:")
pct(lambda r: r["max_sec"] >= 5 and r["n_sec"] >= 5, "有≥5块大节 且 总节≥5（可被去重牺牲但有真实内容）")
print()
print("典型大节论文样例:")
for r in sorted([x for x in rows if x["max_sec"] >= 8], key=lambda x: -x["max_sec"])[:10]:
    print(f"  {r['pid']} chunk={r['n_chunk']} 节数={r['n_sec']} 最大节={r['max_sec']} "
          f"top节块数={r['top2_sec']}")
