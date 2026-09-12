"""临时：cap=1 节级去重后，top_k 实际能填多少块（量化"尾巴空多少"）。

实际填满上限 = min(top_k, 该篇不同 top_section 数)（每节至多 1 块，遍历全文保序）。
若不同节数 < top_k → top_k 取不满，尾巴空。本脚本统计每篇不同节数分布 +
cap=1 下取 top_k∈{8,12,16} 的实际块数分布。
"""
from __future__ import annotations
import json
import sys
from collections import Counter

sys.path.insert(0, "src")
from paperpilot.agents.document_cache import ordered_chunks, top_section  # noqa: E402

d = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in d["items"]})

# 每篇：chunk 总数、不同节数（含空节名计 1 类）
rows = []
for pid in pids:
    chunks = ordered_chunks(f"qasper_{pid}.qpdf")
    secs = [top_section(c) or "(no-sec)" for c in chunks]
    nsec = len(set(secs))
    rows.append({"pid": pid, "n_chunk": len(chunks), "n_sec": nsec,
                 "sec_hist": dict(Counter(secs))})

n = len(rows)
print(f"论文 {n} 篇")
ns = sorted(r["n_sec"] for r in rows)
nc = sorted(r["n_chunk"] for r in rows)
print(f"不同节数/篇: mean={sum(ns)/n:.1f} median={ns[n//2]} max={ns[-1]} "
      f"(chunk/篇 mean={sum(nc)/n:.1f})")
print("节数分布: <5:", sum(1 for x in ns if x < 5),
      "| 5-7:", sum(1 for x in ns if 5 <= x < 8),
      "| 8-11:", sum(1 for x in ns if 8 <= x < 12),
      "| 12-15:", sum(1 for x in ns if 12 <= x < 16),
      "| >=16:", sum(1 for x in ns if x >= 16))
print()
for k in (8, 12, 16):
    filled = [min(k, r["n_sec"]) for r in rows]     # cap=1 实际最多
    tail = [k - f for f in filled]
    n_short = sum(1 for r in rows if r["n_sec"] < k)  # 该 k 取不满的篇
    print(f"cap=1 + top{k}:")
    print(f"  实际平均取 {sum(filled)/n:.1f} 块（中位 {sorted(filled)[n//2]}）| "
          f"取不满 top{k} 的篇: {n_short}/{n} ({n_short/n:.0%})")
    print(f"  尾巴空分布: 空0={sum(1 for t in tail if t==0)} 空1-3={sum(1 for t in tail if 1<=t<=3)} "
          f"空4+={sum(1 for t in tail if t>=4)} | 平均空 {sum(tail)/n:.1f} 块")
# 补一刀：现状(无 cap, 直接 top_k)实际取多少块 —— 若 chunk < k 也取不满
print()
for k in (12,):
    filled_cur = [min(k, r["n_chunk"]) for r in rows]
    short = sum(1 for r in rows if r["n_chunk"] < k)
    print(f"现状(无cap) top{k}: 平均实际取 {sum(filled_cur)/n:.1f} 块 | "
          f"chunk<{k} 的篇 {short}/{n} ({short/n:.0%}) 尾巴空")
# 列出 cap=1+top12 空的篇做样例（节数 < 12）
print()
print("cap=1+top12 填不满的篇（前 15，节数/块数）:")
for r in sorted(rows, key=lambda x: x["n_sec"])[:15]:
    print(f"  {r['pid']}  chunk={r['n_chunk']} 节={r['n_sec']} "
          f"cap1实取={min(12, r['n_sec'])} | 节hist={r['sec_hist']}")
