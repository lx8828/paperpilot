"""临时：统计 recall_set 208 篇的 chunk 数分布 → top12 占全论文比例。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402

d = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in d["items"]})
ns = []
for pid in pids:
    ns.append(len(ordered_chunks(f"qasper_{pid}.qpdf")))
ns.sort()
n = len(ns)
print(f"论文 {n} 篇")
print(f"chunk/篇: mean={sum(ns)/n:.1f}  median={ns[n//2]}  min={ns[0]} max={ns[-1]}")
print("分布: <12:", sum(1 for x in ns if x < 12),
      "| 12-24:", sum(1 for x in ns if 12 <= x < 24),
      "| 24-40:", sum(1 for x in ns if 24 <= x < 40),
      "| >=40:", sum(1 for x in ns if x >= 40))
# top12 覆盖率（占全篇 chunk 比例）
cov = [12 / x for x in ns]
print(f"top12 平均覆盖全篇 {sum(cov)/n*100:.0f}% | 中位 {sorted(cov)[n//2]*100:.0f}%")
print(f"top8 平均覆盖全篇 {sum(8/x for x in ns)/n*100:.0f}%")
pct = [sum(1 for x in ns if 12 / x >= t) / n for t in (0.25, 0.3, 0.4, 0.5)]
print(f"top12 覆盖>=25%篇:{pct[0]:.0%} >=30%:{pct[1]:.0%} >=40%:{pct[2]:.0%} >=50%:{pct[3]:.0%}")
