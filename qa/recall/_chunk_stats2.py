"""临时：准确统计 top12 覆盖全篇 chunk 比例（cap 到 100%）。"""
from __future__ import annotations
import json
import sys

sys.path.insert(0, "src")
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402

d = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in d["items"]})
ns = sorted(len(ordered_chunks(f"qasper_{pid}.qpdf")) for pid in pids)
n = len(ns)


def cov(k):
    return [min(k, x) / x for x in ns]


print(f"论文 {n} 篇 | chunk/篇 mean={sum(ns)/n:.1f} median={ns[n//2]} max={ns[-1]}")
for k in (8, 12, 16):
    c = cov(k)
    print(f"top{k}: 覆盖全篇 平均 {sum(c)/n*100:.0f}% | 中位 {sorted(c)[n//2]*100:.0f}% | "
          f"覆盖≥75%篇 {sum(1 for x in c if x >= .75)/n:.0%} | 覆盖≥90%篇 {sum(1 for x in c if x >= .9)/n:.0%} | "
          f"覆盖<50%篇 {sum(1 for x in c if x < .5)/n:.0%}")
# 各 k 下候选 = 全篇的多少
print("—" * 60)
for k in (8, 12, 16):
    # top_k 真实取到多少篇（chunk < top_k 时取全部）
    actual = [min(k, x) for x in ns]
    print(f"top{k}: 平均实际喂给 answer 的块 {sum(actual)/n:.1f}（全篇 {sum(ns)/n:.1f}）")
