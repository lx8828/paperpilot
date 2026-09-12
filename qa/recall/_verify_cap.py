"""临时：验证 embedder 的 section cap 开关（env 开/关行为对照）。

关 → 与旧 search_hybrid 一致（top12 等权 RRF 截断）；
开(cap=1) → 保序节级去重：看是否出现"同一节只保留首个命中、候选跨节更多"。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, "src")
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402

os.environ.pop("PAPERPILOT_RETRIEVE_SECTION_CAP", None)
idx = ChunkIndex("qasper_2002.00652.qpdf")
off = idx.search_hybrid("What two large datasets are used for evaluation?", top_k=12)
os.environ["PAPERPILOT_RETRIEVE_SECTION_CAP"] = "1"
on = idx.search_hybrid("What two large datasets are used for evaluation?", top_k=12)

def sec(h):  # noqa: E731
    tp = h.get("title_path") or []
    return tp[0] if tp else ""


def secs_of(hits):  # noqa: E731
    return [sec(h) for h in hits]


print("=== 关 (现状 top12) ===")
for i, h in enumerate(off, 1):
    print(f"  {i:2} [{sec(h)}] {h['text'][:48].replace(chr(10),' ')}")
print("=== 开 cap=1（每节最多1块，仍取到12 = 跨12节）===")
for i, h in enumerate(on, 1):
    print(f"  {i:2} [{sec(h)}] {h['text'][:48].replace(chr(10),' ')}")
# 验证 cap=1 是否真的"每节唯一"
secs = [sec(h) for h in on]
dup = [s for s in set(secs) if secs.count(s) > 1]
print(f"\n关: {len(off)} 块 / {len(set(secs_of(off)))} 节")
print(f"开: {len(on)} 块 / {len(set(secs))} 节 | 重复节: {dup or '无'}")
