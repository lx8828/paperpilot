"""临时：看真实 chunk 的 title_path / top_section 形态（定 intro/conclusion 匹配规则）。"""
from __future__ import annotations
import json
import sys
from collections import Counter

sys.path.insert(0, "src")
from paperpilot.agents.document_cache import ordered_chunks, top_section  # noqa: E402

d = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in d["items"]})[:8]
sections = Counter()
paths = []
for pid in pids:
    for c in ordered_chunks(f"qasper_{pid}.qpdf"):
        sections[top_section(c)] += 1
        if len(paths) < 30:
            paths.append((c.chunk_id, c.title_path))
print("top_section 高频（前 25）:")
for s, cnt in sections.most_common(25):
    print(f"  {cnt:3}  {s!r}")
print("—" * 50)
print("title_path 样例:")
for cid, tp in paths:
    print(f"  {cid}: {tp}")
