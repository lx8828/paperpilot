"""临时：验证二级切分产物——同 title_path 的多块是否共享同一节名（含 part）。"""
from __future__ import annotations
import sys
from collections import Counter

sys.path.insert(0, "src")
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402

for pid in ("2002.06053", "1911.12579", "1912.00423"):
    chunks = ordered_chunks(f"qasper_{pid}.qpdf")
    print(f"=== {pid}: {len(chunks)} chunks ===")
    # 按 title_path 分组，看哪些组有多个 chunk（=二级切分共享路径）
    by_path = {}
    for c in chunks:
        key = tuple(c.title_path)
        by_path.setdefault(key, []).append(c)
    multi = {k: v for k, v in by_path.items() if len(v) > 1}
    print(f"  不同 title_path: {len(by_path)} | 有多块的路径组: {len(multi)}")
    for k, v in sorted(multi.items(), key=lambda x: -len(x[1]))[:4]:
        print(f"    {list(k)}: {len(v)} 块, parts={[c.part for c in v][:8]}, "
              f"chunk_ids={[c.chunk_id for c in v][:4]}")
        print(f"      p1 前80字: {v[0].text[:80].replace(chr(10),' ')}")
        if len(v) > 1:
            print(f"      p2 前80字: {v[1].text[:80].replace(chr(10),' ')}")
    # 块大小分布
    lens = [len(c.text) for c in chunks]
    print(f"  块字符长: mean={sum(lens)/len(lens):.0f} max={max(lens)} | "
          f"≥3900 的块: {sum(1 for l in lens if l >= 3900)}")
    print()
