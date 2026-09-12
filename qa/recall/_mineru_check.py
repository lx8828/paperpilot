"""临时：验证 10 篇 MinerU 解析产物 → chunk 桥接是否正常（重点看表格是否文本化）。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "src"))
os.environ["PAPERPILOT_USE_MINERU"] = "1"

from paperpilot.agents.document_cache import current_source, ordered_chunks  # noqa: E402

PIDS = ["2608.27843v1", "2608.28447v1", "2608.29179v1", "2608.29290v1", "2608.30333v1",
        "2608.30938v1", "2608.31046v1", "2608.31108v1", "2609.01316v1", "2609.01456v1"]
for pid in PIDS:
    pdf = f"{pid}.pdf"
    src = current_source(pdf)
    chs = ordered_chunks(pdf)
    tabs = sum(1 for c in chs if "|" in c.text and c.text.count("|") > 4)
    lens = sorted(len(c.text) for c in chs)
    print(f"{pid:16} src={src:8} chunks={len(chs):3} 含表块={tabs:2} "
          f"中位len={lens[len(lens)//2] if lens else 0} max={lens[-1] if lens else 0}")
    if pid == PIDS[0]:
        for c in chs[:3]:
            print(f"     [{c.chunk_id}] {(c.title_path or ['?'])[-1]!r} {len(c.text)} | "
                  f"{' '.join(c.text.split())[:80]}")
