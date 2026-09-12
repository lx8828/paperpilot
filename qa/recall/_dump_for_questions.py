"""临时：导出论文关键材料（摘要 + 贡献句 + 表格块），供人工出 hard 题并写金标准。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "src"))
os.environ["PAPERPILOT_USE_MINERU"] = "1"

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402

PIDS = sys.argv[1].split(",") if len(sys.argv) > 1 else []
LIM = int(sys.argv[2]) if len(sys.argv) > 2 else 900
OUT = Path("qa/recall/_qp_dump.md")

lines = []
for pid in PIDS:
    chs = ordered_chunks(f"{pid}.pdf")
    lines.append("=" * 100)
    lines.append(f"### {pid}")
    for c in chs:
        leaf = ((c.title_path or [""])[-1] or "")
        txt = c.text
        is_tab = txt.count("|") > 4
        low = leaf.lower()
        keep = (is_tab or "abstract" in low)
        if not keep:
            continue
        tag = "[TABLE]" if is_tab else ""
        body = " ".join(txt.split())
        lines.append(f"\n-- [{c.chunk_id}] {leaf!r} {len(txt)}字 {tag}")
        lines.append(body[:LIM])
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"written {OUT} | 字符 {sum(len(x) for x in lines)}")
