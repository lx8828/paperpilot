"""QASPER 数据质量扫描：recall_set 用到的论文里，有多少 full_text 正文是空的？

判 b13d0e46 异常是「数据集缺陷」还是「我们的 loader 丢内容」：
  对比 raw 段落数（json 原样） vs build_chunks 后的字符合计。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import build_chunks, load_papers  # noqa: E402


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    pids = sorted({it["pid"] for it in data["items"]})
    papers = load_papers()

    rows = []
    for pid in pids:
        p = papers.get(pid)
        if not p:
            continue
        ft = p.get("full_text") or []
        raw_paras = sum(len(s.get("paragraphs") or []) for s in ft)
        raw_chars = sum(len(x) for s in ft for x in (s.get("paragraphs") or []))
        ch = build_chunks(p)
        ch_chars = sum(len(c.text) for c in ch)
        rows.append({"pid": pid, "secs": len(ft), "raw_paras": raw_paras,
                     "raw_chars": raw_chars, "chunks": len(ch), "ch_chars": ch_chars})

    rows.sort(key=lambda r: r["ch_chars"])
    empty = [r for r in rows if r["ch_chars"] == 0]
    tiny = [r for r in rows if 0 < r["ch_chars"] < 2000]
    L = ["# QASPER 数据质量扫描（recall_set 用到的论文）", "",
         f"- 论文 {len(rows)} 篇｜**正文 0 字符：{len(empty)} 篇**｜"
         f"正文 <2000 字符：{len(tiny)} 篇",
         "",
         "## 正文最短的 15 篇（raw=json 原始段落，map=我们切出的块）",
         "| pid | sections | raw段落 | raw字符 | 块数 | 块字符 |",
         "|---|---|---|---|---|---|"]
    for r in rows[:15]:
        L.append(f"| {r['pid']} | {r['secs']} | {r['raw_paras']} | {r['raw_chars']} | "
                 f"{r['chunks']} | {r['ch_chars']} |")
    txt = "\n".join(L)
    Path("qa/recall/QASPER_DATA_QUALITY_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
