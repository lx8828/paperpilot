"""定位某一组论文的表格页并渲染为 PNG —— 供人工**直接看版面**出表格数字题。

为什么渲染图像而不是读文本：
  表格数字题的 gold 必须是**论文的真值**。任何解析器（pymupdf 撕碎表格 / MinerU
  重组表格）都是"对版面的解释"，用任一方出 gold 都会让该题为那一方**预设立场**
  （MinerU 出 gold → 表格题天然只对 MinerU 路友好 → 循环论证）。
  直接看渲染图 = 绕过所有解析器。

用法：python retrieval/scripts/_render_tables.py [--group group1]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import ROOT, material_dir, papers  # noqa: E402

try:
    import pymupdf as fitz
except ImportError:                 # pragma: no cover
    import fitz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    out = material_dir(args.group) / "pages"
    out.mkdir(parents=True, exist_ok=True)
    for stem in papers(args.group):
        pdf = ROOT / "assets/papers" / f"{stem}.pdf"
        if not pdf.exists():
            print(f"✗ {stem}: PDF 不存在")
            continue
        doc = fitz.open(pdf)
        # 题注定位：正文里出现 "Table N:" / "Table N." / "Table N<行尾>" 的页 = 表格所在页
        by_page: dict[int, set[str]] = {}
        for i, page in enumerate(doc, 1):
            t = page.get_text("text")
            for m in re.finditer(r"\bTable\s+(\d+)\s*[:.]|\bTable\s+(\d+)\s*$", t, re.MULTILINE):
                num = m.group(1) or m.group(2)
                if num:
                    by_page.setdefault(i, set()).add(num)
        order = sorted(by_page, key=lambda p: sorted(int(x) for x in by_page[p]))
        print(f"\n{stem}  ({doc.page_count} 页)")
        print(f"  表格题注出现页: {[(p, sorted(by_page[p], key=int)) for p in order]}")
        for p in order:
            f = out / f"{stem}_p{p}.png"
            doc[p - 1].get_pixmap(dpi=150).save(f)
            print(f"    渲染 p{p} → {f.name}  ({f.stat().st_size // 1024} KB)")
    print(f"\n图像目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
