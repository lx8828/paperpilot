"""导出某一组论文的 PDF 原文（逐页，带 [[pN]] 页标记）—— 供人工出题读原文用。

**刻意不用任何项目产物**（不读 report.json / out_mineru / summary.json / overview.json）：
只用 pymupdf 从 PDF 抽纯文本，保证题目与 gold 是"论文里本来就有的事实"，
而不是"我们系统已经答得好的东西"（避免过拟合）。

用法：python retrieval/scripts/_dump_papers.py [--group group1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import ROOT, material_dir, papers  # noqa: E402

try:
    import pymupdf as fitz          # 新版包名
except ImportError:                 # pragma: no cover
    import fitz                     # 旧包名


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    out = material_dir(args.group)
    out.mkdir(parents=True, exist_ok=True)
    for stem in papers(args.group):
        pdf = ROOT / "assets/papers" / f"{stem}.pdf"
        if not pdf.exists():
            print(f"✗ {stem}: PDF 不存在（{pdf}）")
            continue
        doc = fitz.open(pdf)
        parts = [f"##### PAPER {stem} — 共 {doc.page_count} 页（正文由 pymupdf 抽取，未做任何加工）\n"]
        for i, page in enumerate(doc, 1):
            parts.append(f"\n\n========== [[p{i}]] ==========\n")
            parts.append(page.get_text("text"))
        txt = "".join(parts)
        (out / f"{stem}.txt").write_text(txt, encoding="utf-8")
        print(f"✓ {stem}: {doc.page_count} 页, {len(txt)} 字符 → {out / (stem + '.txt')}")
    print(f"\n素材目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
