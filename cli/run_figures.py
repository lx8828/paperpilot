"""图表处理：识别 caption + 正文引用上下文（方案1）+ LLM 读图指南（方案2）。

用法：
    uv run python run_figures.py 2608.31079v1.pdf
    uv run python run_figures.py 2608.31079v1.pdf --no-guide   # 只做方案1，不调 LLM
    uv run python run_figures.py --all
产物：out_views/<pdf>.figures.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools import llm
from paperpilot.tools.figures import extract_figures, generate_guides
from paperpilot.tools.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"
VIEW_DIR = ROOT / "out_views"


def process_one(pdf: str, with_guide: bool, force: bool) -> dict[str, Any] | None:
    pdf_path = PAPERS_DIR / pdf
    stem = Path(pdf).stem
    out = VIEW_DIR / f"{stem}.figures.json"
    if out.exists() and not force:
        print(f"  [跳过] {pdf}（已有 figures.json）")
        return None
    blocks = parse_pdf(str(pdf_path))["blocks"]
    figs = extract_figures(blocks)
    if with_guide:
        print(f"  {pdf}: 识别 {len(figs)} 个图表，生成读图指南（LLM）…")
        guides = generate_guides(figs)
        for f in figs:
            f["guide"] = guides.get(f["id"], "")
    else:
        print(f"  {pdf}: 识别 {len(figs)} 个图表（无 guide）")
        for f in figs:
            f["guide"] = ""
    # 精简落盘：refs 只留前 3 段
    for f in figs:
        f["refs"] = f.get("refs", [])[:3]
    data = {"pdf": pdf, "figures": figs}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    n_guide = sum(1 for f in figs if f.get("guide"))
    print(f"    → {len(figs)} 图 / {n_guide} 带指南，已保存 {out.name}")
    return data


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-guide", action="store_true", help="跳过 LLM 读图指南")
    args = ap.parse_args()

    if args.all:
        pdfs = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    else:
        pdfs = list(args.pdfs)
    if not pdfs:
        print("请指定 PDF 或 --all")
        return 1

    rows = []
    for pdf in pdfs:
        data = process_one(pdf, not args.no_guide, args.force)
        if data:
            rows.append((pdf, len(data["figures"])))
    if len(rows) > 1:
        print("\n" + "=" * 44)
        for pdf, n in rows:
            print(f"  {pdf:<22s} figures={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
