"""单篇精读报告：概述 + 核心要点 + 论证骨架 + 局限 + 章节精读（完整 Markdown）。

用法：
    uv run python run_report.py 2608.31079v1.pdf           # 单篇（LLM 生成概述）
    uv run python run_report.py 2608.31079v1.pdf --force   # 忽略概述缓存重跑

产物：assets/artifacts/out_views/<pdf>.report.md（完整报告）+ assets/artifacts/out_views/<pdf>.overview.json
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
from paperpilot.tools.pdf_parser import parse_pdf
from paperpilot.tools.report import build_guide, build_overview, render_report

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "assets" / "papers"
CLAIMS_DIR = ROOT / "assets/artifacts/out_claims"
VIEW_DIR = ROOT / "assets/artifacts/out_views"


def load_all(pdf: str):
    stem = Path(pdf).stem
    sum_f = VIEW_DIR / f"{stem}.summary.json"
    skel_f = VIEW_DIR / f"{stem}.skeleton.json"
    if not sum_f.exists() or not skel_f.exists():
        print(f"[错误] {pdf}: 需要先跑 run_view.py 和 run_skeleton.py")
        return None
    groups = json.loads(sum_f.read_text(encoding="utf-8"))["groups"]
    hubs = json.loads(skel_f.read_text(encoding="utf-8"))["hubs"]
    claim_f = CLAIMS_DIR / f"{stem}.claims.json"
    n_claims = (json.loads(claim_f.read_text(encoding="utf-8"))["n_claims"]
                if claim_f.exists() else len(groups))
    title = ""
    pdf_path = PAPERS_DIR / pdf
    if pdf_path.exists():
        title = str(parse_pdf(str(pdf_path)).get("metadata", {}).get("title", ""))
    return groups, hubs, n_claims, title or stem


def get_overview(pdf: str, groups: list[dict[str, Any]], title: str, force: bool) -> str:
    cache = VIEW_DIR / f"{Path(pdf).stem}.overview.json"
    if cache.exists() and not force:
        return str(json.loads(cache.read_text(encoding="utf-8"))["overview"])
    print("  生成概述（LLM）…")
    overview = build_overview(groups, title)
    cache.write_text(json.dumps({"overview": overview}, ensure_ascii=False,
                                indent=2), encoding="utf-8")
    return overview


def get_guide(pdf: str, groups: list[dict[str, Any]], title: str, force: bool) -> str:
    cache = VIEW_DIR / f"{Path(pdf).stem}.guide.json"
    if cache.exists() and not force:
        return str(json.loads(cache.read_text(encoding="utf-8"))["guide"])
    print("  生成导读（LLM）…")
    guide = build_guide(groups, title)
    cache.write_text(json.dumps({"guide": guide}, ensure_ascii=False,
                                indent=2), encoding="utf-8")
    return guide


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    got = load_all(args.pdf)
    if got is None:
        return 1
    groups, hubs, n_claims, title = got
    if not llm.is_configured():
        print("[错误] LLM 未配置（生成概述需要）。请填写 .env 或设置环境变量。")
        return 1
    print(f"为 {args.pdf} 生成单篇精读报告（{len(groups)} 组 / {len(hubs)} Hub）…")
    overview = get_overview(args.pdf, groups, title, args.force)
    guide = get_guide(args.pdf, groups, title, args.force)

    figures = []
    fig_file = VIEW_DIR / f"{Path(args.pdf).stem}.figures.json"
    if fig_file.exists():
        figures = json.loads(fig_file.read_text(encoding="utf-8"))["figures"]

    md = render_report(groups, hubs, overview, guide,
                       {"pdf": args.pdf, "title": title,
                        "n_claims": n_claims}, figures=figures)
    out = VIEW_DIR / f"{Path(args.pdf).stem}.report.md"
    out.write_text(md, encoding="utf-8")
    print(f"已保存: {out}\n")
    print("-" * 70)
    print("\n".join(md.splitlines()[:90]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
