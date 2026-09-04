"""论证骨架：为必读核心主张(Hub)建 implements/supports/limits/contrasts 关系边。

用法：
    uv run python run_skeleton.py 2608.31079v1.pdf           # 单篇（详细预览）
    uv run python run_skeleton.py 2608.31079v1.pdf 2608.28433v2.pdf  # 多篇
    uv run python run_skeleton.py --all --force              # 全部（force 重跑）

输入：out_views/<pdf>.summary.json（已有分组/打标/打分）+ out_claims/<pdf>.claims.json
产物：out_views/<pdf>.skeleton.json（Hub × 关系边，证据已绑定）
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
from paperpilot.tools.skeleton import build_skeleton

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
CLAIMS_DIR = ROOT / "out_claims"
VIEW_DIR = ROOT / "out_views"


def load_inputs(pdf: str) -> tuple[list[dict], dict[str, dict]] | None:
    stem = Path(pdf).stem
    sum_file = VIEW_DIR / f"{stem}.summary.json"
    claim_file = CLAIMS_DIR / f"{stem}.claims.json"
    if not sum_file.exists():
        print(f"  [跳过] {pdf}: 无 summary（先跑 run_view.py）")
        return None
    groups = json.loads(sum_file.read_text(encoding="utf-8"))["groups"]
    if not claim_file.exists():
        print(f"  [跳过] {pdf}: 无 claims（先跑 run_claims.py）")
        return None
    claim_map = {c["claim_id"]: c
                 for c in json.loads(claim_file.read_text(encoding="utf-8"))["claims"]}
    return groups, claim_map


def render_hub(h: dict) -> list[str]:
    lines = [f"● Hub [{h['label']}] {h['rep_text'][:90]}"]
    if not h["edges"]:
        lines.append("    （无关系边）")
    for e in h["edges"]:
        ev = e["evidence"].replace("\n", " ")[:70]
        lines.append(f"    - {e['relation']:<10s} ← {e['source']}  p{e['page']} "
                     f"{'✓' if e['ev'] == 'hit' else '~' if e['ev'] == 'loose' else '⚠'}"
                     f" | {e['why'][:28]}")
        if ev:
            lines.append(f"        原文: {ev}")
    return lines


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略已有 skeleton 重跑")
    args = ap.parse_args()

    if args.all:
        files = sorted(VIEW_DIR.glob("*.summary.json"))
        pdfs = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            stem = f.name.replace(".summary.json", "")
            pdf = data.get("pdf") or f"{stem}.pdf"
            pdfs.append(pdf)
    else:
        pdfs = list(args.pdfs)
    if not pdfs:
        print("请指定 PDF 或 --all")
        return 1
    if not llm.is_configured() and any(
            not (VIEW_DIR / f"{Path(p).stem}.skeleton.json").exists() or args.force
            for p in pdfs):
        print("[错误] LLM 未配置。请填写 .env 或设置环境变量。")
        return 1

    summary_rows = []
    verbose = len(pdfs) == 1
    for i, pdf in enumerate(pdfs, 1):
        stem = Path(pdf).stem
        out = VIEW_DIR / f"{stem}.skeleton.json"
        if out.exists() and not args.force:
            print(f"  [{i}/{len(pdfs)}] [跳过] {pdf}（已有 skeleton）")
            continue
        got = load_inputs(pdf)
        if got is None:
            continue
        groups, claim_map = got
        print(f"  [{i}/{len(pdfs)}] {pdf} 建骨架…（{sum(1 for g in groups if g['importance']>=5 and g['label'] in ('core_claim','result_primary'))} 个 Hub）")
        skeleton = build_skeleton(pdf, groups, claim_map)
        out.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        n_edges = sum(len(h["edges"]) for h in skeleton["hubs"])
        n_noedge = sum(1 for h in skeleton["hubs"] if not h["edges"])
        print(f"    已保存 {out.name}：{len(skeleton['hubs'])} Hub / {n_edges} 边 "
              f"/ {n_noedge} Hub 无边")
        summary_rows.append((pdf, len(skeleton["hubs"]), n_edges))
        if verbose:
            for h in skeleton["hubs"]:
                print("\n" + "\n".join(render_hub(h)))

    if len(summary_rows) > 1:
        print("\n" + "=" * 50)
        for pdf, nh, ne in summary_rows:
            print(f"  {pdf:<20s} Hub={nh} 边={ne}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
