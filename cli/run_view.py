"""重点摘要视图：读 out_claims/<pdf>.claims.json → 去重 → 打标 → 算分 → Markdown。

用法：
    uv run python run_view.py 2608.31079v1.pdf              # 单篇全流程（LLM 去重+打标）
    uv run python run_view.py 2608.31079v1.pdf --force      # 忽略缓存重跑
    uv run python run_view.py 2608.31079v1.pdf --skip-llm   # 只用缓存重新渲染 md
    uv run python run_view.py 2608.31079v1.pdf --rescore    # 实验：改权重后本地重打分
    uv run python run_view.py --all                         # 全部论文（已有缓存跳过）
    uv run python run_view.py --all --force                 # 全部重跑

产物：
    out_views/<pdf>.summary.json   主张组明细（label/importance/score/ev 状态）
    out_views/<pdf>.md             重点摘要 Markdown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.models.schema import Claim, ClaimGroup
from paperpilot.tools import analyzer, llm
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.evidence import dict_to_claim, evidence_state
from paperpilot.tools.pdf_parser import parse_pdf
from paperpilot.tools.viewer import dedupe_groups, label_groups, render_markdown, score_groups

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"
CLAIMS_DIR = ROOT / "out_claims"
VIEW_DIR = ROOT / "out_views"


def group_to_dict(g: ClaimGroup) -> dict:
    return g.model_dump(mode="json")


def dict_to_group(d: dict) -> ClaimGroup:
    return ClaimGroup(**d)


def load_claims(pdf_name: str) -> tuple[list[Claim], dict[str, Any]]:
    claims_file = CLAIMS_DIR / f"{Path(pdf_name).stem}.claims.json"
    if not claims_file.exists():
        raise FileNotFoundError(f"缺少 claims 结果：{claims_file}（先跑 run_claims.py）")
    data = json.loads(claims_file.read_text(encoding="utf-8"))
    return [dict_to_claim(d) for d in data["claims"]], data


def build_ev_map(pdf_name: str, claims: list[Claim]) -> dict[str, str]:
    pdf_path = PAPERS_DIR / pdf_name
    if not pdf_path.exists():
        print("  [警告] 找不到 PDF，跳过 evidence 核对")
        return {}
    result = parse_pdf(str(pdf_path))
    target = analyzer.extractable(chunk_document(result["blocks"], max_len=4000))
    text_map = {c.chunk_id: c.text for c in target}
    return {c.claim_id: evidence_state(c, text_map) for c in claims}


def process_pdf(pdf_name: str, *, force: bool = False,
                skip_llm: bool = False, rescore: bool = False,
                verbose: bool = False) -> dict[str, Any] | None:
    """处理单篇：去重/打标（LLM）→ 算分 → 落盘 summary.json + md。

    Returns: 统计 dict；失败返回 None。
    """
    try:
        claims, _ = load_claims(pdf_name)
    except FileNotFoundError as e:
        print(f"  [跳过] {e}")
        return None
    claim_map = {c.claim_id: c for c in claims}
    pdf_stem = Path(pdf_name).stem
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    cache = VIEW_DIR / f"{pdf_stem}.summary.json"

    groups: list[ClaimGroup]
    if rescore:
        if not cache.exists():
            print(f"  [错误] {pdf_name}: 无缓存，先不带 --rescore 跑一次")
            return None
        groups = [dict_to_group(d)
                  for d in json.loads(cache.read_text(encoding="utf-8"))["groups"]]
        print(f"  [{pdf_name}] 重打分（本地规则）…")
        score_groups(groups, claim_map, build_ev_map(pdf_name, claims))
        mode = "rescore"
    elif (cache.exists() and not force) or skip_llm:
        if not cache.exists():
            print(f"  [错误] {pdf_name}: 无缓存，先不带 --skip-llm 跑一次")
            return None
        groups = [dict_to_group(d)
                  for d in json.loads(cache.read_text(encoding="utf-8"))["groups"]]
        mode = "cache"
    else:
        if not llm.is_configured():
            print(f"  [错误] {pdf_name}: LLM 未配置")
            return None
        if verbose:
            print(f"① LLM 去重归并：{len(claims)} 条 claims …")
        groups = dedupe_groups(claims)
        if verbose:
            print(f"   → {len(groups)} 个主张组")
            print("② LLM 角色打标 …")
        label_groups(groups)
        score_groups(groups, claim_map, build_ev_map(pdf_name, claims))
        mode = "new"

    n_must = sum(1 for g in groups if g.importance >= 5)
    n_mid = sum(1 for g in groups if 4 <= g.importance < 5)
    n_low = sum(1 for g in groups if g.importance < 4)
    n_ev_miss = sum(1 for g in groups if g.ev_state == "miss")
    n_lim = sum(1 for g in groups if g.label == "limitation")

    payload = {"pdf": pdf_name, "n_claims": len(claims),
               "groups": [group_to_dict(g) for g in groups]}
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    pdf_title = ""
    pdf_path = PAPERS_DIR / pdf_name
    if pdf_path.exists():
        pdf_title = str(parse_pdf(str(pdf_path)).get("metadata", {}).get("title", ""))
    md = render_markdown(groups, {"pdf": pdf_name, "n_claims": len(claims),
                                  "title": pdf_title or pdf_stem})
    md_file = VIEW_DIR / f"{pdf_stem}.md"
    md_file.write_text(md, encoding="utf-8")

    if verbose:
        print(f"结果（{mode}）：组数={len(groups)} 必读≥5={n_must} "
              f"应读4={n_mid} 按需<4={n_low} | 局限={n_lim} | ⚠待核={n_ev_miss}")
        print(f"已保存: {cache}")
        print(f"已保存: {md_file}\n")
        print("-" * 70)
        print("\n".join(md.splitlines()[:80]))
    return {"pdf": pdf_name, "n_claims": len(claims), "n_groups": len(groups),
            "must": n_must, "mid": n_mid, "low": n_low,
            "lim": n_lim, "ev_miss": n_ev_miss}


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", help="PDF 文件名（可多个）")
    ap.add_argument("--all", action="store_true", help="跑全部论文（有缓存跳过）")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新跑 LLM 步骤")
    ap.add_argument("--skip-llm", action="store_true",
                    help="不调 LLM：仅用已有 summary 重新渲染")
    ap.add_argument("--rescore", action="store_true",
                    help="实验用：用缓存的分组/label，仅按最新权重重新算分")
    args = ap.parse_args()

    if args.all:
        files = sorted(CLAIMS_DIR.glob("*.claims.json"))
        if not files:
            print(f"[错误] {CLAIMS_DIR} 下没有 claims 结果，先跑 run_claims.py")
            return 1
        pdfs = [json.loads(f.read_text(encoding="utf-8"))["pdf"] for f in files]
    else:
        pdfs = list(args.pdfs)
    if not pdfs:
        ap.error("需要指定 PDF 文件名，或用 --all")

    done = []
    for i, pdf in enumerate(pdfs, 1):
        cache = VIEW_DIR / f"{Path(pdf).stem}.summary.json"
        if cache.exists() and not args.force and not args.rescore:
            print(f"  [{i}/{len(pdfs)}] [跳过] {pdf}（已有缓存，--force 重跑）")
            continue
        verbose = len(pdfs) == 1 and not args.all
        if verbose:
            print(f"\n=== {pdf} ===")
        st = process_pdf(pdf, force=args.force, skip_llm=args.skip_llm,
                         rescore=args.rescore, verbose=verbose)
        if st:
            done.append(st)
            if not verbose:
                print(f"  [{i}/{len(pdfs)}] {pdf}: claims={st['n_claims']} "
                      f"组={st['n_groups']} 必读={st['must']} 应读={st['mid']} "
                      f"按需={st['low']} 局限={st['lim']} 待核={st['ev_miss']}")

    if done and len(done) > 1:
        print("\n" + "=" * 72)
        print(f"{'pdf':<20s}{'claims':>6s}{'组':>5s}{'必读':>5s}{'应读':>5s}"
              f"{'按需':>5s}{'局限':>5s}{'待核':>5s}")
        for st in done:
            print(f"{st['pdf']:<20s}{st['n_claims']:>6d}{st['n_groups']:>5d}"
                  f"{st['must']:>5d}{st['mid']:>5d}{st['low']:>5d}"
                  f"{st['lim']:>5d}{st['ev_miss']:>5d}")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
