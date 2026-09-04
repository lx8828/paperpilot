"""claims 提取实验入口：parse → chunk → LLM 提取四类 claims，并核对证据。

跑之前请先填写根目录 .env（参考 .env.example），或设置环境变量：
    PAPERPILOT_LLM_BASE_URL / PAPERPILOT_LLM_API_KEY / PAPERPILOT_LLM_MODEL

用法：
    uv run python run_claims.py 2608.28433v2.pdf                        # 单篇全跑（详细输出）
    uv run python run_claims.py 2608.28433v2.pdf --max-chunks 6         # 单篇试水 N 块
    uv run python run_claims.py --all                                   # 全部论文（每篇一行小结）
    uv run python run_claims.py 2608.28433v2.pdf 2608.30023v1.pdf       # 指定多篇
    uv run python run_claims.py --all --skip-existing                   # 跳过 out_claims 已存在的篇目（续跑）
    uv run python run_claims.py --json out_claims/xxx.claims.json       # 复核已有结果（不调 LLM）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.models.schema import Claim
from paperpilot.tools import analyzer, llm
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"
DEFAULT_OUT = ROOT / "out_claims"

# ───────────────────────── 工具（纯函数在 tools/evidence） ─────────────────────────


from paperpilot.tools.evidence import (  # noqa: E402
    claim_to_dict as claims_to_dict,
    dict_to_claim,
    norm_text,
    verify_evidence,
)


def evidence_stats(claims: list[Claim], target_chunks: list[Any]) -> tuple[int, int, int]:
    chunk_text_map = {c.chunk_id: c.text for c in target_chunks}
    hit, loose, miss = verify_evidence(claims, chunk_text_map)
    return len(hit), len(loose), len(miss)


def print_samples(claims: list[Claim], per_type: int = 3) -> None:
    print(f"\n样例（完整原文证据，每类最多 {per_type} 条）：")
    seen: Counter[str] = Counter()
    for c in claims:
        if seen[c.type] >= per_type:
            continue
        seen[c.type] += 1
        print(f"\n[{c.type}] {c.claim_id} <- {c.chunk_id} p{c.page}")
        print(f"  文本: {c.text}")
        print(f"  原文: {c.evidence_quote}")


def report_miss(claims: list[Claim], target_chunks: list[Any]) -> None:
    chunk_text_map = {c.chunk_id: c.text for c in target_chunks}
    hit, loose, miss = verify_evidence(claims, chunk_text_map)
    print(f"证据核对：✓ 逐字 {len(hit)} | ~ 省略式 {len(loose)} | ✗ 未命中 {len(miss)}")
    for c in miss[:8]:
        print(f"\n  ✗ 未命中 [{c.type}] {c.claim_id} <- {c.chunk_id}")
        print(f"     证据: {c.evidence_quote}")
        nq = norm_text(c.evidence_quote)
        drifted = [cid for cid, t in chunk_text_map.items()
                   if cid != c.chunk_id and nq and nq in norm_text(t)]
        if drifted:
            print(f"     → 证据实际位于其它 chunk {drifted}（来源 chunk 可能标错）")


# ───────────────────────── 单篇流程 ─────────────────────────


def run_one(pdf_name: str, max_chunks: int, workers: int,
            out_dir: Path, skip_existing: bool,
            verbose: bool) -> dict[str, Any] | None:
    """跑单篇。返回统计 dict；跳过/出错返回 None。"""
    pdf_path = PAPERS_DIR / pdf_name
    if not pdf_path.exists():
        print(f"  [错误] 找不到 {pdf_path}")
        return None
    out_file = out_dir / f"{pdf_path.stem}.claims.json"
    if skip_existing and out_file.exists():
        print(f"  - 已存在，跳过: {out_file.name}")
        return None

    result = parse_pdf(str(pdf_path))
    chunks = chunk_document(result["blocks"], max_len=4000)
    target = analyzer.extractable(chunks)
    if max_chunks:
        target = target[:max_chunks]

    claims, errors = analyzer.extract_claims(target, workers=workers)
    hit, loose, miss = evidence_stats(claims, target)
    by_type = Counter(c.type for c in claims)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdf": pdf_name,
        "model": os.environ.get("PAPERPILOT_LLM_MODEL", ""),
        "n_chunks": len(target),
        "n_claims": len(claims),
        "by_type": dict(by_type),
        "errors": errors,
        "claims": [claims_to_dict(c) for c in claims],
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    if verbose:
        print(f"正文 chunk: {len(chunks)} 个（可提取 {len(target)} 个）")
        print(f"提取完成：claims {len(claims)} 条 | 类型分布 {dict(by_type)}")
        print(f"失败 chunk：{len(errors)}/{len(target)}")
        for e in errors[:10]:
            print(f"  ✗ {e}")
        report_miss(claims, target)
        print_samples(claims)
        print(f"\n结果已保存: {out_file}")

    return {
        "pdf": pdf_name,
        "n_chunks": len(target),
        "n_claims": len(claims),
        "by_type": dict(by_type),
        "n_errors": len(errors),
        "hit": hit, "loose": loose, "miss": miss,
        "out": str(out_file),
    }


# ───────────────────────── 汇总打印 ─────────────────────────

TYPE_ORDER = ["contribution", "method", "result", "limitation"]


def print_table(rows: list[dict[str, Any]]) -> None:
    print("=" * 100)
    hdr = (f"{'pdf':<22s} {'chunk':>5s} {'claims':>6s} |"
           f" {'ctr':>3s} {'met':>3s} {'res':>3s} {'lim':>3s} |"
           f" {'ev✓':>4s} {'ev~':>4s} {'ev✗':>4s} {'err':>3s}")
    print(hdr)
    print("-" * 100)
    tot = Counter()
    for r in rows:
        bt = r["by_type"]
        print(f"{r['pdf']:<22s} {r['n_chunks']:>5d} {r['n_claims']:>6d} |"
              f" {bt.get('contribution', 0):>3d} {bt.get('method', 0):>3d}"
              f" {bt.get('result', 0):>3d} {bt.get('limitation', 0):>3d} |"
              f" {r['hit']:>4d} {r['loose']:>4d} {r['miss']:>4d} {r['n_errors']:>3d}")
        tot["n_claims"] += r["n_claims"]
        tot["hit"] += r["hit"]
        tot["loose"] += r["loose"]
        tot["miss"] += r["miss"]
        for t in TYPE_ORDER:
            tot[t] += bt.get(t, 0)
    print("-" * 100)
    print(f"{'合计':<22s} {'':>5s} {tot['n_claims']:>6d} |"
          f" {tot['contribution']:>3d} {tot['method']:>3d}"
          f" {tot['result']:>3d} {tot['limitation']:>3d} |"
          f" {tot['hit']:>4d} {tot['loose']:>4d} {tot['miss']:>4d} {'':>3s}")
    print("=" * 100)


# ───────────────────────── 复核模式 ─────────────────────────


def run_recheck(json_path: str) -> int:
    path = ROOT / json_path if not os.path.isabs(json_path) else Path(json_path)
    if not path.exists():
        print(f"[错误] 找不到 {path}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = [dict_to_claim(d) for d in data["claims"]]
    pdf_name = data.get("pdf", "")
    pdf_path = PAPERS_DIR / pdf_name
    if not pdf_path.exists():
        print(f"[错误] 找不到来源 PDF {pdf_path}，无法核对")
        return 1
    print(f"复核: {path.name}（pdf={pdf_name}, claims={len(claims)}）")
    result = parse_pdf(str(pdf_path))
    target = analyzer.extractable(chunk_document(result["blocks"], max_len=4000))
    by_type = Counter(c.type for c in claims)
    print(f"类型分布 {dict(by_type)}")
    report_miss(claims, target)
    print_samples(claims, per_type=2)
    return 0


# ───────────────────────── 主流程 ─────────────────────────


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", help="PDF 文件名（可多个）")
    ap.add_argument("--all", action="store_true", help="跑 storage/papers 下全部论文")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="每篇只跑前 N 个正文 chunk（试水）；0=全部")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    ap.add_argument("--workers", type=int, default=4,
                    help="并发 LLM 请求数（默认 4）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过 out 目录已存在的 .claims.json（续跑）")
    ap.add_argument("--json", dest="json_path",
                    help="复核已有 .claims.json（不调 LLM）")
    args = ap.parse_args()

    if args.json_path:
        return run_recheck(args.json_path)

    if args.all:
        pdfs = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    else:
        pdfs = list(args.pdfs)
    if not pdfs:
        ap.error("需要指定 PDF 文件名，或用 --all 跑全部，或用 --json 复核")
    if not llm.is_configured():
        print("[错误] LLM 未配置。请先复制 .env.example 为 .env 并填写，或设置环境变量。")
        return 1

    out_dir = Path(args.out)
    verbose = len(pdfs) == 1
    if verbose:
        print(f"单篇模式：{pdfs[0]}")
    else:
        print(f"批量模式：{len(pdfs)} 篇 | 并发 {args.workers}"
              f"{' | 跳过已存在' if args.skip_existing else ''}")

    done: list[dict[str, Any]] = []
    for i, pdf in enumerate(pdfs, 1):
        if not verbose:
            print(f"\n[{i}/{len(pdfs)}] {pdf} …")
        stat = run_one(pdf, args.max_chunks, args.workers, out_dir,
                       args.skip_existing, verbose=verbose)
        if stat is not None:
            done.append(stat)
            if not verbose:
                bt = stat["by_type"]
                print(f"     claims={stat['n_claims']} (c{bt.get('contribution', 0)} "
                      f"m{bt.get('method', 0)} r{bt.get('result', 0)} "
                      f"l{bt.get('limitation', 0)}) ev✓{stat['hit']} "
                      f"ev~{stat['loose']} ev✗{stat['miss']} err={stat['n_errors']}")

    if done:
        print_table(done)
        summary_file = out_dir / "_summary.json"
        summary_file.write_text(
            json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"汇总明细已保存: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
