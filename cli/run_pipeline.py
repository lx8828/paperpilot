"""端到端：**一个问题 → 检索 → 抓取 → 精读 → 报告**。

    uv run python cli/run_pipeline.py "如何用对比学习做跨语言摘要生成"
    uv run python cli/run_pipeline.py "..." --pick 2 -k 8          # 精读第 2 篇
    uv run python cli/run_pipeline.py "..." --paper 2104.09864     # 跳过检索，指定一篇
    uv run python cli/run_pipeline.py "..." --paper 2305.14205,2403.13240   # 批量（跨篇对比用）
    uv run python cli/run_pipeline.py "..." --force                # 全链路重跑

⏱ 一次真实运行：检索 ~4s + 抓取 ~2s + MinerU 1~5min + 报告链 2~5min + 索引 ~20s。
   想快速验证管路（不出真报告）：
       PAPERPILOT_MINERU=0 PAPERPILOT_MOCK_LLM=1 uv run python cli/run_pipeline.py "..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.workflow import POOL, QueryResult, run_query  # noqa: E402


def _show(r: QueryResult, args: argparse.Namespace) -> int:
    """打印一次运行的交付物。返回进程退出码。"""
    if not r.ok:
        print(f"❌ 失败：{r.reason}")
        _print_steps(r)
        return 1

    if len(r.hits) > 1:
        print("【检索候选】")
        for p in r.hits:
            mark = "→" if p is r.picked else " "
            print(f" {mark} {p.rank}. [{p.stage}] {p.title[:76]}")
            print(f"      {p.arxiv_id}  https://arxiv.org/abs/{p.arxiv_id}")
        print()

    print("【交付】")
    print(f"  论文   : {r.pdf_name}")
    print(f"  PDF    : {r.pdf_path}")
    mst = str(r.mineru.get("status") or "-")
    print(f"  MinerU : {mst}" + (f" — {r.mineru.get('reason')}" if mst not in ("ok", "skipped") else ""))
    if r.report_md:
        print(f"  报告   : {r.report_md}")
        print(f"           （{r.report_md.stat().st_size / 1024:.1f} KB）")
    else:
        print("  报告   : **未落盘**")
    print()

    if r.report_md and args.head > 0:
        lines = r.report_md.read_text(encoding="utf-8").splitlines()
        print("-" * 92)
        print(f"【报告正文（前 {min(args.head, len(lines))} / {len(lines)} 行）】")
        print("-" * 92)
        for ln in lines[:args.head]:
            print(ln)
        if len(lines) > args.head:
            print(f"…（其余 {len(lines) - args.head} 行见上方路径）")
        print("-" * 92)

    _print_steps(r)
    return 0


def _print_steps(r: QueryResult) -> None:
    print()
    print(f"{'阶段':<28}{'状态':>9}{'秒':>10}  说明")
    print("-" * 92)
    for s in r.steps:
        print(f"{s.name:<28}{s.status:>9}{s.seconds:>10.1f}  {s.note}")
    print("-" * 92)
    print(f"{'合计':<28}{'':>9}{r.seconds:>10.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="自然语言检索问题")
    ap.add_argument("-k", type=int, default=5, help="检索返回的候选篇数（默认 5）")
    ap.add_argument("--pick", type=int, default=1, help="精读第几篇（1-based，默认 1）")
    ap.add_argument("--paper", default="",
                    help="直接指定 arXiv id 跳过检索；**逗号分隔可批量**（跨篇对比用）")
    ap.add_argument("--force", action="store_true", help="MinerU 与报告链全链路重跑")
    ap.add_argument("--skip-llm", action="store_true", help="报告链只装配已有产物，不调 LLM")
    ap.add_argument("--workers", type=int, default=4, help="claims 提取并发数（默认 4）")
    ap.add_argument("--no-llm", action="store_true", help="检索跳过 LLM 精排级")
    ap.add_argument("--pool", type=int, default=POOL, help=f"检索候选池（默认 {POOL}）")
    ap.add_argument("--no-index", action="store_true", help="不建问答向量索引")
    ap.add_argument("--head", type=int, default=40, help="报告正文打印行数（默认 40）")
    args = ap.parse_args()

    q = " ".join(args.query)
    common = dict(k=args.k, pick=args.pick, force=args.force, skip_llm=args.skip_llm,
                  workers=args.workers, use_llm=not args.no_llm, pool=args.pool,
                  build_index=not args.no_index, verbose=True)

    papers = [s.strip() for s in args.paper.split(",") if s.strip()]
    if not papers:
        print("=" * 92)
        print(f"[问题] {q}")
        print("=" * 92)
        print("【串联 ①②③】")
        return _show(run_query(q, **common), args)

    # 批量：逐篇串行（MinerU 吃 GPU，并行会抢显存）
    rc = 0
    for i, pid in enumerate(papers, 1):
        print("=" * 92)
        print(f"[{i}/{len(papers)}] {q}  →  直接指定 {pid}（跳过检索）")
        print("=" * 92)
        print("【串联 ①②③】")
        rc |= _show(run_query(q, paper=pid, **common), args)
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
