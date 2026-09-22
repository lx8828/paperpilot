"""检索命令行：查询 → 最相关的 N 篇论文（**支持时间窗**）。

用法（在仓库根目录）：
    uv run python cli/run_search.py "如何用对比学习做跨语言摘要"
    uv run python cli/run_search.py "..." --recent-days 180       # 只要近半年
    uv run python cli/run_search.py "..." --since 2026-01         # 只要 2026 年起
    uv run python cli/run_search.py "..." --since 2025-07 --until 2025-12
    uv run python cli/run_search.py "..." -k 10 --no-llm          # 只到 CE 级（省 ~0.8s）
    uv run python cli/run_search.py "..." --show-abstract

⏱ 首次调用要加载索引（~1s）+ 两个模型（~30~60s），之后常驻复用。
⚠️ 时间窗是**前置过滤**（在取 top 之前 mask），代价约 1ms；窗口内篇数会打印出来。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools.corpus_search import POOL, search  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="自然语言查询")
    ap.add_argument("-k", type=int, default=5, help="交付篇数（默认 5）")
    ap.add_argument("--pool", type=int, default=POOL, help=f"进 CE 的候选数（默认 {POOL}）")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 精排级")
    ap.add_argument("--show-abstract", action="store_true", help="打印摘要")
    ap.add_argument("--since", default="", help="只要该日期后**提交**的论文（2026 / 2026-04 / 2026-04-01）")
    ap.add_argument("--until", default="", help="只要该日期前提交的论文")
    ap.add_argument("--recent-days", type=int, default=0,
                    help="便捷：最近 N 天（等价 --since 今天−N）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="重复跑同一查询 N 次 —— 用来分离**首次预热**与**稳态耗时**"
                         "（首次含 CUDA kernel 编译/显存分配，会明显偏慢）")
    args = ap.parse_args()

    since = args.since
    if args.recent_days > 0:
        since = (dt.date.today() - dt.timedelta(days=args.recent_days)).isoformat()

    q = " ".join(args.query)
    win = f"{since or '不限'} ~ {args.until or '不限'}" if (since or args.until) else "不限"
    print(f"[检索] {q}")
    print(f"[检索] 池深 {args.pool} | 交付 {args.k} 篇 | "
          f"LLM={'开' if not args.no_llm else '关'} | 时间窗 {win}")
    for i in range(max(1, args.repeat)):
        r = search(q, k=args.k, use_llm=not args.no_llm, pool=args.pool,
                   since=since or None, until=args.until or None)
        tm = " | ".join(f"{k_}={v * 1000:.0f}ms" for k_, v in r.timings.items())
        tag = "首次(含预热)" if i == 0 else f"第 {i + 1} 次"
        print(f"[耗时] {tag}: {tm} | **合计 {r.total_ms:.0f}ms**")

    if r.degraded:
        print(f"[降级] {r.degraded}")
    print("=" * 88)
    for p in r.papers:
        print(f"{p.rank}. [{p.stage}] CE={p.score:+.3f}  {p.arxiv_id}  {p.created}")
        print(f"   {p.title}")
        if args.show_abstract:
            print(f"   {p.abstract[:300]}…")
    print("=" * 88)
    print(f"候选 {r.n_candidates} → LLM 输入 {r.n_llm_input} | "
          f"时间窗内语料 {r.n_in_window:,} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
