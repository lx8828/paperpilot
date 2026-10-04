"""成果层：`report.json` → **跨篇可比、可算**的规范卡。

    uv run python cli/eval/run_outcome.py 2408.09273.pdf
    uv run python cli/eval/run_outcome.py 2408.09273.pdf --force     # 忽略缓存重建
    uv run python cli/eval/run_outcome.py 2408.09273.pdf --json      # 打印原始 JSON

⏱ 一次约 20~60s（1 次 LLM 调用）；之后**走缓存秒回**。
⚠️ 严格校验：`quote` 必须 verbatim 回原文，`value` 还必须在该 quote 里 ——
   不通过的条目标 `✗`（**保留但不可用于跨篇比较**）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.outcome import build_outcome, load_outcome, outcome_path  # noqa: E402
from paperpilot.tools import llm  # noqa: E402


def main() -> int:
    llm._load_dotenv(str(ROOT))          # 与本项目其它 CLI 一致
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="assets/papers 下的文件名")
    ap.add_argument("--force", action="store_true", help="忽略缓存重建")
    ap.add_argument("--json", action="store_true", help="打印原始 JSON")
    ap.add_argument("--cache-only", action="store_true", help="只读缓存，缺了就退出（不调 LLM）")
    args = ap.parse_args()

    if args.cache_only:
        o = load_outcome(args.pdf)
        if o is None:
            print(f"❌ 无可用缓存：{outcome_path(args.pdf)}")
            return 1
    else:
        o = build_outcome(args.pdf, force=args.force)

    if args.json:
        print(o.model_dump_json(indent=2))
        return 0

    print("=" * 94)
    print(f"[成果卡] {o.title or o.pdf}")
    print("=" * 94)
    print(f"\n问题  : {o.problem or '（空）'}")
    print(f"方法族: {', '.join(o.method_family) or '（空）'}")

    print(f"\n数据集 ({len(o.good_datasets)}/{len(o.datasets)} 已核)")
    for d in o.datasets:
        mark = "✓" if d.verified else "✗"
        anchor = f"{d.anchor_chunk} p{d.anchor_page}" if d.verified else "**未回溯到原文**"
        print(f"  {mark} {d.name:<34} {anchor}")

    print(f"\n指标 ({len(o.good_metrics)}/{len(o.metrics)} 已核)")
    if o.metrics:
        print(f"  {'':2}{'':2} {'指标':<12}{'数值':>10}{'基线':>10}{'Δ':>9}  "
              f"{'模型':<13}{'数据集/子集':<24}{'锚点':<10}")
        for m in o.metrics:
            # ✓ = quote 逐字命中 ｜ ≈ = 数值同块共现（表格列式散开的兜底）｜ ✗ = 未通过
            mark = {"exact": "✓", "values": "≈"}.get(m.verify, "✗")
            cmp_ = "=" if m.comparable else "~"      # = 公认可跨篇比 / ~ 自创
            anchor = f"{m.anchor_chunk} p{m.anchor_page}" if m.anchor_chunk else "—"
            print(f"  {mark} {cmp_} {m.name[:12]:<12}{m.value[:10]:>10}"
                  f"{(m.baseline or '—')[:10]:>10}{(m.delta or '—'):>9}  "
                  f"{m.model[:13]:<13}{m.dataset[:24]:<24}{anchor:<10}")
        n_cmp = sum(1 for m in o.good_metrics if m.comparable)
        print(f"  → `=` 公认指标 {n_cmp} 个（可跨篇比）| `~` 自创指标 "
              f"{len(o.good_metrics) - n_cmp} 个（**仅本篇内可比**，跨篇摆表要分开）")

    print(f"\n主要结论 ({len(o.key_findings)})")
    for f in o.key_findings:
        print(f"  · {f.text}" + (f"   [{f.group_id}]" if f.group_id else ""))

    print(f"\n局限 ({len(o.limitations)})")
    for f in o.limitations:
        print(f"  · {f.text}" + (f"   [{f.group_id}]" if f.group_id else ""))

    st = o.meta.get("stats") or {}
    print("\n" + "-" * 94)
    print(f"[严格校验] 数据集 {st.get('datasets')} | 指标 {st.get('metrics')} "
          f"| 结论+局限 {st.get('findings')} 条")
    print(f"[耗时] {o.meta.get('seconds')}s | 缓存 {outcome_path(args.pdf).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
