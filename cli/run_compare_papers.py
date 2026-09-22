"""跨篇分析：N 篇同主题论文 → **一次 LLM 调用** → 对比分析。

    uv run python cli/run_compare_papers.py 2408.09273.pdf 2305.14205.pdf 2403.13240.pdf
    uv run python cli/run_compare_papers.py <pdf> ... --question "我更关心低资源语言上的表现"
    uv run python cli/run_compare_papers.py <pdf> ... --cache-only   # 成果层缺了就报错，不调 LLM

产物：`assets/artifacts/out_compare/<名字>.md` + `.json`

设计：**不做对照表** —— N 篇（≤5）的成果卡合计约 30KB（≈8K token），全在上下文里，
让 LLM 直接读卡片分析比预计算一张对齐表更直接（理由见 `paperpilot/prompts/compare.py`）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.compare import compare_papers, save  # noqa: E402
from paperpilot.tools import llm  # noqa: E402


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", help="assets/papers 下的文件名（2~5 篇）")
    ap.add_argument("--question", default="", help="研究者关注点（引导分析角度）")
    ap.add_argument("--force", action="store_true", help="成果层忽略缓存重建")
    ap.add_argument("--cache-only", action="store_true", help="成果层缺了报错，不调 LLM")
    ap.add_argument("--out", default="", help="Markdown 文件名（默认 analysis_<n>p.md）")
    args = ap.parse_args()
    if len(args.pdfs) < 2:
        print("至少 2 篇才能对比")
        return 2

    a = compare_papers(args.pdfs, question=args.question,
                       build=not args.cache_only, force=args.force)
    p = save(a, args.out)
    print()
    print(a.to_markdown())
    print()
    print(f"已写 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
