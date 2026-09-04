"""一步入口：一个 PDF → report.json。

把论文精读链路（claims → 去重/打标/打分 → 骨架 → 图表 → 报告）收敛成一条命令，
最终产物为结构化 report.json（供前端渲染与问答层消费）。

用法：
    uv run python main.py 2609.00859v1.pdf                # 有缓存复用，缺环节自动生成
    uv run python main.py 2609.00859v1.pdf --force        # 全链路重跑（调 LLM，较贵）
    uv run python main.py 2609.00859v1.pdf --skip-llm     # 只装配已有产物为 report.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.pipeline import process_pdf
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", help="storage/papers 下的 PDF 文件名")
    ap.add_argument("--force", action="store_true", help="全链路重跑（忽略缓存）")
    ap.add_argument("--skip-llm", action="store_true",
                    help="只装配已有产物为 report.json（不调 LLM，缺环节报错）")
    ap.add_argument("--workers", type=int, default=4,
                    help="claims 提取并发数（默认 4）")
    args = ap.parse_args()

    if not args.skip_llm and not llm.is_configured():
        print("[错误] LLM 未配置。请先复制 .env.example 为 .env 并填写，"
              "或设置环境变量；或加 --skip-llm 只装配已有产物。")
        return 1

    for pdf in args.pdfs:
        try:
            process_pdf(pdf, force=args.force, skip_llm=args.skip_llm,
                        workers=args.workers)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[错误] {pdf}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
