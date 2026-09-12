"""摄取命令行：PDF → 论文库 → MinerU（检索）→ 报告产物（pymupdf）。

用法（在仓库根目录）：
    uv run python cli/run_ingest.py 2609.01456v1.pdf          # 幂等，已有产物则跳过
    uv run python cli/run_ingest.py 2609.01456v1.pdf --force  # 重跑（含 MinerU）
    uv run python cli/run_ingest.py --all                     # 论文库全部
    uv run python cli/run_ingest.py x.pdf --src D:/in/x.pdf   # 先把外部 PDF 收进论文库
    uv run python cli/run_ingest.py x.pdf --no-mineru         # 只出报告（不跑 MinerU）

退出码：0=成功；1=有论文 MinerU 失败（报告仍已生成）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.ingest import PAPERS_DIR, ingest, mineru_available  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", help="论文库（assets/papers）下的文件名")
    ap.add_argument("--all", action="store_true", help="摄取论文库全部 PDF")
    ap.add_argument("--src", default=None,
                    help="把该路径的 PDF 复制进论文库后再摄取（用于上传场景）")
    ap.add_argument("--force", action="store_true", help="重跑（含 MinerU）")
    ap.add_argument("--skip-llm", action="store_true", help="只装配已有报告产物")
    ap.add_argument("--no-mineru", action="store_true", help="跳过 MinerU（仅出报告）")
    args = ap.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    if args.src:
        src = Path(args.src)
        if not src.exists():
            print(f"[错误] 找不到源 PDF：{src}")
            return 1
        dst = PAPERS_DIR / (args.pdfs[0] if args.pdfs else src.name)
        shutil.copy2(src, dst)
        print(f"[摄取] 已收入论文库：{dst}")

    if args.all:
        pdfs = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    elif args.pdfs:
        pdfs = [args.pdfs[0]] if args.src else args.pdfs
    else:
        ap.print_help()
        return 1
    if not pdfs:
        print("[错误] 论文库为空")
        return 1

    ok, why = mineru_available()
    if not args.no_mineru:
        print(f"[MinerU] {'可用' if ok else '不可用：' + why}")
    else:
        print("[MinerU] 本次跳过（--no-mineru）")

    failed: list[str] = []
    for name in pdfs:
        print("=" * 78)
        print(f"[摄取] {name}")
        try:
            res = ingest(name, force=args.force, skip_llm=args.skip_llm,
                         mineru=not args.no_mineru, verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [错误] {type(e).__name__}: {e}")
            failed.append(name)
            continue
        if (res["meta"].get("mineru") or {}).get("status") == "failed":
            failed.append(name)

    print("=" * 78)
    if failed:
        print(f"完成，但以下论文 MinerU 失败（报告已生成，问答将明确拒绝）：{failed}")
        return 1
    print(f"全部完成：{len(pdfs)} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
