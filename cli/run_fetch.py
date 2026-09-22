"""获取命令行：arXiv id → 论文库 PDF（可选：抓完直接摄取）。

用法（在仓库根目录）：
    uv run python cli/run_fetch.py 1706.03762                # 抓一篇（幂等）
    uv run python cli/run_fetch.py 1706.03762 --force        # 强制重抓
    uv run python cli/run_fetch.py --ids-file ids.txt        # 批量（每行一个 id）
    uv run python cli/run_fetch.py 1706.03762 --ingest       # 抓完直接走摄取链

退出码：0=全部成功；1=有失败（失败原因逐条打印，不会抛异常中断批量）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools.arxiv_fetch import (  # noqa: E402
    PAPERS_DIR, fetch_arxiv, normalize_arxiv_id, pdf_url,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="arXiv id（现代 `2301.12345` 或老式 `cs/9811009`）")
    ap.add_argument("--ids-file", default=None, help="从文件读 id（每行一个，# 开头跳过）")
    ap.add_argument("--force", action="store_true", help="即使本地已有也重抓")
    ap.add_argument("--timeout", type=int, default=60, help="单次请求超时秒数")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="请求最小间隔秒数（批量时调大，arXiv 要求礼貌抓取）")
    ap.add_argument("--ingest", action="store_true", help="抓完直接调用 ingest（MinerU + 报告）")
    ap.add_argument("--no-mineru", action="store_true", help="--ingest 时跳过 MinerU")
    args = ap.parse_args()

    ids: list[str] = [normalize_arxiv_id(x) for x in args.ids if x.strip()]
    if args.ids_file:
        p = Path(args.ids_file)
        if not p.exists():
            print(f"[错误] 找不到 id 文件：{p}")
            return 1
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(normalize_arxiv_id(s))
    if not ids:
        ap.print_help()
        return 1

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[获取] 论文库：{PAPERS_DIR}")
    print(f"[获取] {len(ids)} 篇 | 间隔 {args.interval}s"
          + ("（强制重抓）" if args.force else ""))

    failed: list[tuple[str, str]] = []
    got: list[str] = []
    for aid in ids:
        print("-" * 78)
        # ⚠️ `pdf_url` 会**校验** id 并抛 FetchError —— 这里只用于展示，
        # 校验失败就退回原样，交给 `fetch_arxiv` 统一走"优雅失败"路径。
        try:
            shown = pdf_url(aid)
        except Exception:  # noqa: BLE001
            shown = "(id 格式待校验)"
        print(f"[获取] {aid}  ← {shown}")
        r = fetch_arxiv(aid, force=args.force, timeout=args.timeout,
                        min_interval=args.interval)
        if not r["ok"]:
            print(f"  [失败] {r['reason']}")
            failed.append((aid, r["reason"]))
            continue
        tag = "命中缓存" if r["cached"] else "已下载"
        print(f"  [成功] {tag} {r['bytes'] / 1024:.0f} KB → {r['pdf_name']}")
        got.append(r["pdf_name"])

    if args.ingest and got:
        from paperpilot.ingest import ingest
        print("=" * 78)
        print(f"[摄取] {len(got)} 篇")
        for name in got:
            print("-" * 78)
            print(f"[摄取] {name}")
            try:
                res = ingest(name, mineru=not args.no_mineru, verbose=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [错误] {type(e).__name__}: {e}")
                failed.append((name, f"ingest: {type(e).__name__}: {e}"))
                continue
            if (res["meta"].get("mineru") or {}).get("status") == "failed":
                failed.append((name, "MinerU 失败（报告已生成）"))

    print("=" * 78)
    print(f"完成：成功 {len(got)} 篇，失败 {len(failed)} 篇")
    for aid, why in failed:
        print(f"  ✗ {aid}：{why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
