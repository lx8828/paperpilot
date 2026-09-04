"""Claim 噪声筛查：识别表格/数字碎片句，保存 out_claims/<pdf>.noise.json。

用法：
    uv run python run_scan_noise.py 2608.31079v1.pdf
    uv run python run_scan_noise.py --all
    uv run python run_scan_noise.py --all --force
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
from paperpilot.tools.quality import is_abnormal, scan_claims

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
CLAIMS_DIR = ROOT / "out_claims"


def scan_one(pdf: str, force: bool) -> dict[str, Any] | None:
    stem = Path(pdf).stem
    claim_file = CLAIMS_DIR / f"{stem}.claims.json"
    if not claim_file.exists():
        print(f"  [跳过] {pdf}: 无 claims")
        return None
    cache = CLAIMS_DIR / f"{stem}.noise.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  [缓存] {pdf}: {len(data['noisy'])} 条噪声")
        return data
    claims = json.loads(claim_file.read_text(encoding="utf-8"))["claims"]
    print(f"  扫描 {pdf}（{len(claims)} claims，两轮判定取交集）…")
    noisy = scan_claims(claims)
    data = {"pdf": pdf, "noisy": noisy}
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    flag = " ⚠比例异常，疑似误判，请人工复核" if is_abnormal(len(noisy), len(claims)) else ""
    print(f"  → {len(noisy)}/{len(claims)} 条噪声，已保存 {cache.name}{flag}")
    return data


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", type=int, default=6,
                    help="每篇打印前 N 条噪声样例")
    args = ap.parse_args()

    if args.all:
        pdfs = []
        for f in sorted(CLAIMS_DIR.glob("*.claims.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            pdfs.append(data["pdf"])
    else:
        pdfs = list(args.pdfs)
    if not pdfs:
        print("请指定 PDF 或 --all")
        return 1
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    tot = 0
    tot_claims = 0
    rows = []
    for pdf in pdfs:
        data = scan_one(pdf, args.force)
        if data is None:
            continue
        n = len(data["noisy"])
        tot += n
        rows.append((pdf, n))
        for item in data["noisy"][: args.samples]:
            print(f"      ✗ {item['claim_id']}: {item['reason'][:50]}")

    if rows:
        print("\n" + "=" * 50)
        for pdf, n in rows:
            print(f"  {pdf:<20s} noisy={n}")
        print(f"  合计 noisy={tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
