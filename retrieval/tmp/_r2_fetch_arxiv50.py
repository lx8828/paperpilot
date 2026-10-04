"""**下载扩语料的 103 篇干扰项 PDF**（全在 arXiv → 必然可下载）

约定：PDF 存 `assets/papers/{arxiv_base}.pdf`（与现有 47 篇同目录、同命名）。
产物：`data/r2dev/corpus50/pdf_map.json` —— docid → {pdf, arxiv_id, ok}

要点：
· URL `https://arxiv.org/pdf/{base}`（arXiv 会 302 到实际版本）
· 校验首 5 字节 `%PDF`（避免把 HTML 错误页当 PDF 存下来）
· 幂等：已存在且 >20KB 跳过（**可续跑**）
· 6 线程并发 + 3 次退避（arXiv 对同一 IP 有速率限制）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fetch_arxiv50.py --limit 4   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fetch_arxiv50.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEV = HERE / "data" / "r2dev" / "corpus50"
PAPERS = ROOT / "assets" / "papers"
OUT = DEV / "pdf_map.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = 30
MIN_BYTES = 20000


def base_id(a: str) -> str:
    return re.sub(r"v\d+$", "", str(a or "").strip())


def fetch(url: str, dest: Path) -> tuple[bool, str]:
    for a in range(3):
        if a:
            time.sleep(2.5 * a)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept": "application/pdf,*/*"})
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with op.open(req, timeout=TIMEOUT) as r:
                data = r.read()
            if not data[:5].startswith(b"%PDF"):
                return False, f"非 PDF（{len(data)}B）"
            dest.write_bytes(data)
            return True, f"{len(data) / 1024:.0f}KB"
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return False, f"HTTP {e.code}"
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
    return False, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    PAPERS.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[str, str, Path]] = []
    for ci in range(3):
        d = pd.read_parquet(DEV / f"c{ci}.parquet")
        d = d[d["src"] == "distractor"]
        for _, r in d.iterrows():
            b = base_id(r["arxiv_id"])
            todo.append((str(r["docid"]), b, PAPERS / f"{b}.pdf"))
    if args.limit:
        todo = todo[: args.limit]

    have = [t for t in todo if t[2].exists() and t[2].stat().st_size > MIN_BYTES]
    go = [t for t in todo if t not in have]
    print("=" * 106)
    print(f"【下载干扰项 PDF】共 {len(todo)} 篇 ｜ 已存在 {len(have)} ｜ 待下 **{len(go)}**")
    print(f"  目标目录 {PAPERS}")

    mp: dict[str, dict] = {}
    for docid, b, dest in have:
        mp[docid] = {"pdf": dest.name, "arxiv_id": b, "ok": True, "note": "已存在"}

    t0 = time.time()
    if go:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch, f"https://arxiv.org/pdf/{b}", dest): (docid, b, dest)
                    for docid, b, dest in go}
            for i, f in enumerate(futs, 1):
                docid, b, dest = futs[f]
                try:
                    ok, note = f.result()
                except Exception as e:  # noqa: BLE001
                    ok, note = False, type(e).__name__
                mp[docid] = {"pdf": dest.name if ok else "", "arxiv_id": b, "ok": ok,
                             "note": note}
                mark = "✓" if ok else "✗"
                print(f"  [{i:>3}/{len(go)}] {mark} {docid:<7}{b:<12}{note}",
                      flush=True)

    OUT.write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for v in mp.values() if v["ok"])
    print(f"\n【完成】{n_ok}/{len(todo)} 篇可用（{time.time() - t0:.0f}s）→ {OUT.name}")
    bad = [k for k, v in mp.items() if not v["ok"]]
    if bad:
        print(f"  ⚠️ 失败 {len(bad)} 篇：{bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
