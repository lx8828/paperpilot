"""拉取 A 桶（输入通道缺口）涉及的 arXiv 原始 PDF，供 MinerU 解析表格。

背景：QASPER 的 figures_and_tables 只提供 PNG + caption span，**表格数值无文本形式**，
所以这些题的 gold（表里的数）在 QASPER 文本通道中不可达。原文都在 arXiv（paper id = arXiv id），
因此可以拉原始 PDF → MinerU 解析表格 → 补进输入通道。

产物：assets/papers/<arxiv_id>v1.pdf（与本地论文命名一致，stem = <id>v1）
用法：uv run python qa/recall/_fetch_arxiv.py qa/recall/_attrib_500_20260911.json
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
DEST = ROOT / "assets" / "papers"
UA = {"User-Agent": "Mozilla/5.0 (compatible; PaperPilot-eval/1.0; research use)"}


def fetch_one(pid: str) -> tuple[str, str]:
    out = DEST / f"{pid}v1.pdf"
    if out.exists() and out.stat().st_size > 20_000:
        return pid, f"skip(已有 {out.stat().st_size // 1024}KB)"
    url = f"https://arxiv.org/pdf/{pid}"
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                return pid, f"FAIL(非 PDF，前 40 字节={data[:40]!r})"
            out.write_bytes(data)
            return pid, f"OK {len(data) // 1024}KB"
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return pid, f"FAIL({type(e).__name__}: {e})"
            time.sleep(3 * attempt)
    return pid, "FAIL(unknown)"


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "qa/recall/_attrib_500_20260911.json"
    payload = json.loads(Path(src).read_text(encoding="utf-8"))
    pids = sorted({x["pid"].replace("qasper_", "").replace(".qpdf", "")
                   for x in payload["A 输入通道缺口"]})
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"待拉 {len(pids)} 篇 → {DEST}")
    ok = fail = 0
    for i, pid in enumerate(pids, 1):
        pid, msg = fetch_one(pid)
        if msg.startswith("FAIL"):
            fail += 1
        else:
            ok += 1
        print(f"  [{i}/{len(pids)}] {pid} {msg}", flush=True)
        time.sleep(1.2)  # 对 arXiv 友好
    print(f"\n完成：成功/已存在 {ok}，失败 {fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
