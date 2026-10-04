"""下载 LoFT 的 RAG-QAMPARI（GCS 直链）并解剖：语料规模 / 问题数 / 每题 gold 答案数 / 字段。

LoFT: `https://storage.googleapis.com/loft-bench/rag/qampari.zip`
README 口径：Task Type = `multi_value_rag`，主指标 = **`subspan_em`**；
32k / 128k / 1m = **上下文长度档位**（对应"语料能不能塞进窗口"）
→ 正好是我们要的 **S/C 规模扫描**。
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parents[1] / "data" / "loft"
URL = "https://storage.googleapis.com/loft-bench/rag/qampari.zip"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "qampari.zip"
    if not dst.exists():
        print(f"下载 {URL} …", flush=True)
        with urllib.request.urlopen(URL, timeout=180) as r, open(dst, "wb") as f:
            f.write(r.read())
    print(f"zip {dst.stat().st_size / 1e6:.1f} MB")

    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
        print(f"\n【条目 {len(names)}】")
        for n in names[:40]:
            print(f"  {n}")
        # 逐档解剖
        for length in ("32k", "128k", "1m"):
            cname = next((n for n in names if f"/{length}/" in n and "corpus" in n), None)
            if not cname:
                continue
            corpus = [json.loads(x) for x in z.read(cname).decode("utf-8").splitlines() if x.strip()]
            body = [len(str(c.get("passage_text", ""))) for c in corpus]
            ttl = [len(str(c.get("title_text", ""))) for c in corpus]
            tot = sum(body) + sum(ttl)
            print(f"\n===== 档位 {length} =====")
            print(f"  语料：{len(corpus):,} 条 ｜ 字段 {list(corpus[0].keys())}")
            print(f"        段落字符：中位 {int(sorted(body)[len(body) // 2]):,} ｜ 合计 {tot:,}"
                  f" ≈ **{tot / 4 / 1000:.0f}k token**（按 4 字符/token 粗估，档位名 = {length}）")
            for split in ("dev_queries", "test_queries"):
                qname = next((n for n in names if f"/{length}/" in n and split in n), None)
                if not qname:
                    print(f"  {split}: **缺失**")
                    continue
                qs = [json.loads(x) for x in z.read(qname).decode("utf-8").splitlines() if x.strip()]
                na = [len(q.get("answers", [])) for q in qs]
                nq = sum(len(q.get("metadata", {}).get("qrels", [])) for q in qs)
                print(f"  {split}: {len(qs):,} 题 ｜ 每题 gold 答案 均值 {sum(na) / len(na):.2f}"
                      f"（中位 {int(sorted(na)[len(na) // 2])}，max {max(na)}）"
                      f" ｜ qrels 段落/题 {nq / len(qs):.2f}"
                      f" ｜ 单答案题 {sum(1 for x in na if x == 1) / len(na):.1%}")
            print(f"  语料总段数 ÷ 题数 = {len(corpus) / 100:.1f} 段/题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
