"""**Step 2｜47 篇 PDF → 解析 + 生产切块**（三臂对照的第一半）

产物：
  `data/r2dev/pdftext/c{ci}.parquet`   每篇 PDF 的 pymupdf 纯文本（供"固定窗"臂）
  `data/r2dev/prodchunk/c{ci}.parquet` 每篇 PDF 的**生产切块**（`document_cache.ordered_chunks` → 章节段落切块）

⚠️ MinerU 关闭（`PAPERPILOT_USE_MINERU=0`）→ 走 pymupdf 路（快；MinerU 太慢）。
   诚实标注：pymupdf 拿不到表格数值 → 表格类问题会退化（与生产"降级"态同）。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_parse_pdfs.py --limit 3   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_parse_pdfs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ **不要**在这里强制关 MinerU：生产默认 `USE_MINERU=1`（MinerU 骨架 + 表格在内）。
#    需要 pymupdf 对照时，用 env 显式关：PAPERPILOT_USE_MINERU=0 PAPERPILOT_MINERU=0
os.environ.setdefault("PAPERPILOT_USE_MINERU", "1")
os.environ.setdefault("PAPERPILOT_MINERU", "1")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

from paperpilot.agents.document_cache import MAX_CHUNK_LEN, ordered_chunks  # noqa: E402
from paperpilot.tools.pdf_parser import parse_pdf  # noqa: E402
from paperpilot.workflow import _ensure_env  # noqa: E402

PAPERS = ROOT / "assets" / "papers"
PDFMAP = HERE / "data" / "r2dev" / "pdf_map.json"
ARX = HERE / "data" / "r2dev" / "arxiv_map.json"
TXT_OUT = HERE / "data" / "r2dev" / "pdftext"
CHK_OUT = HERE / "data" / "r2dev" / "prodchunk"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出子目录标签（默认按解析路径自动：mineru/pymupdf）")
    args = ap.parse_args()
    _ensure_env()
    from paperpilot.agents.document_cache import mineru_status

    tag = args.tag or ("mineru" if os.environ.get("PAPERPILOT_USE_MINERU", "1") == "1"
                       and os.environ.get("PAPERPILOT_MINERU", "1") == "1" else "pymupdf")
    txt_dir = TXT_OUT if tag != "mineru" else TXT_OUT
    chk_dir = CHK_OUT / tag
    chk_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [输出] 切块 → {chk_dir}")
    print(f"  [路径] PAPERPILOT_USE_MINERU={os.environ.get('PAPERPILOT_USE_MINERU', '1')}"
          f"  PAPERPILOT_MINERU={os.environ.get('PAPERPILOT_MINERU', '1')}")

    pm = json.loads(PDFMAP.read_text(encoding="utf-8"))
    am = json.loads(ARX.read_text(encoding="utf-8"))
    items = [(d, v) for d, v in pm.items() if v["ok"]]
    items.sort(key=lambda kv: (int(am[kv[0]]["cluster"]), kv[0]))
    if args.limit:
        items = items[: args.limit]

    TXT_OUT.mkdir(parents=True, exist_ok=True)
    CHK_OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 112)
    print(f"【Step 2｜PDF → 解析 + 生产切块】{len(items)} 篇 ｜ MAX_CHUNK_LEN={MAX_CHUNK_LEN}"
          f" ｜ MinerU=off（pymupdf）")
    print(f"  {'docid':<7}{'簇':>3}{'页/块':>8}{'文本字符':>11}{'块数':>6}"
          f"{'块/篇':>7}{'字符/块(中位)':>14}{'带标题':>8}{'耗时':>7}")

    txt_rows, chk_rows = [], []
    t00 = time.time()
    for i, (docid, v) in enumerate(items, 1):
        pdf = v["pdf"]
        cl = int(am[docid]["cluster"])
        t0 = time.time()
        try:
            res = parse_pdf(str(PAPERS / pdf))
            blocks = res.get("blocks") or []
            text = "\n".join(str(b.get("text") or "") for b in blocks)
        except Exception as e:  # noqa: BLE001
            print(f"  {docid:<7}{cl:>3}  ❌ parse_pdf 失败：{type(e).__name__}: {str(e)[:70]}")
            continue
        try:
            chs = ordered_chunks(pdf)
            mstat, mnote = mineru_status(pdf)
        except Exception as e:  # noqa: BLE001
            print(f"  {docid:<7}{cl:>3}  ❌ ordered_chunks 失败：{type(e).__name__}")
            continue
        txt_rows.append(dict(docid=docid, cluster=cl, pdf=pdf, n_blocks=len(blocks),
                             text=text, n_chars=len(text), mineru=mstat, mineru_note=mnote))
        for c in chs:
            chk_rows.append(dict(docid=docid, cluster=cl, pdf=pdf, chunk_id=str(c.chunk_id),
                                 title_path=" · ".join(c.title_path or []),
                                 n_title=len(c.title_path or []),
                                 page=int(c.page_span[0]), n_chars=len(c.text),
                                 text=str(c.text)))
        nch = [len(c.text) for c in chs]
        med = int(pd.Series(nch).median()) if nch else 0
        nt = sum(1 for c in chs if c.title_path) / max(len(chs), 1)
        print(f"  {docid:<7}{cl:>3}{len(blocks):>8}{len(text):>11,}{len(chs):>6}"
              f"{(len(chs) / max(len(blocks), 1)):>7.2f}{med:>14,}{nt:>8.0%}"
              f"{time.time() - t0:>6.1f}s", flush=True)

    d_txt = pd.DataFrame(txt_rows)
    d_chk = pd.DataFrame(chk_rows)
    for ci in sorted(d_txt["cluster"].unique()):
        d_txt[d_txt.cluster == ci].to_parquet(TXT_OUT / f"c{ci - 1}.parquet", index=False)
        d_chk[d_chk.cluster == ci].to_parquet(chk_dir / f"c{ci - 1}.parquet", index=False)

    print("\n" + "=" * 112)
    print(f"【汇总】{len(d_txt)} 篇解析成功 ｜ {len(d_chk)} 块（生产切块·{tag}）｜ {time.time() - t00:.0f}s")
    if "mineru" in d_txt.columns:
        vc = d_txt["mineru"].value_counts().to_dict()
        print(f"  MinerU 状态分布：{vc}"
              f"  ← ✅ok = 用 MinerU 骨架（表格在内）；⚠️degraded = 退回 pymupdf（丢表值）")
    print(f"  {'簇':>3}{'篇':>5}{'文本字符':>13}{'块数':>7}{'块/篇':>8}{'字符/块(中位)':>14}{'≤4000':>8}")
    for ci in sorted(d_chk["cluster"].unique()):
        g = d_chk[d_chk.cluster == ci]
        t = d_txt[d_txt.cluster == ci]
        print(f"  {ci:>3}{len(t):>5}{int(t['n_chars'].sum()):>13,}{len(g):>7}"
              f"{len(g) / len(t):>8.1f}{int(g['n_chars'].median()):>14,}"
              f"{(g['n_chars'] <= MAX_CHUNK_LEN).mean():>8.1%}")
    print(f"  合计 {len(d_chk)} 块 ｜ 对照：LitSearch 固定窗 3,579 块（54~64/篇）")
    print(f"\n  切出的标题路径 top 12（共 {d_chk[d_chk.n_title > 0]['title_path'].nunique()} 种）：")
    for k, n in d_chk[d_chk.n_title > 0]["title_path"].value_counts().head(12).items():
        print(f"    {n:>5}×  {str(k)[:86]}")
    print(f"\n→ {TXT_OUT} ｜ {CHK_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
