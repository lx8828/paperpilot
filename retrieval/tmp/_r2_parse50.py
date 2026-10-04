"""**扩语料解析 + 生产切块**（150 篇 = 47 原有 + 103 干扰项，**MinerU 关闭**）

## 为什么先关 MinerU
用户口径：**先用现有工具看效果，确认没问题再上 MinerU**。
· 本步走 pymupdf 路（快：150 篇 ~10 分钟）；MinerU 路是 `_r2_mineru_run.py`（150 篇要数小时）
· 两条路**共用同一个生产切块器** `document_cache.ordered_chunks` → 切块逻辑一致，
  差别只在"PDF → blocks"那一步（pymupdf 拿不到表格数值，MinerU 能）

## 口径必须统一
150 篇**全部重解析**（不复用旧的 47 篇产物）→ 否则"原篇用 MinerU、干扰项用 pymupdf"
会把"语料规模效应"和"解析器效应"混在一起。

产物：
  `data/r2dev/pdftext50/c{ci}.parquet`           每篇 pymupdf 纯文本
  `data/r2dev/prodchunk50/pymupdf/c{ci}.parquet` 每篇生产切块
  `data/r2dev/corpus50/pdf_map_all.json`         docid → pdf（47 原有 + 103 新增）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_parse50.py --limit 3   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_parse50.py
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

# ★ 口径 = **生产口径（MinerU 开）**：与现有 47 篇一致，150 篇同一解析器。
#    需要 pymupdf 快路径做对照时，显式 `PAPERPILOT_MINERU=0 PAPERPILOT_USE_MINERU=0`。
os.environ.setdefault("PAPERPILOT_MINERU", "1")
os.environ.setdefault("PAPERPILOT_USE_MINERU", "1")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

DEV = HERE / "data" / "r2dev"
CORPUS = DEV / "corpus50"
PAPERS = ROOT / "assets" / "papers"
TXT_OUT = DEV / "pdftext50"
CHK_OUT = DEV / "prodchunk50"
MAP_ALL = CORPUS / "pdf_map_all.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from paperpilot.agents.document_cache import MAX_CHUNK_LEN, mineru_status, ordered_chunks
    from paperpilot.tools.pdf_parser import parse_pdf
    from paperpilot.workflow import _ensure_env

    _ensure_env()

    # ── 合并 PDF 映射：原有 47（pdf_map.json）+ 新增 103（corpus50/pdf_map.json）──
    old = json.loads((DEV / "pdf_map.json").read_text(encoding="utf-8"))
    new = json.loads((CORPUS / "pdf_map.json").read_text(encoding="utf-8"))
    amap = {}
    for ci in range(3):
        d = pd.read_parquet(CORPUS / f"c{ci}.parquet")
        for _, r in d.iterrows():
            amap[str(r["docid"])] = {"cluster": ci + 1, "src": str(r["src"]),
                                     "arxiv_id": str(r["arxiv_id"]), "title": str(r["title"])}
    mp: dict[str, dict] = {}
    for docid, meta in amap.items():
        v = new.get(docid) or old.get(docid) or {}
        if v.get("ok") and v.get("pdf"):
            mp[docid] = {"pdf": v["pdf"], "cluster": meta["cluster"], "src": meta["src"],
                         "arxiv_id": meta["arxiv_id"], "ok": True,
                         "note": v.get("note", "")}
    MAP_ALL.write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8")

    items = sorted(mp.items(), key=lambda kv: (kv[1]["cluster"], kv[0]))
    if args.limit:
        items = items[: args.limit]

    # 解析路径 tag：MinerU 开 → `mineru`，关 → `pymupdf`（与现有 47 篇的目录口径一致）
    tag = ("mineru" if os.environ.get("PAPERPILOT_USE_MINERU", "1") == "1"
           and os.environ.get("PAPERPILOT_MINERU", "1") == "1" else "pymupdf")
    TXT_OUT.mkdir(parents=True, exist_ok=True)
    (CHK_OUT / tag).mkdir(parents=True, exist_ok=True)
    by_src = {}
    for _, v in items:
        by_src[v["src"]] = by_src.get(v["src"], 0) + 1
    print("=" * 116)
    print(f"【扩语料解析】{len(items)} 篇 ｜ 组成 {by_src} ｜ MAX_CHUNK_LEN={MAX_CHUNK_LEN}")
    print(f"  MinerU=**{'on' if tag == 'mineru' else 'off'}**（{tag} 路）｜ "
          f"text→{TXT_OUT.name} ｜ chunks→{CHK_OUT.name}/{tag}")
    print(f"  {'docid':<7}{'c':>2}{'src':>10}{'页块':>7}{'文本字符':>11}{'块数':>6}"
          f"{'块/篇':>7}{'字符/块中位':>13}{'带标题':>8}{'秒':>7}")

    txt_rows, chk_rows = [], []
    t00 = time.time()
    for i, (docid, v) in enumerate(items, 1):
        pdf, cl = v["pdf"], v["cluster"]
        t0 = time.time()
        try:
            res = parse_pdf(str(PAPERS / pdf))
            blocks = res.get("blocks") or []
            text = "\n".join(str(b.get("text") or "") for b in blocks)
            chs = ordered_chunks(pdf)
            mstat, mnote = mineru_status(pdf)
        except Exception as e:  # noqa: BLE001
            print(f"  {docid:<7}{cl:>2}{v['src']:>10}  ❌ {type(e).__name__}: {str(e)[:80]}")
            continue
        txt_rows.append(dict(docid=docid, cluster=cl, pdf=pdf, src=v["src"],
                             n_blocks=len(blocks), text=text, n_chars=len(text),
                             mineru=mstat, mineru_note=mnote))
        for c in chs:
            chk_rows.append(dict(docid=docid, cluster=cl, pdf=pdf, src=v["src"],
                                 chunk_id=str(c.chunk_id),
                                 title_path=" · ".join(c.title_path or []),
                                 n_title=len(c.title_path or []),
                                 page=int(c.page_span[0]), n_chars=len(c.text),
                                 text=str(c.text)))
        nch = [len(c.text) for c in chs]
        med = int(pd.Series(nch).median()) if nch else 0
        nt = sum(1 for c in chs if c.title_path) / max(len(chs), 1)
        print(f"  {docid:<7}{cl:>2}{v['src']:>10}{len(blocks):>7}{len(text):>11,}{len(chs):>6}"
              f"{(len(chs) / max(len(blocks), 1)):>7.2f}{med:>13,}{nt:>8.0%}"
              f"{time.time() - t0:>6.1f}s", flush=True)

    d_txt, d_chk = pd.DataFrame(txt_rows), pd.DataFrame(chk_rows)
    if not len(d_txt):
        print("  ⚠️ 无成功解析")
        return 1
    for ci in sorted(d_txt["cluster"].unique()):
        d_txt[d_txt.cluster == ci].to_parquet(TXT_OUT / f"c{ci - 1}.parquet", index=False)
        d_chk[d_chk.cluster == ci].to_parquet(CHK_OUT / tag / f"c{ci - 1}.parquet",
                                              index=False)

    print(f"\n  【汇总】{len(d_txt)} 篇解析成功 / {len(items)} 篇 ｜ "
          f"{len(d_chk)} 块（{time.time() - t00:.0f}s）")
    print(f"  {'簇':>3}{'篇':>5}{'块':>7}{'块/篇':>8}{'字符/块中位':>13}"
          f"{'文本字符/篇中位':>17}{'带标题':>9}")
    for ci in sorted(d_txt["cluster"].unique()):
        t = d_txt[d_txt.cluster == ci]
        c = d_chk[d_chk.cluster == ci]
        print(f"  {ci:>3}{len(t):>5}{len(c):>7}{len(c) / max(len(t), 1):>8.1f}"
              f"{int(c['n_chars'].median()):>13,}{int(t['n_chars'].median()):>17,}"
              f"{(c['n_title'] > 0).mean():>9.0%}")
    print(f"\n  【口径】150 篇统一走 **{tag}**（与现有 47 篇一致）→ 语料规模效应与解析器效应不混。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
