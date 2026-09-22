"""语料级 chunk 检索尺子（组内 5 篇，**零 LLM**）：Recall@k / MRR@k。

## 口径

- **候选池** = 整组语料的扁平 chunk 空间（`MultiChunkIndex`，与线上 L3 同源、
  默认走 `retrieval_chunks` 检索视图）—— 所以"串篇干扰"会被真实计入。
- **gold** = 题集里的 `evidence` 引文（**逐字 PDF 片段**）→ 子串定位到 chunk。
  定位用**对 PDF 伪影鲁棒**的归一化（行尾连字符折行 / 印刷体标点，多口径比对）。
- Recall@k = gold 里**任一** chunk 落在 top-k；MRR@k = 首个 gold 的位次（<k 才计）。
- 三种检索：`vec` / `bm25` / `hyb`（RRF 等权，与线上一致）。

## 为什么不复用 `cli/run_retrieval_eval.py`

那个的 gold 来源**绑死 QASPER**（`qa.sources.qasper`），而本组语料是**真 arXiv PDF**，
gold 在 `retrieval/tmp/<group>/` 的题集里（`evidence` 刻意不导出给端到端 runner，
就是留给检索层当 gold 的 —— 见 `_export_questions.py`）。定位与指标逻辑沿用同一套。

用法：
    python cli/run_chunk_recall.py --group group1 [--ks 5 10 20] [--out <md>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))   # 复用校验器的鲁棒归一化
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import numpy as np  # noqa: E402

from _validate_questions import norm, sources_variants  # noqa: E402
from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.agents.embedder import (BM25Index,  # noqa: E402
                                        MultiChunkIndex, encode_query)

RRF_K = 60          # 与线上一致


def locate_in(ntexts: list[str], variants: list[list[str]], quote: str) -> set[int]:
    """引文 → 该篇 chunk 下标集（多口径子串命中；命中多块=跨块 gold）。"""
    q = norm(quote)
    if not q:
        return set()
    hit = {i for i, vs in enumerate(variants) if any(q in v for v in vs)}
    if hit:
        return hit
    # 兜底：引文可能跨块 —— 用前 60 字锚（足够独特，又不至于落空）
    a = q[:60]
    return {i for i, vs in enumerate(variants) if any(a in v for v in vs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--mrr-k", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gdir = ROOT / "retrieval" / "tmp" / args.group
    grp = json.loads((gdir / "_group.questions.json").read_text(encoding="utf-8"))
    corpus: list[str] = [str(p) for p in grp["corpus"]]
    pdfs = [p if p.endswith(".pdf") else f"{p}.pdf" for p in corpus]

    # ── 扁平候选池（与 MultiChunkIndex 的向量顺序必须一致）──
    flat, spans = [], {}
    for pdf in pdfs:
        cs = list(retrieval_chunks(pdf))
        spans[pdf] = (len(flat), len(cs))
        flat.extend(cs)
    vecs = MultiChunkIndex(pdfs).vectors()
    assert vecs.shape[0] == len(flat), f"向量数 {vecs.shape[0]} ≠ chunk 数 {len(flat)}"
    bm = BM25Index([c.text for c in flat])
    print(f"语料 {len(pdfs)} 篇 ｜ 扁平 {len(flat)} 块 ｜ 向量 {vecs.shape}")

    # ── 装载题目：单篇 70 题 + 组级 20 题（后者 evidence 带 paper 字段）──
    items: list[dict] = []
    for stem in corpus:
        f = gdir / f"{stem}.questions.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))   # dict（含 _note）或裸 list
        for q in (d.get("questions") if isinstance(d, dict) else d) or []:
            items.append({**q, "scope": "single", "home": f"{stem}.pdf"})
    for q in grp["questions"]:
        if q.get("evidence"):
            items.append({**q, "scope": "multi"})
    print(f"题目 {len(items)} 道（single/multi="
          f"{sum(1 for q in items if q['scope'] == 'single')}/"
          f"{sum(1 for q in items if q['scope'] == 'multi')}）")

    ntexts = {pdf: [norm(c.text) for c in flat[o:o + n]] for pdf, (o, n) in spans.items()}
    # ⚠️ 必须**每块一份**的变体列表：拼成一整篇再比对会退化成「逐字符找」
    variants = {pdf: [sources_variants(t) for t in ntexts[pdf]] for pdf in pdfs}

    cols = ("vec", "bm25", "hyb")
    hit = {s: {c: {k: 0 for k in args.ks} for c in cols}
           for s in ("single", "multi", "all")}
    mrr = {s: {c: 0.0 for c in cols} for s in ("single", "multi", "all")}
    den = {s: 0 for s in ("single", "multi", "all")}
    fail: list[str] = []
    n_page = 0      # 走「按页码兜底」定位的题数（多为表格题）

    t0 = time.time()
    for q in items:
        # gold：逐条引文按 paper 定位 → 映射到扁平下标
        cg: set[int] = set()
        by_page = False
        for ev in q.get("evidence") or []:
            paper = str(ev.get("paper") or q.get("home") or "")
            if paper and not paper.endswith(".pdf"):
                paper += ".pdf"            # 跨篇题的 paper 字段不带后缀
            if paper not in spans:
                continue
            off, nch = spans[paper]
            loc = locate_in(ntexts[paper], variants[paper], str(ev.get("quote") or ""))
            if not loc and ev.get("page"):
                # 兜底：**按页码**取该篇内该页的全部块。
                # 为什么需要：表格题的 gold 引文取自 pymupdf 文本形态（逐格数字），
                # 而上线检索视图走 MinerU（表格是另一种形态）→ 子串必然对不上 ✗
                loc = {i for i in range(nch)
                       if int(getattr(flat[off + i], "page", 0) or 0) == int(ev["page"])}
                by_page = by_page or bool(loc)
            cg |= {off + i for i in loc}
        if by_page:
            n_page += 1
        if not cg:
            fail.append(f"{q.get('qid')}(no-locate)")
            continue

        n = len(flat)
        qv = encode_query(str(q["question"]))
        vs = (qv @ vecs.T).astype("float64")
        bs = np.asarray(bm.score(str(q["question"])), dtype="float64")
        ov = np.argsort(-vs, kind="stable")
        ob = np.argsort(-bs, kind="stable")
        rv = np.empty(n); rb = np.empty(n)
        rv[ov] = np.arange(n); rb[ob] = np.arange(n)
        rrf = 0.5 / (RRF_K + rv + 1) + 0.5 / (RRF_K + rb + 1)     # 等权，与线上一致
        orders = {"vec": ov, "bm25": ob, "hyb": np.argsort(-rrf, kind="stable")}

        for s in (q["scope"], "all"):
            den[s] += 1
        for col, order in orders.items():
            first = next((p for p in range(n) if int(order[p]) in cg), None)
            if first is None:
                continue
            for s in (q["scope"], "all"):
                for k in args.ks:
                    if first < k:
                        hit[s][col][k] += 1
                if first < args.mrr_k:
                    mrr[s][col] += 1.0 / (first + 1)

    # ── 报告 ──
    L = [f"# 语料级 chunk 检索尺子（`{args.group}`）", "",
         f"> 语料 {len(pdfs)} 篇 ｜ 扁平候选池 **{len(flat)}** 块（=上线 L3 同源检索视图）",
         f"> 题目 {len(items)} 道 ｜ gold 定位失败 **{len(fail)}** 道"
         + (f"（{'；'.join(fail[:4])}）" if fail else ""),
         f"> 三种检索：vec（向量）/ bm25 / hyb（RRF 等权 α=0.5，与线上一致）",
         f"> gold 定位：**按页码兜底 {n_page} 道**（表格题：引文取自 pymupdf 形态，"
         f"检索视图走 MinerU → 子串对不上，退化为该页全部块；口径偏宽）", ""]
    for s, name in (("single", "单篇题（gold 都在同一篇内）"),
                    ("multi", "跨篇题（gold 跨 ≥2 篇）"), ("all", "合计")):
        d = den[s]
        L += [f"## {name}（分母 {d}）",
              f"| 检索器 | {' | '.join(f'R@{k}' for k in args.ks)} | MRR@{args.mrr_k} |",
              "|---" * (len(args.ks) + 2) + "|"]
        for c in cols:
            if not d:
                L.append(f"| {c} | " + " | ".join("—" for _ in range(len(args.ks) + 1)) + " |")
                continue
            rr = " | ".join(f"{hit[s][c][k]/d:.3f}" for k in args.ks)
            L.append(f"| {c} | {rr} | {mrr[s][c]/d:.3f} |")
        L.append("")
    out = Path(args.out) if args.out else gdir / f"CHUNK_RECALL_{args.group}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n完成 {time.time()-t0:.1f}s → {out}")
    print("\n".join(L[-10:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
