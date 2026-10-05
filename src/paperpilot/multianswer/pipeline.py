"""多答案开放域检索 · **端到端编排**（方向 → rag1 → 取料 → 语料库 → rag2 → 答案）。

## 用户口径的链路（**这是目标架构**）

    ① 用户问**方向问题** ──rag1 论文检索──▶ 拿到 **arxiv-id**
    ② 这 N 个 id         ──**论文取料**──▶ 下载 PDF（+ 解析）
    ③ 切块（生产切块器）  ──▶ **语料库**
    ④ 用户问**多答案开放问题** ──rag2──▶ **返回答案**

## ★ 本模块**只做编排** —— 四段各自的实现都是现成的，这里一个都不重写

| 段 | 现成实现 | 对应 CLI |
|---|---|---|
| ① rag1 论文检索 | `paperpilot.tools.corpus_search.search` | `cli/run_search.py` |
| ② 论文取料 | `paperpilot.tools.arxiv_fetch.fetch_arxiv` + `paperpilot.ingest.ingest` | `cli/run_fetch.py` |
| ③ 切块 | `paperpilot.agents.document_cache.ordered_chunks`（**生产切块器**） | `retrieval/tmp/_r2_parse50.py` 同款 |
| ④ rag2 | `paperpilot.multianswer.answer` + `render` | `cli/eval/run_multianswer.py` |

## ★ 为什么「切块」必须用 `ordered_chunks`（而不是检索视图）

既有 50 篇口径的语料（`retrieval/data/r2dev/prodchunk50/mineru/*.parquet`）就是
`ordered_chunks` 产出的（`_r2_parse50.py:113`），列名见 `_r2_parse50.py:122-127`。

★ 生产问答链路的 `retrieval_chunks` 会在 `ordered_chunks` 之上**再注入 MinerU 表格/公式块**
（两者只在"无 MinerU 产物"时相等）。若这里改用检索视图，**块就不一样** → 语料不同
→ 与既有评测的数字**不可比**。所以本模块**照抄 `_r2_parse50.py` 的口径**
（同切块器、同列名），产出的语料与 `prodchunk50/mineru/*.parquet` **逐块可比**。

## ⚠️ 一个必须一起看的边界

rag1 是**为"交付 top-5"设计**的（`corpus_search.py`：`POOL = 200` 进 CE、`N_OUT = 10`
交给 LLM 精排）→ **取 top-N（N=10~150）本来就出了它的工作区**。
实测 2026-10-05：拿论文**自己的标题**去检索，4/4 都进不了 top-50。
⇒ 所以：
  · **`n=10`** = rag1 在本地**真能给的上限**（诚实口径）；
  · 要更大语料，**要么换检索源、要么加 rag1 预算** —— 见 `docs/RAG2_CORPUS_WIRING.md`。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]

# 切块 parquet 的列（★ 与 `_r2_parse50.py` 逐列对齐，别改）
CHUNK_COLS = ("docid", "cluster", "pdf", "src", "chunk_id", "title_path",
              "n_title", "page", "n_chars", "text")

# 现成的论文簇（"展示"档用；各自 50 篇，生产切块）
_R2DEV = ROOT / "retrieval" / "data" / "r2dev"
_CLUSTER_REL = ("prodchunk50", "mineru", "c{ci}.parquet")
CLUSTER_NAMES = {1: "簇1 · RAG 问答", 2: "簇2 · 指令微调/对齐", 3: "簇3 · 高效/长上下文 Transformer"}


def cluster_parquet(ci: int) -> Path:
    """现成簇的切块 parquet（`ci` = 1/2/3）。★ 这是"展示"档的语料。"""
    return _R2DEV.joinpath(*[p.format(ci=ci - 1) for p in _CLUSTER_REL])


def list_clusters() -> list[dict[str, Any]]:
    """枚举现成簇（供前端渲染"展示"档的选项）。"""
    out = []
    for ci in (1, 2, 3):
        p = cluster_parquet(ci)
        out.append({"ci": ci, "name": CLUSTER_NAMES[ci], "path": str(p),
                    "exists": p.exists()})
    return out


# ── ① rag1 论文检索 ────────────────────────────────────────────────────────


def rag1_search(direction: str, n: int, *, use_llm: bool = True,
                pool: int | None = None, since: str | None = None,
                until: str | None = None,
                verbose: bool = True) -> list[dict[str, Any]]:
    """**① rag1**：方向问题 → topN 篇，返回每篇的 **arxiv_id**（取料的钥匙）。

    直接调生产的 `tools.corpus_search.search`（= 线上工具①，与 `cli/run_search.py` 同源）。
    ⚠️ 生产默认交付数远小于本编排要的 N —— 这里**显式传 k=N**，因为下游要拿它当语料。
    """
    import sys

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from paperpilot.tools import corpus_search as CS

    kw: dict[str, Any] = {}
    if pool is not None:
        kw["pool"] = pool
    if since:
        kw["since"] = since
    if until:
        kw["until"] = until

    if verbose:
        print(f"[① rag1] 「{direction}」→ top {n}（LLM 精排 {'开' if use_llm else '关'}）…",
              flush=True)
    r = CS.search(direction, k=n, use_llm=use_llm, **kw)
    if getattr(r, "degraded", ""):
        print(f"[① rag1] ⚠️ 降级：{r.degraded}", flush=True)
    out = [{"rank": int(getattr(p, "rank", i + 1)),
            "arxiv_id": str(getattr(p, "arxiv_id", "")),
            "title": str(getattr(p, "title", "")),
            "score": float(getattr(p, "score", 0.0))}
           for i, p in enumerate(r.papers)]
    if verbose:
        print(f"[① rag1] 交付 {len(out)} 个 id", flush=True)
    return out


# ── ② 论文取料 ─────────────────────────────────────────────────────────────


def fetch_papers(refs: Sequence[dict[str, Any]], *, force: bool = False,
                 ingest: bool = True, interval: float = 1.0,
                 verbose: bool = True) -> list[dict[str, Any]]:
    """**② 论文取料**：arXiv id → `assets/papers/<id>.pdf`（可选：抓完直接解析）。

    直接调生产的 `tools.arxiv_fetch.fetch_arxiv`（幂等）+ `ingest.ingest`
    （与 `cli/run_fetch.py --ingest` 同源）。

    ⚠️ 失败的篇**不静默丢弃**：逐条打印原因，并在返回里带 `ok=False`。
    """
    import re

    from paperpilot.ingest import ingest as _ingest
    from paperpilot.tools.arxiv_fetch import fetch_arxiv, normalize_arxiv_id

    def base_id(a: str) -> str:
        return re.sub(r"v\d+$", "", str(a or "").strip())

    out: list[dict[str, Any]] = []
    for i, ref in enumerate(refs, 1):
        aid = normalize_arxiv_id(str(ref.get("arxiv_id") or ""))
        if not aid:
            out.append({**ref, "ok": False, "reason": "空 arxiv_id"})
            continue
        r = fetch_arxiv(aid, force=force, min_interval=interval)
        if not r["ok"]:
            if verbose:
                print(f"[② 取料] {i}/{len(refs)} ✗ {aid}：{r['reason']}", flush=True)
            out.append({"docid": base_id(aid), "arxiv_id": aid, "ok": False,
                        "reason": r["reason"]})
            continue
        pdf = str(r["pdf_name"])
        if verbose:
            tag = "命中缓存" if r["cached"] else f"已下载 {r['bytes'] / 1024:.0f}KB"
            print(f"[② 取料] {i}/{len(refs)} ✓ {aid}（{tag}）→ {pdf}", flush=True)
        if ingest:
            try:
                _ingest(pdf, mineru=True, verbose=False)
            except Exception as e:  # noqa: BLE001  解析失败不该中断整条链
                print(f"[② 取料]   ⚠️ {pdf} 解析失败（切块会回退）："
                      f"{type(e).__name__}: {str(e)[:90]}", flush=True)
        out.append({"docid": base_id(aid), "arxiv_id": aid, "pdf": pdf,
                    "cached": bool(r["cached"]), "ok": True})
    ok = [x for x in out if x.get("ok")]
    if verbose:
        print(f"[② 取料] 成功 {len(ok)} / {len(out)} 篇", flush=True)
    return out


# ── ③ 切块 → 语料库 ────────────────────────────────────────────────────────


def chunk_papers(pairs: Sequence[tuple[str, str]], out_path: str | Path, *,
                 cluster: int = 1, src: str = "flow", verbose: bool = True) -> Path:
    """**③ 切块**：生产切块器 `ordered_chunks` → 切块 parquet（`docid` / `text`）。

    `pairs` = `[(docid, pdf 文件名), …]`（**顺序即语料顺序**，保证可复现）。
    ★ 与既有 50 篇口径（`_r2_parse50.py`）**同切块器、同列名** —— 见模块 docstring。
    """
    import pandas as pd

    from paperpilot.agents.document_cache import ordered_chunks

    rows: list[dict[str, Any]] = []
    for i, (docid, pdf) in enumerate(pairs, 1):
        chs = ordered_chunks(pdf)
        for c in chs:
            rows.append(dict(docid=str(docid), cluster=int(cluster), pdf=str(pdf),
                             src=str(src), chunk_id=str(c.chunk_id),
                             title_path=" · ".join(c.title_path or []),
                             n_title=len(c.title_path or []),
                             page=int(c.page_span[0]), n_chars=len(c.text),
                             text=str(c.text)))
        if verbose:
            print(f"[③ 切块] {i}/{len(pairs)} {docid} → {len(chs)} 块", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(CHUNK_COLS)).to_parquet(out, index=False)
    if verbose:
        print(f"[③ 切块] 共 {len(rows)} 块 / {len(pairs)} 篇 → {out}", flush=True)
    return out


def build_index(chunks_parquet: str | Path, *, n: int | None = None,
                docids: Iterable[str] | None = None, verbose: bool = True):
    """**③′ 语料库**：切块 parquet → `CorpusIndex`（rag2 要的语料契约，含向量缓存）。"""
    from paperpilot.multianswer import CorpusIndex

    return CorpusIndex.from_parquet(chunks_parquet, n=n, docids=docids, verbose=verbose)


# ── ④ rag2 + 全链路 ───────────────────────────────────────────────────────


def run_flow(question: str, *, direction: str | None = None, n: int = 10,
             cluster: int | None = None, chunks_parquet: str | Path | None = None,
             out_dir: str | Path | None = None, tag: str = "flow",
             use_llm: bool = True, force_fetch: bool = False, fetch: bool = True,
             chunk: bool = True, extract: bool | None = None,
             b: int | None = None, workers: int | None = None,
             verbose: bool = True) -> dict[str, Any]:
    """**全链路编排**：① 方向 → rag1 → id → ② 取料 → ③ 切块 → ④ rag2 → **答案**。

    三种语料来源（优先级从高到低）：

    - `chunks_parquet=…`：**直接用现成切块**（跳过 ①②③）—— 最快；
    - `cluster=1|2|3`   ：**用现成的簇**（`prodchunk50/mineru/c{ci}.parquet`，50 篇）
                          —— ★ 这是「**展示**」档：语料现成，用来跑通/演示接线；
    - `direction="…"`   ：★ **目标架构**：① rag1 检索 → ② 取料下载 → ③ 切块。

    Returns: `{direction, question, n, papers, pdfs, chunks_parquet, answer, answer_text}`
    """
    from paperpilot.multianswer import answer as _answer
    from paperpilot.multianswer import render as _render

    srcs: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    ck: Path

    # ── 语料来源 A：直接用现成切块 ──
    if chunks_parquet is not None:
        ck = Path(chunks_parquet)
        if not ck.exists():
            raise FileNotFoundError(f"给定切块不存在：{ck}")
        if verbose:
            print(f"[编排] 直接用现成切块 {ck.name}（跳过 ①②③）", flush=True)
        return _answer_and_wrap(question, _answer, _render, ck, srcs, pairs,
                                direction, extract, b, workers, verbose)

    # ── 语料来源 B：「展示」档 —— 用现成的簇 ──
    if cluster is not None:
        ck = cluster_parquet(int(cluster))
        if not ck.exists():
            raise FileNotFoundError(
                f"现成簇不存在：{ck}。★ 语料是可再生产物，需先跑 "
                f"`retrieval/tmp/_r2_parse50.py` 生成（见 docs/RAG2_CORPUS_WIRING.md）")
        if verbose:
            print(f"[编排] 用现成簇{cluster}（{CLUSTER_NAMES[int(cluster)]}）"
                  f"→ {ck.name}（跳过 ①②）", flush=True)
        return _answer_and_wrap(question, _answer, _render, ck, srcs, pairs,
                                direction, extract, b, workers, verbose)

    # ── 语料来源 C：★ 目标架构 —— 方向 → rag1 → id → 取料 → 切块 ──
    if not direction:
        raise ValueError("必须给 `direction`（目标架构）/ `cluster`（展示档）/ `chunks_parquet`")
    refs = rag1_search(direction, n, use_llm=use_llm, verbose=verbose)
    if fetch:
        fetched = fetch_papers(refs, force=force_fetch, verbose=verbose)
        pairs = [(x["docid"], x["pdf"]) for x in fetched if x.get("ok")]
        srcs = fetched
    else:
        import re
        for r in refs:
            base = re.sub(r"v\d+$", "", str(r["arxiv_id"]))
            pairs.append((base, f"{base}.pdf"))
        srcs = [{"docid": d, "pdf": p, "src": "arxiv"} for d, p in pairs]

    if not pairs:
        raise RuntimeError("语料为空：rag1 未交付 / 取料全失败")

    if out_dir is None:
        out_dir = ROOT / "retrieval" / "data" / "rag2_flow" / tag
    ck = Path(out_dir) / f"{tag}.parquet"
    if chunk:
        chunk_papers(pairs, ck, verbose=verbose)
    elif not ck.exists():
        raise FileNotFoundError(f"要跳过切块但语料不存在：{ck}")
    return _answer_and_wrap(question, _answer, _render, ck, srcs, pairs,
                            direction, extract, b, workers, verbose)


def _answer_and_wrap(question, _answer, _render, ck: Path, srcs, pairs,
                     direction, extract, b, workers, verbose) -> dict[str, Any]:
    """公共收口：建语料库 → rag2 → **答案文本**。"""
    if verbose:
        print(f"[④ rag2] 语料 {Path(ck).name} → 多答案开放问题「{question[:36]}…」",
              flush=True)
    idx = build_index(ck, verbose=verbose)
    res = _answer(question, idx, b=b, workers=workers, extract=extract)
    if verbose:
        print(f"[④ rag2] 判 yes {res.get('n_yes')} 篇 / 候选 {res.get('n_papers')} 篇",
              flush=True)
    return {"direction": direction or "", "question": question, "n": len(idx.pdfs),
            "papers": srcs, "pdfs": [p for _, p in pairs],
            "chunks_parquet": str(ck), "n_yes": res.get("n_yes"),
            "yes": [x["pdf"] for x in res.get("papers", [])],
            "answer": res, "answer_text": _render(question, res)}


def save_run(result: dict[str, Any], path: str | Path) -> Path:
    """把编排结果落盘（可回溯：方向、问题、篇集、语料、答案）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "answer"}   # 大对象另存
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
