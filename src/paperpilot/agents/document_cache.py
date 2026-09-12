"""QA 层的正文 chunk 文档缓存（进程级，跨节点复用 parse）。

为什么独立成模块：L3 全文检索、ChunkIndex 向量化都要用到"这篇论文切出的
extractable chunks"——之前各节点各自 parse，浪费。这里统一 lru_cache：
同一 pdf 首次 parse+chunk，之后所有节点复用。

两套视图（2026-09-10 起，**同一批 chunk_id，只是检索文本更厚**）：
    ordered_chunks(pdf)    报告/溯源视图：pymupdf 切块，claims / report / cites 用它
    retrieval_chunks(pdf)  检索视图：在上面的基础上，把 MinerU 的表格/公式文本
                           按页注入到该页首个 chunk → 索引与作答用它
    两者 chunk_id / 页码完全一致 → cites 只有一套 id 空间，溯源闭环不受影响。
    QASPER（无 PDF）与无 MinerU 产物的论文：retrieval_chunks == ordered_chunks。

约定（与 chunker/analyzer 一致）：
    top_section(chunk)  → chunk 所属顶层节名（title_path[0] 的 "· " 后半段），
    与 report.groups 的 sections 字段 / ClaimHit.home_section 统一。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from paperpilot.models.schema import Chunk
from paperpilot.tools import analyzer
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.pdf_parser import parse_pdf
from paperpilot.qasper_source import load_papers, build_chunks

# document_cache.py → agents/ → paperpilot/ → src/ → 根
ROOT = Path(__file__).resolve().parents[3]
PAPERS_DIR = ROOT / "assets" / "papers"
MINERU_OUT = ROOT / "assets/artifacts/out_mineru"   # MinerU 解析产物根（env 开启时优先）

MAX_CHUNK_LEN = 4000   # 与 pipeline / run_qa_eval 的二级切分阈值一致

# MinerU 的 `table_caption` 里"真正表标题"的锚点（见 `_clean_table_captions`）。
# ⚠️ 必须同时接受**罗马数字**表号（`TABLE IV`）与带后缀的阿拉伯号（`Table 5a`）：
#    首版写成 `Table\s+[IVXLC]*\d+` 只认"罗马前缀 + 数字"，于是 `TABLE IV COMPARISON`
#    不匹配 → **整段真 caption 被删掉**（自测抓到，语料里确有罗马数字表的论文）。
_TBL_CAP_RE = re.compile(r"Table\s+(?:[IVXLC]+|\d+(?:\.\d+)*[A-Za-z]?)", re.I)


def _clean_table_captions(el: dict) -> list[str]:
    """只保留 `table_caption` 里**真正的表标题**片段。

    MinerU 的 `table_caption` 会混进两类噪声（实测，`qa/recall/MINERU_COVERAGE_20260912.md` §5）：
      1. 上方**图的标题**：`['Figure 6: Attention Distributions...', 'Table 4: Average prec...']`
         → 进块污染 BM25/向量，且会让"按编号定位目标表"的评测口径取到**错的号**；
      2. **表头行文字**：`['Age Race Gender', 'Table 3: Annotator agreement...']`。
    两类都能靠"从首个 `Table <编号>` 处截断"清掉；不含 `Table` 的片段整段丢弃。
    """
    out: list[str] = []
    for c in el.get("table_caption") or []:
        s = str(c).strip()
        m = _TBL_CAP_RE.search(s)
        if m:
            out.append(s[m.start():].strip())
    return out


def _table_header_cells(text: str) -> list[str]:
    """块文本的**表头单元格**（首行 md 表格行）；保序去重。

    给无 caption 的表块补身份时用：同一张表被 MinerU 切成多块时各块表头相同 →
    可互相借 caption；借不到就用列名拼一行身份串（不加长、不加新信息）。
    """
    for ln in str(text).splitlines():
        if "|" not in ln or ln.strip().startswith("|--"):
            continue
        cells: list[str] = []
        for c in ln.split("|"):
            c = c.strip()
            if c and c not in cells:
                cells.append(c)
        if len(cells) >= 2:
            return cells
    return []


def _md_rows(text: str) -> list[list[str]]:
    """md 表格的**数据行**（逐行拆单元格；跳过分隔行/非表格行）。"""
    rows: list[list[str]] = []
    for ln in str(text).splitlines():
        s = ln.strip()
        if not s.startswith("|") or set(s) <= set("|-: "):
            continue
        rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def _table_summary(text: str, max_len: int = 600) -> str:
    """表块的**语义摘要**（P2 双写：只喂向量侧，BM25/作答仍用原表）。

    v1（标题 + 列名 + 首列取值）**实测更差**（`qa/recall/P2_DUALWRITE_20260912.md`）：
    同篇表块间平均余弦 0.593→0.616、目标 margin +0.033→+0.018、并集增益 +4→+2。
    机制：**单篇检索靠"同篇内区分度"**，而标题/列名/实体名在同篇的表之间高度重合
    → 向量被平均化、目标失去优势。同源现象：加 150 字通用填充、加 refs 全量都变差。

    v2 因此**保留具体数值**——每个数值在同篇内几乎唯一，正是区分目标表的东西；
    同时保留（压缩的）列名，因为提问会用度量词（accuracy / F1 / AUC）。

    成分：表标题（含 `Table N`） + 列名 + **行标签全集**（实体名，最多 16 个）
    + **前 6 行的数值样例**（`标签: v1 v2 …`）。
    """
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    cap = lines[0] if lines and not lines[0].startswith("|") else ""
    rows = _md_rows(text)
    cols: list[str] = []
    labels: list[str] = []
    samples: list[str] = []
    if rows:
        cols = [c for c in rows[0] if c][:12]
        for r in rows[1:]:
            if not r:
                continue
            label = (r[0] or "").strip()
            vals = [c for c in r[1:] if c][:5]
            if label and not re.fullmatch(r"[\d.\-–—%]+", label) and label not in labels:
                labels.append(label)
            if vals:
                samples.append(f"{label}: {' '.join(vals)}" if label else " ".join(vals))
            if len(labels) >= 16 and len(samples) >= 6:
                break
    parts = [f"表格：{cap}"] if cap else (["表格"] if cols else [])
    if cols:
        parts.append("列：" + "/".join(cols))
    if labels:
        parts.append("行：" + "、".join(labels[:16]))
    if samples:
        parts.append("行样例：" + "；".join(samples[:6]))
    return "。".join(parts)[:max_len]


def is_qasper(pdf: str) -> bool:
    """QASPER 虚拟论文名：qasper_<paper_id>（不落 PDF，从数据集构造）。"""
    return pdf.startswith("qasper_") and pdf.endswith(".qpdf")


@lru_cache(maxsize=64)
def qasper_chunks(pid: str) -> list[Chunk]:
    """返回 QASPER 某篇论文的 Chunk[]（与 claims 提取共用，chunk_id 恒定）。"""
    papers = load_papers()
    paper = papers.get(pid)
    if not paper:
        raise FileNotFoundError(f"QASPER 缺论文: {pid}")
    return build_chunks(paper)


def _mineru_chunks(pdf: str) -> list[Chunk] | None:
    """env PAPERPILOT_USE_MINERU=1 时优先读 assets/artifacts/out_mineru/<stem>/ 产物转 chunks。

    MinerU content_list（阅读序 + 表格 HTML）→ chunker 同构 Chunk[]
    （tools/mineru_bridge）。产物不存在/损坏 → None 退回 pymupdf 路径。
    References 已在桥内丢弃；PREAMBLE 用 extractable 过滤，与 pymupdf 路径一致。
    """
    base = MINERU_OUT / Path(pdf).stem
    if not base.is_dir():
        return None
    from paperpilot.tools.mineru_bridge import chunks_from_mineru_dir
    try:
        mc = chunks_from_mineru_dir(base)
    except Exception:  # noqa: BLE001  产物损坏 → 退回
        return None
    if mc is None:
        return None
    return analyzer.extractable(mc)


@lru_cache(maxsize=16)
def ordered_chunks(pdf: str) -> list[Chunk]:
    """返回该 pdf 的 extractable chunks（按正文顺序，已去 PREAMBLE/References/空）。"""
    if is_qasper(pdf):
        # QASPER：full_text → chunks，无 PDF 版面概念（page=0）
        raw = qasper_chunks(pdf.removeprefix("qasper_").removesuffix(".qpdf"))
        return [c for c in raw if c.title_path != ["(PREAMBLE)"]]
    if os.environ.get("PAPERPILOT_USE_MINERU") == "1":
        mc = _mineru_chunks(pdf)
        if mc is not None:
            return mc
    result = parse_pdf(str(PAPERS_DIR / pdf))
    chunks = chunk_document(result["blocks"], max_len=MAX_CHUNK_LEN)
    return analyzer.extractable(chunks)


@lru_cache(maxsize=64)
def _mineru_extra_by_page(stem: str) -> dict[int, list[str]]:
    """MinerU content_list 的 table/equation 元素 → {页码(1基): [元素文本]}。

    只取 pymupdf 拿不到（或拿得烂）的两类：table（结构化表值，pymupdf 会撕成碎片）
    与 equation（LaTeX）。**不取 MinerU 的正文 text**——pymupdf 已有，重复注入只会膨胀。
    """
    base = MINERU_OUT / stem
    if not base.is_dir():
        return {}
    from paperpilot.tools.mineru_bridge import _find_content_list, element_text
    try:
        cl = _find_content_list(base)
    except Exception:  # noqa: BLE001 产物损坏 → 不增强
        return {}
    if not cl:
        return {}
    out: dict[int, list[str]] = {}
    for el in cl:
        if el.get("type") not in ("table", "equation"):
            continue
        t = element_text(el)
        if not t:
            continue
        page = int(el.get("page_idx", 0)) + 1
        out.setdefault(page, []).append(t)
    return out


@lru_cache(maxsize=32)
def retrieval_chunks(pdf: str) -> list[Chunk]:
    """**检索视图**：chunk_id/页码与 ordered_chunks 完全一致，仅文本更厚。

    背景（2026-09-10，方案 B3）：报告链无论如何需要 pymupdf（页码/版面/claims），
    但 MinerU 对**检索**更好（表格结构化、公式 LaTeX、图内文字不污染正文）。
    两者不是二选一：
      - chunk 空间只保留 pymupdf 一套 → cites / 前端高亮 / claims 锚点全部不动；
      - 表格/公式文本按页注入检索视图 → 表值能被召回、能被作答读到；
      - claims 抽取读的是 ordered_chunks → **表格只进检索不进 claims**（阶段 O 决策）。

    注入规则：MinerU 第 N 页（page_idx=N-1）的表格/公式，挂到**该页在文档序中
    第一个 chunk** 的末尾；同一页只挂一次（避免跨页 chunk 重复注入）。

    降级：QASPER / 无 MinerU 产物 / MinerU chunks 已成主源（env 遗留路径）→ 原样返回。
    """
    base = ordered_chunks(pdf)
    if is_qasper(pdf):
        return _with_external_tables(pdf, base)
    if not base:
        return base
    if os.environ.get("PAPERPILOT_USE_MINERU") == "1" and _mineru_chunks(pdf) is not None:
        return base  # 遗留全链 MinerU 模式：base 已是 MinerU chunks，不再二次注入
    if os.environ.get("PAPERPILOT_MINERU_INJECT", "1") != "1":
        return base  # 评测用开关：关掉表格/公式注入（默认开 → 线上行为不变）
    extra = _mineru_extra_by_page(Path(pdf).stem)
    if not extra:
        return base
    used: set[int] = set()
    out: list[Chunk] = []
    for c in base:
        ps, pe = int(c.page_span[0]), int(c.page_span[1])
        add: list[str] = []
        for p in range(ps, pe + 1):
            if p in extra and p not in used:
                used.add(p)
                add.extend(extra[p])
        if add:
            c = c.model_copy(update={"text": c.text + "\n\n" + "\n".join(add)})
        out.append(c)
    return out


@lru_cache(maxsize=32)
def _external_table_chunks(pdf: str) -> tuple[Chunk, ...]:
    """QASPER **外部表格通道**（实验）：从 arXiv 原始 PDF 的 MinerU 产物补表格/公式文本。

    为什么需要：QASPER 的 `figures_and_tables` 只提供 PNG + caption 的字符 span，
    **表格数值没有任何文本形式** → 凡 gold 出自表格的题，在纯 QASPER 文本通道下**结构上不可达**
    （实测 500 题里 36 道，占 8.0pt）。

    开启条件：`PAPERPILOT_QASPER_TABLES=1` **且** `assets/artifacts/out_mineru/<pid>v1/` 有 content_list。
    默认**关**（线上行为不变）。产出为**追加在正文块之后**的独立块（chunk_id `xtbl-*`），
    只进检索视图，不动 ordered_chunks（报告/claims 锚点不受影响）。
    """
    if os.environ.get("PAPERPILOT_QASPER_TABLES") != "1":
        return ()
    pid = pdf.removeprefix("qasper_").removesuffix(".qpdf")
    base = MINERU_OUT / f"{pid}v1"
    if not base.is_dir():
        return ()
    from paperpilot.tools.mineru_bridge import (EXT_CHUNK_PREFIX, _find_content_list,
                                                element_text)
    try:
        cl = _find_content_list(base)
    except Exception:  # noqa: BLE001
        return ()
    if not cl:
        return ()
    # 两遍扫：先（清 caption + 取表头指纹），再组装（缺 caption 的借/补）
    items: list[tuple[int, dict, str, list[str], list[str]]] = []
    for i, el in enumerate(cl):
        typ = el.get("type")
        if typ not in ("table", "equation"):
            continue
        caps: list[str] = []
        if typ == "table":
            caps = _clean_table_captions(el)
            # 只在确有噪声被清掉时才复制 el（省一次 dict 拷贝）
            if caps != [str(x).strip() for x in (el.get("table_caption") or []) if str(x).strip()]:
                el = {**el, "table_caption": caps}
        # ⚠️ 不要再拼 caption：`element_text` 对 table 已返回 `cap + body + footnote`
        # （旧代码又拼了一次 `table_caption` → caption 在块内出现 2 次，浪费 ~280 字、
        #  且 caption 词在 BM25 里权重翻倍）。实测去重后目标表进表池 top-1 从 50% → 65%，
        #  位次净改善 4 题（`qa/recall/EXT_REPR_20260911.md`）。
        t = element_text(el)
        if not t:
            continue
        items.append((i, el, t, caps, _table_header_cells(t) if typ == "table" else []))

    # 同篇内"表头相同"的块可互相借 caption：MinerU 会把**一张表切成多块**，
    # 只有其中一块带 caption（实例 `2002.10361`: xtbl-58/59 ← xtbl-60 的 "Table 5: ..."）。
    donor: dict[str, str] = {}
    for _i, _el, _t, caps, cells in items:
        if caps and cells:
            donor.setdefault("|".join(sorted(cells)), " ".join(caps))

    out: list[Chunk] = []
    # P2 双写：`PAPERPILOT_TABLE_EMBED_SUMMARY=1` 时，表块额外给向量侧一行语义摘要
    #（BM25/作答仍读 `text`）。**默认关** —— 它是待验证的实验项，不改变线上行为。
    want_emb = os.environ.get("PAPERPILOT_TABLE_EMBED_SUMMARY") == "1"
    for i, el, t, caps, cells in items:
        if el.get("type") == "table" and not caps:
            # 补身份（**只补缺，不加长**）：借不到就用列名拼一行身份串，
            # 让 BM25 至少能命中 "method / accuracy" 这类词，向量也有语义锚点。
            # 不要给**已有 caption 的好块**加料 —— 实测 refs 全量会让向量净 7 题变差
            # （`qa/recall/EXT_REPR_20260911.md`），情形不同、别学。
            borrowed = donor.get("|".join(sorted(cells))) if cells else None
            if borrowed:
                t = f"{borrowed}\n{t}"
            elif cells:
                t = "表格（列：" + "、".join(cells[:12]) + "）\n" + t
        emb = (_table_summary(t) or None) if (want_emb and el.get("type") == "table") else None
        out.append(Chunk(chunk_id=f"{EXT_CHUNK_PREFIX}{i}", title_path=["(External Tables)"],
                         page_span=(int(el.get("page_idx", 0)) + 1, int(el.get("page_idx", 0)) + 1),
                         text=t[:MAX_CHUNK_LEN], n_blocks=1, embed_text=emb))
    return tuple(out)


def _with_external_tables(pdf: str, base: list[Chunk]) -> list[Chunk]:
    extra = _external_table_chunks(pdf)
    return list(base) + list(extra) if extra else base


@lru_cache(maxsize=32)
def current_source(pdf: str) -> str:
    """该 pdf 当前解析源：'qasper' | 'mineru' | 'pymupdf'。

    QA/报告链据此决定标题源、figures 源与产物缓存源打标，保证整链同源。
    """
    if is_qasper(pdf):
        return "qasper"
    if os.environ.get("PAPERPILOT_USE_MINERU") == "1" and _mineru_chunks(pdf) is not None:
        return "mineru"
    return "pymupdf"


def top_section(chunk: Chunk | None) -> str:
    """chunk 所属顶层节名：title_path[0] 取 '· ' 后半段。

    "L1 1 · 1 Introduction"  → "1 Introduction"
    "(PREAMBLE)"              → "(PREAMBLE)"
    取首个 '· ' 之后的**全部**（保留标题内后续 '·'），保证与 claim 端 normalize 一致。
    """
    if chunk is None:
        return ""
    return section_from_path(list(chunk.title_path))


def section_from_path(title_path: list[str]) -> str:
    """title_path → 顶层节名（与 top_section 同一 normalize，供 claim 端复用）。

    claim.title_path 与 chunk.title_path 同构（chunker 产出），
    ClaimHit.home_section 用它在节点层归一，expand_l2 才能按节匹配。
    """
    if not title_path:
        return ""
    head = title_path[0]
    if " · " in head:
        return head.split(" · ", 1)[1].strip()
    return head.strip()


def section_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """按顶层节名分组（保序）。L2 取圆心、判定节边界都用它。"""
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        groups.setdefault(top_section(c), []).append(c)
    return groups


def section_of_chunk_id(chunks: list[Chunk], cid: str) -> str:
    """按 chunk_id 反查所属顶层节（找不到返回空串）。"""
    for c in chunks:
        if c.chunk_id == cid:
            return top_section(c)
    return ""
