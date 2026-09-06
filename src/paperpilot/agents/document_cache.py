"""QA 层的正文 chunk 文档缓存（进程级，跨节点复用 parse）。

为什么独立成模块：L2 增量扩展、L3 全文检索、ChunkIndex 向量化都要用到
"这篇论文切出的 extractable chunks"——之前 pull_chunk 各自 parse，浪费。
这里统一 lru_cache：同一 pdf 首次 parse+chunk，之后所有节点复用。

约定（与 chunker/analyzer 一致）：
    ordered_chunks(pdf)    → 按正文顺序的 extractable chunks（已过滤 PREAMBLE/References/空）
    top_section(chunk)     → chunk 所属顶层节名（title_path[0] 的 "· " 后半段）
    顶层节名格式与 report.groups 的 sections 字段 / ClaimHit.home_section 统一，
    expand_l2 按它做 section 匹配。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from paperpilot.models.schema import Chunk
from paperpilot.tools import analyzer
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.pdf_parser import parse_pdf
from paperpilot.qasper_source import load_papers, build_chunks

# document_cache.py → agents/ → paperpilot/ → src/ → 根
ROOT = Path(__file__).resolve().parents[3]
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"

MAX_CHUNK_LEN = 4000   # 与 pipeline / run_qa_eval 的二级切分阈值一致


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


@lru_cache(maxsize=16)
def ordered_chunks(pdf: str) -> list[Chunk]:
    """返回该 pdf 的 extractable chunks（按正文顺序，已去 PREAMBLE/References/空）。"""
    if is_qasper(pdf):
        # QASPER：full_text → chunks，无 PDF 版面概念（page=0）
        raw = qasper_chunks(pdf.removeprefix("qasper_").removesuffix(".qpdf"))
        return [c for c in raw if c.title_path != ["(PREAMBLE)"]]
    result = parse_pdf(str(PAPERS_DIR / pdf))
    chunks = chunk_document(result["blocks"], max_len=MAX_CHUNK_LEN)
    return analyzer.extractable(chunks)


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
