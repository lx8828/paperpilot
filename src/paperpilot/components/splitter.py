"""DocumentSplitter 门面：论文预处理 / 结构切分 / 主张抽取。

现状实现（工具层）：pdf_parser.parse_pdf（版面块）→ chunker.chunk_document（标题树切块，
段落原子 ~4000）→ analyzer.extractable（去前言/参考文献/空块）；QASPER 走 qasper_source.build_chunks。
独特资产：主张抽取层 analyzer.extract_claims（LLM 逐块抽 claim + evidence）——帮读/溯源差异来源，
不要退回纯分块。表格/扫描/复杂版面 → MinerU（解析层对照实验，见 RAG_COMPONENT_NOTES §2-①）。
"""
from __future__ import annotations

from typing import Any


def split_pdf(pdf_path: str) -> dict[str, Any]:
    """解析 PDF 为版面块（含标题层级）。返回 parse_pdf 的 blocks dict。"""
    from paperpilot.tools.pdf_parser import parse_pdf
    return parse_pdf(pdf_path)


def chunk_blocks(blocks: dict[str, Any], max_len: int | None = None) -> list[Any]:
    """按标题树把版面块切为 Chunk[]（段落原子，max_len 缺省用 chunker 默认）。"""
    from paperpilot.tools.chunker import DEFAULT_MAX_LEN, chunk_document
    blk = blocks.get("blocks", blocks)
    return chunk_document(blk, max_len=max_len or DEFAULT_MAX_LEN)


def extractable(chunks: list[Any]) -> list[Any]:
    """过滤不可用块（PREAMBLE/References/空），产出可检索/抽取的正文 Chunk[]。"""
    from paperpilot.tools.analyzer import extractable as _extractable
    return _extractable(chunks)


def ordered(pdf: str) -> list[Any]:
    """统一入口：给定论文文件名（pdf / qasper_<id>.qpdf）返回 extractable Chunk[]。"""
    from paperpilot.agents.document_cache import ordered_chunks
    return ordered_chunks(pdf)


def chunks_from_qasper(paper: dict[str, Any]) -> list[Any]:
    """QASPER 虚拟论文 → Chunk[]（full_text 按标题切，page=0）。"""
    from paperpilot.qasper_source import build_chunks
    return build_chunks(paper)


def extract_claims(chunks: list[Any], *, workers: int = 4):
    """主张抽取（独特资产）：逐 chunk LLM 抽 claim + evidence。返回 (claims, skipped)。"""
    from paperpilot.tools.analyzer import extract_claims as _extract_claims
    return _extract_claims(chunks, workers=workers)
