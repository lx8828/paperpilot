"""按标题切分 + 二级切分（Chunker）。

将解析出的文本块组装成语义 chunk：
  1. 标题识别（heading.find_headings）→ 维护标题层级栈，每个 chunk 带**完整根路径**
     title_path（如 ["3 Method", "3.2 RLOO"]），即使父标题无独立正文也不丢上下文
  2. 超大 chunk（超过 max_len）做**段落级二级切分**：段落累加、到达上限即切，
     子块继承父 title_path 并以 part（"1/3"）编号，保证仍可溯源
  3. 每个 chunk 记录组成 block 的锚点（spans），支撑 Verifier 精确回溯

References / PREAMBLE 不参与二级切分。
"""
from __future__ import annotations

from typing import TypedDict

from paperpilot.models.schema import Chunk, ChunkSpan
from paperpilot.tools.heading import BlockDict, DetailDict, find_headings
from paperpilot.tools.paragraph import Paragraph, split_paragraphs

# 二级切分阈值（字符，默认 4000）
DEFAULT_MAX_LEN = 4000

REF_MARKERS = ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY")


class _RawChunk(TypedDict):
    """组装阶段的内部 chunk。"""
    chunk_id: str
    title_path: list[str]
    blocks: list[BlockDict]
    has_body: bool


def _heading_label(h: DetailDict) -> str:
    """标题路径标签，如 'L2 3.1 · 3.1 Construction ...'。"""
    no = h.get("no") or ""
    txt = (h.get("text") or "").replace("\n", " ").strip()
    return f"{h.get('kind')} {no} · {txt}" if no else f"{h.get('kind')} · {txt}"


def _block_sort_key(b: BlockDict):
    return (b["page"], b["y0"], b["x0"])


def _span(b: BlockDict) -> ChunkSpan:
    return ChunkSpan(block_id=b["block_id"], page=b["page"], text=b["text"])


def _is_ref_chunk(title_path: list[str]) -> bool:
    if not title_path:
        return False
    head = title_path[0].split("·")[-1].strip()
    return head in REF_MARKERS


def _text_of_blocks(blocks: list[BlockDict]) -> str:
    return "\n".join(b["text"] for b in blocks).strip()


def _assemble(ordered: list[BlockDict], headings: list[DetailDict]) -> list[_RawChunk]:
    """按标题组装基础 chunk（标题栈维护完整 title_path）。返回内部 chunk dict。"""
    hid = {h["block_id"]: h for h in headings}

    stack: list[tuple[int | None, str]] = []  # (level, label)
    chunks: list[_RawChunk] = []
    cur: _RawChunk | None = None
    chunk_no = 0

    for b in ordered:
        h = hid.get(b["block_id"])
        if h is not None:
            label = _heading_label(h)
            level = h.get("level")
            if level is None:
                # SPECIAL/APPENDIX：独立标题段（清空之前栈）
                stack = [(None, label)]
            else:
                # 数字编号标题：弹出栈中所有 SPECIAL/附录（None 级）及更深的数字级
                while stack and (stack[-1][0] is None or stack[-1][0] >= level):
                    stack.pop()
                stack.append((level, label))
            path = [lab for _, lab in stack]

            if cur is not None and not cur["has_body"]:
                # 上一标题无正文（L1 直接跟 L2 等）→ 并入该子章节路径
                cur["title_path"] = path
                cur["blocks"].append(b)
            else:
                chunk_no += 1
                cur = {
                    "chunk_id": f"c{chunk_no}",
                    "title_path": path,
                    "blocks": [b],
                    "has_body": False,
                }
                chunks.append(cur)
        else:
            if cur is None:
                chunk_no += 1
                cur = {
                    "chunk_id": f"c{chunk_no}",
                    "title_path": ["(PREAMBLE)"],
                    "blocks": [],
                    "has_body": True,
                }
                chunks.append(cur)
            cur["blocks"].append(b)
            cur["has_body"] = True
    return chunks


def _split_by_paragraph(blocks: list[BlockDict], max_len: int) -> list[list[Paragraph]]:
    """段落累加切分：段落作为原子单元，累计超过 max_len 即切。"""
    paras = split_paragraphs(blocks)
    parts: list[list[Paragraph]] = []
    cur: list[Paragraph] = []
    cur_len = 0
    for p in paras:
        pl = len(p["text"]) + 1  # +换行
        if cur and cur_len + pl > max_len:
            parts.append(cur)
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += pl
    if cur:
        parts.append(cur)
    # 单段就超 max_len 的极端情况：每段单独成块（不硬拆句子）
    if not parts:
        parts = [[p] for p in paras]
    return parts


def chunk_document(blocks: list[BlockDict], max_len: int = DEFAULT_MAX_LEN) -> list[Chunk]:
    """按标题（+二级切分）把文本块切成 Chunk 列表。

    Args:
        blocks: parse_pdf 输出的文本块（含 block_id/page/x0..y1/text/lines）
        max_len: 二级切分字符阈值（正文 chunk 超过即段落级切分）

    Returns:
        Chunk 列表（按正文页序）。子块带 part（"1/3"），共享父 title_path。
    """
    ordered = sorted(blocks, key=_block_sort_key)
    headings = find_headings(ordered)
    raw_chunks = _assemble(ordered, headings)

    result: list[Chunk] = []
    for c in raw_chunks:
        c_blocks: list[BlockDict] = c["blocks"]
        text = _text_of_blocks(c_blocks)
        pages = sorted({int(b["page"]) for b in c_blocks})
        page_span: tuple[int, int] = (pages[0], pages[-1]) if pages else (0, 0)
        is_special = _is_ref_chunk(c["title_path"]) or c["title_path"] == ["(PREAMBLE)"]

        if len(text) <= max_len or is_special or not c_blocks:
            # 无需二级切分：整块输出
            result.append(Chunk(
                chunk_id=c["chunk_id"],
                title_path=c["title_path"],
                page_span=page_span,
                text=text,
                n_blocks=len(c_blocks),
                spans=[_span(b) for b in c_blocks],
            ))
            continue

        # 二级切分
        parts = _split_by_paragraph(c_blocks, max_len)
        total = len(parts)
        block_map = {b["block_id"]: b for b in c_blocks}
        for i, paras in enumerate(parts, 1):
            texts = [p["text"] for p in paras if p["text"].strip()]
            if not texts:
                continue
            sub_text = "\n\n".join(texts)
            p_pages = sorted({int(p["page"]) for p in paras})
            span_pages: tuple[int, int] = (p_pages[0], p_pages[-1]) if p_pages else page_span
            # 子块的 block 锚点：按段落行的来源 block 顺序
            seen: list[ChunkSpan] = []
            seen_ids: set[str] = set()
            for p in paras:
                for row in p["rows"]:
                    bid = row["block_id"]
                    if bid and bid not in seen_ids:
                        seen_ids.add(bid)
                        src = block_map.get(bid)
                        if src is not None:
                            seen.append(_span(src))
            result.append(Chunk(
                chunk_id=f"{c['chunk_id']}-p{i}",
                title_path=c["title_path"],
                page_span=span_pages,
                text=sub_text,
                n_blocks=len(seen),
                part=f"{i}/{total}",
                spans=seen,
            ))
    return result
