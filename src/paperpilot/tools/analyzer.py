"""Analyzer：从 Chunk 提取四类 Claims（contribution / method / result / limitation）。

最小闭环：chunks → LLM 结构化提取 → Claim[]（携带 chunk 溯源锚点）。
暂不引入 LangGraph，先把提取质量验证通过，再考虑包成 Analyzer Agent。

每个正文 chunk 一次 LLM 调用；失败（网络/解析）不中断整篇，逐块容错并汇总。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from paperpilot.models.schema import Chunk, Claim, ClaimType
from paperpilot.prompts.analyzer import SYSTEM_PROMPT, build_user_prompt
from paperpilot.tools import llm

CLAIM_TYPES: set[str] = set(ClaimType.__args__)  # type: ignore[attr-defined]
SKIP_MARKERS = ("References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY")

# 表格数据行保护：PDF 表格被解析为若干"一行一个值"的短行（如 "58.2%"、"0.11%"），
# 若喂给提取模型会被拼成伪句子。检测纯数值/百分比/序号类表格单元，
# 提取前替换为占位符，模型即不会据此生成 claims。
_TABLE_CELL_RE = re.compile(r"[0-9.,%()+\-·\s]+$")


def _is_table_cell(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # 允许含货币符的短数值行（如 £600）；币符后跟数字
    if s[0] in "£$€¥" and re.fullmatch(r"[£$€¥0-9.,%()+\-·\s]+", s):
        return True
    return bool(_TABLE_CELL_RE.fullmatch(s))


def mask_table_rows(text: str) -> str:
    """把疑似表格数据行替换为 [TABLE_CELL]，供提取时忽略。"""
    return "\n".join("[TABLE_CELL]" if _is_table_cell(l) else l
                     for l in text.splitlines())


def _skip_chunk(chunk: Chunk) -> bool:
    """PREAMBLE / References 不参与提取；空正文不提取。"""
    if not chunk.text.strip():
        return True
    if chunk.title_path == ["(PREAMBLE)"]:
        return True
    head = chunk.title_path[0].split("·")[-1].strip() if chunk.title_path else ""
    return head in SKIP_MARKERS


def extractable(chunks: list[Chunk]) -> list[Chunk]:
    """返回适合提取 claims 的正文 chunk（过滤 PREAMBLE/References/空正文）。"""
    return [c for c in chunks if not _skip_chunk(c)]


def _build_claim(chunk: Chunk, row: dict, seq: int) -> Claim | None:
    t = (row.get("type") or "").strip()
    text = (row.get("text") or "").strip()
    ev = (row.get("evidence_quote") or "").strip()
    if t not in CLAIM_TYPES or not text:
        return None
    return Claim(
        claim_id=f"c{seq:04d}",
        type=t,
        text=text,
        evidence_quote=ev,
        chunk_id=chunk.chunk_id,
        title_path=list(chunk.title_path),
        page=chunk.page_span[0],
        page_span=tuple(chunk.page_span),
    )


def _extract_chunk_rows(chunk: Chunk) -> tuple[list[dict], str | None]:
    """对单个 chunk 调 LLM，返回 (rows, error)。错误不抛出，逐块容错。"""
    prompt = build_user_prompt(chunk.title_path, mask_table_rows(chunk.text))
    try:
        rows = llm.chat_json(SYSTEM_PROMPT, prompt)
    except llm.LLMError as e:
        return [], f"{chunk.chunk_id} [{chunk.title_path[-1]}]: {e}"
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return [], f"{chunk.chunk_id} [{chunk.title_path[-1]}]: 非数组响应"
    return rows, None


def extract_claims(chunks: list[Chunk], *, workers: int = 4) -> tuple[list[Claim], list[str]]:
    """逐 chunk 提取 claims（支持并发，chunk 间相互独立）。

    Args:
        chunks: Chunk 列表（会跳过 PREAMBLE/References/空正文）
        workers: 并发请求数；1=串行

    Returns:
        (claims, errors)：
          claims: 全部成功提取的 Claim（按 chunk 序，claim_id 递增）。
          errors: 失败的 chunk 说明列表。
    """
    target = [c for c in chunks if not _skip_chunk(c)]
    if workers <= 1 or len(target) <= 1:
        results = [_extract_chunk_rows(c) for c in target]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_extract_chunk_rows, target))

    claims: list[Claim] = []
    errors: list[str] = []
    seq = 1
    for chunk, (rows, err) in zip(target, results):
        if err is not None:
            errors.append(err)
            continue
        for row in rows:
            c = _build_claim(chunk, row, seq)
            if c is not None:
                claims.append(c)
                seq += 1
    return claims, errors
