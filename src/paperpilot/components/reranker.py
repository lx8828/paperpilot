"""Reranker 门面：重排（空壳预留，多篇再实现）。

现状：无独立重排。RRF 融合（retriever 内）只是弱排序；无交叉编码/LLM 重排。

证据与判断（RAG_COMPONENT_NOTES §2-⑤）：
- 单篇 top12 候选面小、几乎全相关 → 精排理论上限低（朴素 RAG ≈ 我们）。
- QASPER 净亏题根因是"答案块没进 top12"（召回），不是"排序不精"。
- 多篇/大召回面时再实现并与"不重排"做 A/B（Recall@k / 端到端 T 层）。

实现约定：函数签名 rank(query, hits) -> hits（按相关性降序的新列表），以便将来无痛替换。
"""
from __future__ import annotations

from typing import Any


def rank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """当前：原序透传（空壳）。多篇时代换成 LLM/交叉编码重排后透传实现即可。"""
    return list(hits)
