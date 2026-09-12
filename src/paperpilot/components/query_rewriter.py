"""QueryRewriter 门面：生成检索查询变体（默认关闭）。

现状实现：pull_chunk._rewrite_queries（抽取关键实体为锚 + 同义/上位变体 2-3 条；
retriever 侧多查询 × 向量+BM25 RRF 融合）。开关：PAPERPILOT_QUERY_REWRITE=1。

证据与默认配置（RAG_COMPONENT_NOTES §2-③）：
- 2026-09-07 A/B（40 题同裁判）：改写 ON pass 9 vs OFF pass 12 —— **单篇负收益**
  （救回 2 弄坏 5），故默认关闭。
- 多篇/语料库召回面大、单查询带偏风险高时再开，届时必须重新 A/B。
"""
from __future__ import annotations

import os


def enabled() -> bool:
    """当前是否开启（读 env）。默认关。"""
    return os.environ.get("PAPERPILOT_QUERY_REWRITE", "0") == "1"


def rewrite(question: str) -> list[str]:
    """返回检索查询列表：[原问题, 变体1, ...]（≤3）。未开启/失败 → [question]。"""
    from paperpilot.agents.nodes.pull_chunk import _rewrite_queries
    return _rewrite_queries(question)


def queries_for(question: str) -> list[str]:
    """Retriever 侧的最终查询集（尊重 enabled()，未开启只回原问题）。"""
    return rewrite(question) if enabled() else [question]
