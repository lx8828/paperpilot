"""Retriever 门面：正文/主张的（混合）检索。

现状实现（embedder.py）：
  ChunkIndex  = 正文 chunk 向量索引；search_hybrid = 向量+BM25 RRF 融合（专名/术语互补），
                search_multi_hybrid = 多查询 RRF；向量缓存 assets/artifacts/out_views/*.cvec（秒级复用）。
  ClaimIndex  = claims 主张向量索引（v3 未单列一层，多篇/主题时作检索源复用）。
top12 覆盖 gold 证据 ~85%（top8 只 ~73%）。

证据与默认配置（RAG_COMPONENT_NOTES §2-④）：
- 朴素 RAG(B1) ≈ 我们(B2)：混合检索件不需要花哨。
- 真正的检索短板是数值/表格块（L3 捞不到表值）→ 靠 MinerU/输入质量，不是加检索复杂度。
- 默认：hybrid + top_k=12；改检索先跑框架 C 层 Recall@k（多篇再建真值集）。
"""
from __future__ import annotations

from typing import Any


class ChunkRetriever:
    """单篇正文检索器（薄包装 ChunkIndex）。"""

    def __init__(self, pdf: str):
        from paperpilot.agents.embedder import ChunkIndex
        self._idx = ChunkIndex(pdf)

    def search(self, question: str, top_k: int = 12, hybrid: bool = True) -> list[dict[str, Any]]:
        """正文检索；hybrid=True → 向量+BM25 RRF（默认），False → 纯向量 top-k。"""
        return self._idx.search_hybrid(question, top_k=top_k) if hybrid \
            else self._idx.search(question, top_k=top_k)

    def search_multi(self, queries: list[str], top_k: int = 12) -> list[dict[str, Any]]:
        """多查询 × 向量+BM25 RRF 融合（QueryRewriter 开启时用）。"""
        return self._idx.search_multi_hybrid(queries, top_k=top_k)


class ClaimRetriever:
    """主张检索器（薄包装 ClaimIndex；v3 主路径未用，多篇/主题复用预留）。"""

    def __init__(self, pdf: str):
        from paperpilot.agents.embedder import ClaimIndex
        self._idx = ClaimIndex(pdf)

    def search(self, question: str, top_k: int = 12) -> list[dict[str, Any]]:
        return self._idx.search(question, top_k=top_k)


def search_chunks(pdf: str, question: str, top_k: int = 12, hybrid: bool = True) -> list[dict[str, Any]]:
    """一次性入口（无状态）——按论文文件名字符串检索正文。"""
    return ChunkRetriever(pdf).search(question, top_k=top_k, hybrid=hybrid)
