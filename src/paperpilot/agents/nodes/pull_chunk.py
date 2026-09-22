"""L3 · search_l3（独立全文检索）。

L3 是 v3 两级架构的检索层：L0 总览判不够后触发，独立 ChunkIndex 全文检索
（不继承 L0 证据），结果写 state["l3_chunks"]，供 judge_l3 / generate_answer 使用。

**查询侧优化已抽成独立组件** `components/query_optimizer.py`
（五级：L1 意图识别 / L2 问题重述 / L3 改写扩展 / L4 HyDE / L5 查询分解，
目前仅 L3 实现）。本节点只负责"取查询集 → 检索"，
默认全关（只回原问题）→ 与既有 A/B 基线行为完全一致。

**2026-09-10：v2 的 L2 expand_l2（定向增量扩展：圆心 + 半径 + 预算自环）已下线归档**
（archive/qa_funnel_v2/），本文件只保留 L3。原实现见归档快照。
"""
from __future__ import annotations

from typing import Any

from paperpilot.agents.embedder import MultiChunkIndex
from paperpilot.agents.state import QAState
from paperpilot.components import query_optimizer

L3_TOP_K = 12   # 离线实测：top8 只覆盖 gold 证据的 ~73%，top12 在 ~85% 且上下文可控；改到 12

def _section_tail(paths: list[str]) -> str:
    for p in reversed(paths):
        if " · " in p:
            return p.split(" · ", 1)[1].strip()
    return paths[-1] if paths else ""


def _to_chunk_texts(chunks: list[Any]) -> list[dict[str, Any]]:
    """chunk → 可序列化文本切片。注意：**不再截断**。

    chunk 在分块期已按段落原子切到 ~4000 字符（document_cache.MAX_CHUNK_LEN），
    QA 层再截断就是纯丢信息（答案可能落在被砍掉的尾部）。
    """
    out = []
    for c in chunks:
        out.append({
            "chunk_id": c.chunk_id,
            "page": c.page_span[0],
            "section": _section_tail(list(c.title_path)),
            "text": c.text,
        })
    return out


def search_l3(state: QAState) -> dict[str, Any]:
    """L3 独立全文检索：ChunkIndex 对（优化后的）查询集检索 topK，写 l3_chunks。

    查询集来自 `components/query_optimizer.optimize()`：
      · 默认全关 → 只有原问题 → 走**多篇主路径** `search_layered`（篇内检索 + 跨篇 quota 融合）；
      · 开启任一级（如 `PAPERPILOT_QUERY_LEVELS=l3`）→ 多查询 `search_multi_hybrid`
        （向量+BM25 × 多查询 RRF）。**注意**：多查询的等权 RRF 已被两次 A/B 证伪
        （见 query_optimizer 模块 docstring），此路径仅作对照，新融合策略应在
        检索侧按 `query_optimizer.quota_union` 之类实现后再做 A/B。
    """
    question = state.get("question") or ""
    # ⚠️ 2026-09-23：**语料只有多篇**（本系统没有单篇路径）→ 一律走 MultiChunkIndex。
    idx = MultiChunkIndex([str(p) for p in (state.get("pdfs") or []) if p])
    plan = query_optimizer.optimize(question)
    queries = plan.queries
    if plan.multi:
        hits = idx.search_multi_hybrid(queries, top_k=L3_TOP_K)
        path = "multi_hybrid"
    else:
        # **多篇主路径（2026-09-23 接线）**：每篇内先检索（同粒度可比）→ 跨篇 quota 融合。
        # 为什么不用全局 `search_hybrid`：全局 cosine 实际在比"谁切得细"
        # （5 篇 chunk 数 22~52、中位块长差 5.8×）→ 泛化查询 top_k 被单篇霸占，
        # 且加大 k 无效。实测对比见 `MultiChunkIndex.search_layered` 的 docstring。
        hits = idx.search_layered(question, top_k=L3_TOP_K)
        path = "layered_quota"
    l3_chunks: list[dict[str, Any]] = []
    for h in hits:
        l3_chunks.append({
            "chunk_id": h.get("chunk_id", ""),
            "page": h.get("page", 0),
            "section": _section_tail(list(h.get("title_path") or [])),
            "text": (h.get("text") or ""),  # 不截断：分块期已控制单块长度
            # ⚠️ 多篇语料：每条带**它属于哪一篇** —— cites 才能定位到具体论文（2026-09-23）
            "pdf": h.get("pdf", ""),
        })
    debug = dict(state.get("debug") or {})
    dbg_l3: dict[str, Any] = {"topk": len(l3_chunks), "path": path,
                              "n_papers_hit": len({h.get("pdf") for h in l3_chunks})}
    if plan.levels:
        dbg_l3["query_plan"] = plan.summary()     # 只在真的开了级别时记，避免污染基线
    debug["l3"] = dbg_l3
    route = list(state.get("route") or [])
    if "L3" not in route:
        route.append("L3")
    return {
        "l3_chunks": l3_chunks,
        "route": route,
        "debug": debug,
    }
