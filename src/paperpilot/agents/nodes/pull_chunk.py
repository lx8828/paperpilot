"""L3 · search_l3（独立全文检索）。

L3 是 v3 两级架构的检索层：L0 总览判不够后触发，独立 ChunkIndex 全文检索
（不继承 L0 证据），结果写 state["l3_chunks"]，供 judge_l3 / generate_answer 使用。

检索侧查询改写（只影响"找"，不改作答口径）。PAPERPILOT_QUERY_REWRITE=0 可关闭做 A/B。

**2026-09-10：v2 的 L2 expand_l2（定向增量扩展：圆心 + 半径 + 预算自环）已下线归档**
（archive/qa_funnel_v2/），本文件只保留 L3。原实现见归档快照。
"""
from __future__ import annotations

import os
from typing import Any

from paperpilot.agents.embedder import ChunkIndex
from paperpilot.agents.state import QAState
from paperpilot.tools import llm

L3_TOP_K = 12   # 离线实测：top8 只覆盖 gold 证据的 ~73%，top12 在 ~85% 且上下文可控；改到 12

# 检索侧查询改写（只影响"找"，不改作答口径）。PAPERPILOT_QUERY_REWRITE=0 可关闭做 A/B。
_REWRITE_SYS = (
    "你是检索查询改写器。给定一个面向学术论文的问题，生成 2~3 条**仅供检索**的查询变体。\n"
    "规则：\n"
    "1. 抽取并保留关键实体（数据集/语料名、模型/方法名、指标、语言、数字等），这是检索的锚；\n"
    "2. 补同义/上位/常见论文表述（如 'how many utterances' → 'corpus size / number of utterances'；\n"
    "   'is the dataset multilingual?' → 'dataset language composition'）；\n"
    "3. 若问题是英文/面向英文论文，变体用英文；口语或长句改短、去虚词；每条 ≤12 词；\n"
    "4. 变体必须与问题同一语义方向，不要自创新问题。\n"
    '只输出 JSON：{"queries": ["...", "...", "..."]}'
)


def _rewrite_queries(question: str) -> list[str]:
    """返回 [原问题, 变体1, 变体2]；改写失败/被禁用时只返回原问题。

    **默认关闭**（2026-09-10 复验，仍负收益）。两次独立 A/B：
      - 2026-09-07（40 题）：ON pass 9 vs OFF 12（救 2 / 坏 5）。
      - 2026-09-10（100 题配对，新默认 nol3j+gate）：ON 68% vs OFF 71%（救 3 / 坏 6），
        calls/题 3.0→3.9（+30%），prompt +2%。见 qa/recall/REWRITE_AB_20260910.md。
    机制：`search_multi_hybrid` 把变体排序与原问题**等权 RRF 融合**，偏题变体会稀释
    原问题的正确排序（与"融合低于单路 oracle"同因）。**不是"改写无用"，是"RRF 平权融合有害"**——
    若多篇场景要用，应改成"原问题排序为主 + 变体只做召回补充（按配额并集）"或
    "变体召回 → 原问题重排"，而非 RRF 平权。
    开启：环境变量 PAPERPILOT_QUERY_REWRITE=1 做对照实验。
    """
    if os.environ.get("PAPERPILOT_QUERY_REWRITE", "0") != "1":
        return [question]
    try:
        obj = llm.chat_json(_REWRITE_SYS, f"问题：{question}", temperature=0.2)
        extra = []
        if isinstance(obj, dict):
            for q in (obj.get("queries") or []):
                s = str(q).strip()
                if s and len(s) <= 200 and s.lower() != question.lower():
                    extra.append(s)
        return ([question] + extra)[:3]
    except Exception:  # noqa: BLE001（改写失败不影响主链路）
        return [question]


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
    """L3 独立全文检索：ChunkIndex 对原始问题检索 topK，写 l3_chunks（干净隔离）。

    2026-09-07 改为多查询混合：查询改写（实体/同义变体）+ 原问题，search_multi_hybrid
    （向量+BM25 × 多查询 RRF）；PAPERPILOT_QUERY_REWRITE=0 可回退单查询 hybrid 做 A/B。
    """
    question = state.get("question") or ""
    idx = ChunkIndex(state.get("pdf") or "")
    queries = _rewrite_queries(question)
    if len(queries) > 1:
        hits = idx.search_multi_hybrid(queries, top_k=L3_TOP_K)
    else:
        hits = idx.search_hybrid(question, top_k=L3_TOP_K)
    l3_chunks: list[dict[str, Any]] = []
    for h in hits:
        l3_chunks.append({
            "chunk_id": h.get("chunk_id", ""),
            "page": h.get("page", 0),
            "section": _section_tail(list(h.get("title_path") or [])),
            "text": (h.get("text") or ""),  # 不截断：分块期已控制单块长度
        })
    debug = dict(state.get("debug") or {})
    debug["l3"] = {"topk": len(l3_chunks)}
    route = list(state.get("route") or [])
    if "L3" not in route:
        route.append("L3")
    return {
        "l3_chunks": l3_chunks,
        "route": route,
        "debug": debug,
    }
