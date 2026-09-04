"""节点① retrieve_claims（L1）：claim embedding 检索 top12。

产出（写入 State）：
    retrieved          topK 主张（ClaimHit，含直达 evidence/chunk_id/home_section）
    debug             n_retrieved

L0 report_l0 已产 overview/title/core_points；本层只负责"主张级"检索，
供 Judge1 判"主题级问题是否够"；不够时 Judge1 输出 target_sections
（从命中 claims 的 home_section 候选里选）→ L2 定向圆心。
"""
from __future__ import annotations

from typing import Any

from paperpilot.agents.document_cache import section_from_path
from paperpilot.agents.embedder import ClaimIndex
from paperpilot.agents.state import QAState

TOP_K = 12


def retrieve_claims(state: QAState) -> dict[str, Any]:
    pdf = state.get("pdf", "")
    question = state.get("question", "")

    idx = ClaimIndex(pdf)
    report = idx.report
    hits = idx.search(question, top_k=TOP_K)

    claims_map = {c["claim_id"]: c for c in report.get("claims", [])}
    retrieved: list[dict[str, Any]] = []
    for h in hits:
        rep_id = h.get("rep_claim_id") or (h.get("claim_ids") or [""])[0]
        rep = claims_map.get(rep_id, {})
        pages = h.get("pages") or []
        retrieved.append({
            "gid": h.get("group_id", ""),
            "rep_claim_id": rep_id,
            "label": h.get("label", ""),
            "importance": h.get("importance", 0),
            "text": h.get("rep_text", ""),
            "sections": h.get("sections", []),
            "pages": pages,
            "evidence": rep.get("evidence_quote", ""),
            "page": rep.get("page") or (pages[0] if pages else 0),
            "chunk_id": rep.get("chunk_id", ""),
            "home_section": section_from_path(list(rep.get("title_path") or [])),
            "score": h.get("score", 0.0),
        })

    route = list(state.get("route") or [])
    if "L1" not in route:
        route.append("L1")
    debug = dict(state.get("debug") or {})
    debug["l1"] = {"n_retrieved": len(retrieved)}
    return {
        "retrieved": retrieved,
        "route": route,
        "debug": debug,
    }
