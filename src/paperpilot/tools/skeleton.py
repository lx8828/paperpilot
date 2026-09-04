"""Skeleton：为必读核心主张（Hub）建立四类论证关系边。

数据流（读已封装的 summary/claims，不重跑旧流程）：
    summary.json groups ─┐
    claims.json evidence ─┴→ 选 Hub(5分 core/primary) → 按关系锁死来源池
        → 每 Hub 一次 LLM 选边 → 后处理校验 → 绑定证据 → skeleton.json

关键约束：
    - 来源类型锁死（implements 只 method_core 且 Hub 为 core_claim；…）
    - ev=miss 的组不进任何源池（无证据者无资格支撑别人）
    - 证据由后端绑定 source 的 evidence_quote，LLM 只选边+给 why
"""
from __future__ import annotations

import re
from typing import Any

from paperpilot.prompts.skeleton import SKELETON_SYSTEM, build_user_prompt
from paperpilot.tools import llm

RELATIONS = ("implements", "supports", "limits", "contrasts")
HUB_LABELS = {"core_claim", "result_primary"}
# 每 Hub 每类关系的数量硬上限（兜底，防止 LLM 连太多）
REL_CAP = {"implements": 4, "supports": 6, "limits": 3, "contrasts": 3}

# contrasts 对 method_core 的"比较语义"粗筛（LLM 仍会再确认）
_COMPARE_RE = re.compile(
    r"不同于|区别于|替代|取代|代替|互补|改进|优于|超过|胜过|胜于|"
    r"对比|相比|相对于|而不|不是.{0,10}而是|"
    r"unlike|alternative|instead|complement|differ|distinct|"
    r"outperform|surpass|compared|versus|vs\.?|relative", re.I)


def _ev_ok(state: str) -> bool:
    return state in ("hit", "loose")


def _group_candidates(groups_by_id: dict[str, dict[str, Any]],
                      label_ok: set[str]) -> list[dict[str, Any]]:
    out = []
    for gid, g in groups_by_id.items():
        if g["label"] in label_ok and _ev_ok(g.get("ev_state", "miss")):
            out.append({"group_id": gid, "label": g["label"], "rep_text": g["rep_text"]})
    return out


def build_pools(groups_by_id: dict[str, dict[str, Any]],
                hub: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """按 Hub 类型构造四个关系的候选源池（白名单 + ev 过滤）。"""
    hub_label = hub["label"]
    pools: dict[str, list[dict[str, Any]]] = {}

    # implements：仅 core_claim Hub 可挂，源只允许 method_core
    pools["implements"] = (_group_candidates(groups_by_id, {"method_core"})
                           if hub_label == "core_claim" else [])

    # supports：core/primary Hub 都可挂，源为其它 result 组（排除 Hub 自身）
    pools["supports"] = [c for c in _group_candidates(
        groups_by_id, {"result_primary", "result_supporting"})
        if c["group_id"] != hub["group_id"]]

    # limits：源只允许 limitation
    pools["limits"] = _group_candidates(groups_by_id, {"limitation"})

    # contrasts：仅 core_claim Hub；源 = related + 自带比较语义的 method_core
    if hub_label == "core_claim":
        cand = _group_candidates(groups_by_id, {"related"})
        for c in _group_candidates(groups_by_id, {"method_core"}):
            if _COMPARE_RE.search(c["rep_text"]):
                cand.append(c)
        pools["contrasts"] = cand
    else:
        pools["contrasts"] = []
    return pools


def _extract_edges(rows: Any, pools: dict[str, list[dict[str, Any]]],
                   allowed: dict[str, set[str]]) -> list[dict[str, Any]]:
    """解析 LLM 输出并校验：relation 合法、source 确实在该关系白名单池。"""
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(rows, list):
        return edges
    for r in rows:
        if not isinstance(r, dict):
            continue
        rel = str(r.get("relation", "")).strip()
        src = str(r.get("source", "")).strip()
        why = str(r.get("why", "")).strip()
        if rel not in RELATIONS or src not in allowed.get(rel, set()):
            continue
        if (rel, src) in seen:
            continue
        seen.add((rel, src))
        edges.append({"relation": rel, "source": src, "why": why})
    # 数量硬上限（保留靠前的，LLM 已按重要度降序）
    capped: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        capped.setdefault(e["relation"], []).append(e)
    out: list[dict[str, Any]] = []
    for rel in RELATIONS:
        out.extend(capped.get(rel, [])[: REL_CAP[rel]])
    return out


def build_skeleton(pdf_name: str, groups: list[dict[str, Any]],
                   claim_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """为整篇论文建骨架。返回 {pdf, hubs:[...]}。"""
    groups_by_id = {g["group_id"]: g for g in groups}

    def evidence_of(g: dict[str, Any]) -> str:
        cid = g.get("rep_claim_id", "")
        c = claim_map.get(cid) or {}
        return str(c.get("evidence_quote", "") or "")

    hubs = [g for g in groups
            if g["importance"] >= 5 and g["label"] in HUB_LABELS]
    # 稳定性：Hub 也按重要性/总分排序
    hubs.sort(key=lambda g: g["score"].get("total", 0), reverse=True)

    hub_out: list[dict[str, Any]] = []
    for hub in hubs:
        pools = build_pools(groups_by_id, hub)
        allowed = {rel: {c["group_id"] for c in cand}
                   for rel, cand in pools.items()}
        user = build_user_prompt({
            "group_id": hub["group_id"],
            "label": hub["label"],
            "rep_text": hub["rep_text"],
            "evidence": evidence_of(hub),
        }, pools)
        # 候选源池全空（如无 limitation / 无 method_core）：无源可挂，重试无意义
        if sum(len(c) for c in pools.values()) == 0:
            hub_out.append({
                "hub": hub["group_id"],
                "label": hub["label"],
                "importance": hub["importance"],
                "rep_text": hub["rep_text"],
                "edges": [],
                "isolated_reason": "no_sources",
            })
            continue
        # 建边：LLM 返回空边也算一次尝试（孤岛多为随机性，最多重试 3 次）
        edges: list[dict[str, Any]] = []
        for attempt in range(3):
            try:
                rows = llm.chat_json(SKELETON_SYSTEM, user, temperature=0.0)
            except (llm.LLMError, ValueError) as e:
                if attempt < 2:
                    print(f"  [skeleton] Hub {hub['group_id']} 调用失败，重试: {e}")
                else:
                    print(f"  [skeleton] Hub {hub['group_id']} 重试仍失败: {e}")
                continue
            cand = _extract_edges(rows, pools, allowed)
            if cand:
                edges = cand
                break
            if attempt < 2:
                print(f"  [skeleton] Hub {hub['group_id']} 无边，重试一次…")
        # 绑定证据（后端，不来自 LLM）
        bound = []
        for e in edges:
            src = groups_by_id.get(e["source"]) or {}
            src_claim = claim_map.get(src.get("rep_claim_id", "")) or {}
            bound.append({
                "relation": e["relation"],
                "source": e["source"],
                "why": e["why"],
                "evidence": str(src_claim.get("evidence_quote", "") or ""),
                "page": src_claim.get("page", 0) or (src.get("pages") or [0])[0],
                "ev": src.get("ev_state", ""),
            })
        hub_out.append({
            "hub": hub["group_id"],
            "label": hub["label"],
            "importance": hub["importance"],
            "rep_text": hub["rep_text"],
            "edges": bound,
            **({"isolated_reason": "no_edges"} if not bound else {}),
        })
    return {"pdf": pdf_name, "hubs": hub_out}
