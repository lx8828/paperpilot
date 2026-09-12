"""v3 检索架构（Step1 纯两级基底）：L0 总览直答 → 不够 → L3 全局检索 → 答/拒。

动机（qa/RETRIEVAL_EXPLORATION_20260908.md 阶段 Q）：L2 这一层（节内圆心+扩窗自环）
在同 67 题 L2 目标群上可赢空间≈1 题（comb 45 vs 直 L3 44，噪声内），却带来整块工程
复杂度（judge_l1 定向/圆心/半径/预算自环/judge_l2）。v3 卸掉 L1/L2 层：

    report_l0 → judge_l0 ──够──→ answer(L0: overview+core_points)
                  │不够
                  ▼
              search_l3（全局：向量+BM25 混合，多查询改写默认关）
                  → judge_l3 ──够──→ answer(L3: 独立全文)
                              │不够
                              ▼
                          answer_unknown（诚实收尾/提炼原文）

复用现有节点（search_l3/judge_l3/generate_answer/answer_unknown）+ Router 组件。
generate_answer 靠 state 字段自动定级（有 l3_chunks → L3；否则 core_points → L0）。
本文件只做接线。

**2026-09-10：v2 四层漏斗（L1 claims 检索 / L2 圆心扩窗自环 / judge_l1 / judge_l2）
及其被证否的 V1 claims 分支已全部下线**，快照与设计文档归档在
`archive/qa_funnel_v2/`（原因与结论见该目录 README）。

评测入口：qa/recall/_ab_v3base.py（A=现状图 vs V=本图，同裁判 A/B）。
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from paperpilot.agents.nodes import (answer_unknown, generate_answer,
                                     judge_l3, search_l3)
from paperpilot.agents.state import QAState
from paperpilot.components.router import route_state


def route_after_judge0_pure(state: QAState) -> str:
    """纯两级：judge_l0 不够 → 直接全局 L3（无 L1/L2）。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "search_l3"


def route_after_judge3(state: QAState) -> str:
    """judge_l3：够 → answer(L3)；不够 → 诚实收尾。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "answer_unknown"


def _add_l3_tail(g, nodes):
    g.add_edge("search_l3", "judge_l3")
    g.add_conditional_edges("judge_l3", route_after_judge3,
                            {"answer": "answer", "answer_unknown": "answer_unknown"})


def build_qa_graph_v3():
    """v3 纯两级（Step1 基底 → 2026-09-09 Router 组件接入装配）：
    Router(L0 总览判定) → 够则 L0 直答 → 不够 → 直接全局 L3 → 答/拒。

    Router 节点 = components.router.route_state（内部 report_l0+judge_l0），把路由显式化，
    供将来扩展多维分支/多篇主题分类；行为与原 judge_l0 路由一致（定稿回归背书）。
    """
    g = StateGraph(QAState)
    for name, fn in [("router", route_state),
                     ("search_l3", search_l3), ("judge_l3", judge_l3),
                     ("answer", generate_answer),
                     ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_judge0_pure,
                            {"answer": "answer", "search_l3": "search_l3"})
    _add_l3_tail(g, None)
    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


def build_qa_graph_v3_nol3j():
    """v3 架构变体（2026-09-09 实验）：删 L3 judge，search_l3 → 一律试答。

    依据（"Judge 只做预算路由、验证管质量"）：judge_l0 判够直答省 L3 检索费，
    值得保留；而 L3 是叶子层，judge_l3 的"够/不够"没有更便宜的替代路径（不够只能
    送 unknown），只剩误判成本（cf93a：证据块里已有答案仍被判不够拒答）。
    删 judge_l3 后：L3 一律试答 → 质量交给输出闸门 gate（answer → gate 验证 →
    repair → 仍不过 gate 兜底拒答）。answer_unknown 节点保留但图内无入口。

    开启：env PAPERPILOT_V3_NOL3J=1（配合 PAPERPILOT_VALIDATOR_GATE=1 使用；
    gate 关时该变体等于"无验证的全量试答"，会放行无据答案，仅实验用）。
    """
    g = StateGraph(QAState)
    for name, fn in [("router", route_state),
                     ("search_l3", search_l3),
                     ("answer", generate_answer),
                     ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_judge0_pure,
                            {"answer": "answer", "search_l3": "search_l3"})
    g.add_edge("search_l3", "answer")
    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


def ask(question: str, pdf: str, history: list[dict[str, Any]] | None = None) -> dict:
    """v3 一键问答（与 graph.ask 同签名）。

    **2026-09-10 起默认 = nol3j 变体**（删 L3 judge、一律试答，质量交给 graph.ask 的输出闸门）：
    依据 ab_nol3j（100 题同裁判）：hard 19/50→29/50 (+10pt)，normal 45/50→44/50 (−1pt 噪声)，
    合计 64%→73%。关：env PAPERPILOT_V3_NOL3J=0（回到带 judge_l3 的 build_qa_graph_v3）。
    history: 之前轮次对话 [{role, content}]，供 judge/answer 理解指代（web /api/ask 用）。
    """
    import os
    if os.environ.get("PAPERPILOT_V3_NOL3J", "1") != "0":
        fn = build_qa_graph_v3_nol3j
    else:
        fn = build_qa_graph_v3
    state: QAState = {"question": question, "pdf": pdf}
    if history:
        state["history"] = [
            {"role": str(m.get("role")), "content": str(m.get("content"))}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")]
    return fn().invoke(state)
