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

复用全部现有节点（report_l0/judge_l0/search_l3/judge_l3/generate_answer/answer_unknown），
generate_answer 靠 state 字段自动定级（有 l3_chunks → L3；否则 core_points → L0）。
本文件只做接线，是"两级 + 分支"v3 的基底；后续在此上叠加分类器 + claims 分支 +
多维分支（见 classify/multidim 规划）。v2 漏斗图 qa_graph.py 保留作 A/B 与回退。

评测入口：qa/recall/_ab_v3base.py（A=现状图 vs V=本图，同裁判 A/B）。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paperpilot.agents.nodes import (answer_unknown, generate_answer,
                                     judge_l0, judge_l1, judge_l3,
                                     report_l0, retrieve_claims, search_l3)
from paperpilot.agents.state import QAState


def route_after_judge0(state: QAState) -> str:
    """judge_l0：够 → answer(L0)；不够 → 下钻。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "retrieve_claims"


def route_after_judge1(state: QAState) -> str:
    """judge_l1：够 → answer(L1, claims 精炼上下文)；不够 → 全局 L3（无 L2 层）。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "search_l3"


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


def build_qa_graph_v3_l1():
    """v3 完整版 = L0 总览 → L1 claims 分支（够则精炼直答）→ L3 全局（含兜底/拒答）。

    Step2：L1 作为 L3 前的低成本分支（claims 精炼上下文可答则省 L3 全文与带偏风险）。
    与纯两级 V0 的差异只在 judge_l0 不够后是否先试 claims。
    """
    g = StateGraph(QAState)
    for name, fn in [("report_l0", report_l0), ("judge_l0", judge_l0),
                     ("retrieve_claims", retrieve_claims), ("judge_l1", judge_l1),
                     ("search_l3", search_l3), ("judge_l3", judge_l3),
                     ("answer", generate_answer),
                     ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "report_l0")
    g.add_edge("report_l0", "judge_l0")
    g.add_conditional_edges("judge_l0", route_after_judge0,
                            {"answer": "answer", "retrieve_claims": "retrieve_claims"})
    g.add_edge("retrieve_claims", "judge_l1")
    g.add_conditional_edges("judge_l1", route_after_judge1,
                            {"answer": "answer", "search_l3": "search_l3"})
    _add_l3_tail(g, None)
    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


def build_qa_graph_v3():
    """v3 纯两级（Step1 基底）：L0 总览直答 → 不够 → 直接全局 L3 → 答/拒。"""
    g = StateGraph(QAState)
    for name, fn in [("report_l0", report_l0), ("judge_l0", judge_l0),
                     ("search_l3", search_l3), ("judge_l3", judge_l3),
                     ("answer", generate_answer),
                     ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "report_l0")
    g.add_edge("report_l0", "judge_l0")
    g.add_conditional_edges("judge_l0", route_after_judge0_pure,
                            {"answer": "answer", "search_l3": "search_l3"})
    _add_l3_tail(g, None)
    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


def ask(question: str, pdf: str, with_claims: bool = False) -> dict:
    """v3 一键问答（与 graph.ask 同签名）。

    默认纯两级 V0（2026-09-08 定稿：ab_v3regress 203/250 ≥ v2 201/250）。
    with_claims=True 走 V1（含 L1 claims 分支，Step2 实验证明 62<64，仅作对照保留）。
    """
    fn = build_qa_graph_v3_l1 if with_claims else build_qa_graph_v3
    return fn().invoke({"question": question, "pdf": pdf})
