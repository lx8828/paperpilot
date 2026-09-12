"""论文问答图：L0→L1→L2(自环)→L3→unknown 逐级加料漏斗。

分层（QA_FUNNEL_DESIGN.md）：
    report_l0 → judge_l0 ──够──→ answer(L0: overview+core_points)
                  │不够
    retrieve_claims(L1) → judge_l1 ──够──→ answer(L1: claims)
                  │不够(有 target_sections)          │不够(无把握)
                  ▼                                   ▼
              expand_l2(L2 自环)                  search_l3(L3)
                  │ 每步产窗口
              judge_l2 ──够──→ answer(L2: 窗口正文)
                  │不够 → expand_l2（换半径/圆心）
                  │ 圆心试完/预算尽(done)
                  ▼
              search_l3 → judge_l3 ──够──→ answer(L3: 独立全文)
                                        │不够
                                        ▼
                                   answer_unknown（诚实收尾）

调用：
    from paperpilot.graph import ask
    result = ask("这篇论文的核心结论是什么？", "2608.31079v1.pdf")
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from paperpilot.agents.nodes import (answer_unknown, expand_l2, generate_answer,
                                     judge_l0, judge_l1, judge_l2, judge_l3,
                                     report_l0, retrieve_claims, search_l3)
from paperpilot.agents.state import QAState


# ── 路由函数 ──────────────────────────────────────────────────────────────


def route_after_judge0(state: QAState) -> str:
    """judge_l0：够 → answer；不够 → L1 claim 检索。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "retrieve_claims"


def route_after_judge1(state: QAState) -> str:
    """judge_l1：够 → answer；不够且有 target_sections → L2；无把握(空) → 直接 L3。"""
    v = state.get("verdict") or {}
    if v.get("enough"):
        return "answer"
    return "expand_l2" if (v.get("target_sections") or []) else "search_l3"


def route_after_expand(state: QAState) -> str:
    """expand_l2 后：有推进（!done）→ judge_l2 判新窗口；已耗尽(done) → 直接 L3。

    注意：done 且 last window 已被 judge_l2 判过不足 → 不重复判同一窗口。
    """
    return "search_l3" if (state.get("l2") or {}).get("done") else "judge_l2"


def route_after_judge2(state: QAState) -> str:
    """judge_l2：够 → answer；不够 → 回 expand_l2 换半径/圆心（done 时 expand 无窗口→L3）。"""
    if (state.get("verdict") or {}).get("enough"):
        return "answer"
    return "expand_l2"


def route_after_judge3(state: QAState) -> str:
    """judge_l3：够 → answer；不够 → 诚实收尾。"""
    return "answer" if (state.get("verdict") or {}).get("enough") else "answer_unknown"


# ── 图 ─────────────────────────────────────────────────────────────────────


def build_qa_graph():
    g = StateGraph(QAState)
    for name, fn in [
        ("report_l0", report_l0),
        ("judge_l0", judge_l0),
        ("retrieve_claims", retrieve_claims),
        ("judge_l1", judge_l1),
        ("expand_l2", expand_l2),
        ("judge_l2", judge_l2),
        ("search_l3", search_l3),
        ("judge_l3", judge_l3),
        ("answer", generate_answer),
        ("answer_unknown", answer_unknown),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "report_l0")
    g.add_edge("report_l0", "judge_l0")
    g.add_conditional_edges("judge_l0", route_after_judge0,
                            {"answer": "answer", "retrieve_claims": "retrieve_claims"})

    g.add_edge("retrieve_claims", "judge_l1")
    g.add_conditional_edges("judge_l1", route_after_judge1,
                            {"answer": "answer", "expand_l2": "expand_l2",
                             "search_l3": "search_l3"})

    g.add_conditional_edges("expand_l2", route_after_expand,
                            {"judge_l2": "judge_l2", "search_l3": "search_l3"})
    g.add_conditional_edges("judge_l2", route_after_judge2,
                            {"answer": "answer", "expand_l2": "expand_l2"})

    g.add_edge("search_l3", "judge_l3")
    g.add_conditional_edges("judge_l3", route_after_judge3,
                            {"answer": "answer", "answer_unknown": "answer_unknown"})

    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


app = build_qa_graph()


def ask(question: str, pdf: str,
        history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """一键问答：跑完整张图，返回最终 State（answer/cites/debug/route…）。

    Args:
        question: 本轮用户问题
        pdf: assets/papers 下的论文文件名
        history: 之前轮次的对话 [{"role": "user"|"assistant", "content": str}, ...]，
                 供 judge/answer 理解"这个方法/它"等指代（真追问）。
    """
    state: QAState = {"question": question, "pdf": pdf}
    if history:
        state["history"] = [
            {"role": str(m.get("role")), "content": str(m.get("content"))}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")]
    return app.invoke(state)
