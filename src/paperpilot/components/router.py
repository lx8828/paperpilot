"""Router 门面（显式化）：判断问题该走哪条路径。

现状（v3 定稿）：judge_l0 判"overview+core_points 能否完整回答"——够 → L0 直答；
不够 → 全局检索(L3)。本门面把这一步显式化为 decide()，输出稳定 action，为将来扩展
（多维分解分支 / 多篇主题归并 / 难度分级）留类型位。

证据与默认配置（RAG_COMPONENT_NOTES §2-②）：
- LLM 预测当 Router/Judge 危险：L2 时代 judge 判够精度≈61%；L0 二分可靠因判断对象简单
  （有没有提到，而非证据够不够拼答案）+ 判错可降级不致命。
- 铁律：Router 输出带降级链（action 之后仍有 L3 兜底），绝不做"分类定生死"的硬路由。
- 默认：二类（l0_answer | global_retrieve）。future: "multi_dim"（多维分支，多方法对比题
  已测 82% 不需，未启用）、"topic"（多篇主题，未实现）。
"""
from __future__ import annotations

from typing import Any

# action 枚举（稳定公开）
L0_ANSWER = "l0_answer"          # overview+core_points 足够，直答
GLOBAL_RETRIEVE = "global_retrieve"  # 需全局检索（L3）
ACTION_NAMES = (L0_ANSWER, GLOBAL_RETRIEVE)


def load_l0(pdf: str) -> dict[str, Any]:
    """读取论文报告层 L0 材料：title/overview/core_points（带 claim 锚）。"""
    from paperpilot.agents.nodes.report import report_l0
    st: dict[str, Any] = {"pdf": pdf}
    return report_l0(st)


def decide(question: str, pdf: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Router 决策：跑 L0 材料 + judge_l0，返回
    {action, enough, gap, overview, core_points, n_core_points}。

    action 含义：
      l0_answer        → Generator 用 overview+core_points 直答；
      global_retrieve  → 交给 Retriever（向量+BM25 混合全文检索）。
    任何 action 之后都保留降级链：Retriever 若判不足 → 诚实收尾（不硬答/不编）。
    """
    st: dict[str, Any] = {"question": question, "pdf": pdf}
    if history:
        st["history"] = history
    out = route_state(st)
    st.update(out)
    verdict = st.get("verdict") or {}
    enough = bool(verdict.get("enough"))
    core = list(st.get("core_points") or [])
    return {
        "action": L0_ANSWER if enough else GLOBAL_RETRIEVE,
        "enough": enough,
        "gap": str(verdict.get("gap") or ""),
        "title": st.get("title", ""),
        "overview": st.get("overview", ""),
        "core_points": core,
        "n_core_points": len(core),
    }


def route_state(state: dict[str, Any]) -> dict[str, Any]:
    """Router 节点（LangGraph 用）：读 question/pdf → 产出
    {title, overview, core_points, verdict, route, debug}，等价原 report_l0+judge_l0 两节点。
    v3 图以此为第一级（2026-09-09 Router 显式化接入装配）。
    """
    from paperpilot.agents.nodes.judge import judge_l0
    from paperpilot.agents.nodes.report import report_l0

    st = dict(state)
    st.update(report_l0(st))
    st.update(judge_l0(st))
    keys = ("title", "overview", "core_points", "verdict", "route", "debug")
    return {k: st[k] for k in keys if k in st}
