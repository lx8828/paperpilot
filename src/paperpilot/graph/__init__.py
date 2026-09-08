"""graph：LangGraph 编排层（只接线，节点逻辑在 agents/）。

    graph/qa_graph.py     v2 漏斗图（L0→L1→L2→L3，含 claims 层 + L2 扩窗）——保留作 A/B/回退
    graph/qa_graph_v3.py  v3 两级图（L0 总览 → L3 全局检索）——2026-09-08 定稿为默认
    将来：多篇对比图、主题追踪图等在此扩展。

定稿依据（qa/recall/ab_v3regress_result.json，250 题同裁判 A/B）：
    V0(v3) 203/250 (81.2%) ≥ A(v2) 201/250 (80.4%)，calls 3.7 vs 5.6 (-34%)，prompt +17%。
    定稿前跑：回归护栏（中文 QA176 + bench）确认无退化。切换默认后如需 A/B 走 ask_v2。
"""
from paperpilot.graph.qa_graph import ask as ask_v2, build_qa_graph as build_qa_graph_v2
from paperpilot.graph.qa_graph_v3 import ask as ask, build_qa_graph_v3

__all__ = ["ask", "ask_v2", "build_qa_graph_v2", "build_qa_graph_v3"]
