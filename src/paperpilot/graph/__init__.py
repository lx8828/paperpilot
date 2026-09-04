"""graph：LangGraph 编排层（只接线，节点逻辑在 agents/）。

    graph/qa_graph.py   论文问答图（retrieve → judge →{answer | pull_chunks}→ answer）
    将来：多篇对比图、主题追踪图等在此扩展。
"""
from paperpilot.graph.qa_graph import ask, build_qa_graph

__all__ = ["ask", "build_qa_graph"]
