"""agents：问答 Agent 的节点逻辑层（不依赖 langgraph，纯函数可单测）。

分层：
    agents/state.py     图的数据契约（QAState）
    agents/nodes/       图节点（retrieve → judge → pull_chunks → answer）
    graph/              组装（StateGraph 接线 + 条件边）
"""
