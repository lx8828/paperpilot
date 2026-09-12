"""ContextBuilder 门面：把检索证据组建成"可作答上下文"。

现状实现（answer.py 私有，收敛期委派）：
  _build_context(state) → 按 state 里有哪路证据自动定级并编号（L0 core_points / L1 claims /
  L2 窗口 / L3 全文块）；_compose() 两步法（L2/L3 先穷举 facts 清单再作答）；缺失断言复核闸门
  （浅层答"论文没写"→ 强制全文复核，防误拒，level 低时自动触发）。

证据与默认配置（RAG_COMPONENT_NOTES §2-⑥）：
- 缺失复核闸门对拒答可靠性有贡献（防幻觉 20/20 依赖）；两步法 facts 与 pass 正相关。
- 待办：上下文压缩（多块去冗余）单篇收益未验；与表格块接入配合。cites 引用编排不得破坏。
"""
from __future__ import annotations

from typing import Any


def build(state: dict[str, Any]):
    """从 state（question/pdf/overview/core_points/retrieved/chunks/l3_chunks…）产出
    (header, entries, level)。与 Generator 组合：Generator(state) 会自动内部调它。"""
    from paperpilot.agents.nodes.answer import _build_context
    return _build_context(state)


def extract_facts(question: str, context: str) -> list[dict[str, Any]]:
    """两步法第一步：穷举与问题相关的事实清单（防"踩点但漏细节"）。"""
    from paperpilot.agents.nodes.answer import _extract_facts
    return _extract_facts(question, context)


def compose(question: str, pdf: str, header: str, entries: list[dict[str, Any]],
            level: str, state: dict[str, Any] | None = None):
    """两步法第二步：带上下文+facts 生成 answer + cites。委派 answer._compose。"""
    from paperpilot.agents.nodes.answer import _compose
    return _compose(question, pdf, header, entries, level, state or {})


def audit_absence(question: str, pdf: str) -> dict[str, Any]:
    """缺失断言复核：浅层断言"论文没写"时全文复核能否作答。返回 {found, entries, reason}。"""
    from paperpilot.agents.nodes.answer import _audit_absence
    return _audit_absence(question, pdf)
