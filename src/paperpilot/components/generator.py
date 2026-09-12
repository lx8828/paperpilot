"""Generator 门面：生成答案（LLM 作答 + cites 引用）。

现状实现：agents.nodes.answer.generate_answer（按档位 L0-L3 组装上下文并作答；
缺失断言复核闸门在 L0-L2 自动兜 L3）。主链路模型 deepseek-chat。

默认配置与铁律（RAG_COMPONENT_NOTES §2-⑦）：temperature=0；**cites 输出是产品核心卖点
（溯源闭环），任何重构不得丢**。底层 LLM 封装见 paperpilot.tools.llm（OpenAI 兼容，
支持 usage 计量 / judge 前缀 / 超时）。
"""
from __future__ import annotations

from typing import Any


def generate(state: dict[str, Any]) -> dict[str, Any]:
    """端到端生成节点：喂 QAState（含 question/pdf 及已就绪的证据字段），
    产出 answer/cites/route/debug。等价 graph 内 answer 节点。"""
    from paperpilot.agents.nodes.answer import generate_answer
    return generate_answer(state)


def generate_unknown(state: dict[str, Any]) -> dict[str, Any]:
    """诚实收尾生成：判不足时的兜底（提炼相关原文给用户，不编造）。"""
    from paperpilot.agents.nodes.answer import answer_unknown
    return answer_unknown(state)
