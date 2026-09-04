"""问答图节点（L0→L1→L2→L3 逐级加料漏斗）。

graph/ 只做接线（StateGraph + 条件边），节点实现全部在 nodes/。
每个节点是纯函数：签名 (state: QAState) -> PartialState。

节点清单：
    report_l0       L0 零检索上下文（overview + core_points）
    judge_l0        L0 够不够（全貌类）
    retrieve_claims L1 claim embedding top12
    judge_l1        L1 够不够 + 不够时定向 target_sections
    expand_l2       L2 增量扩展步进（圆心 + 半径 + 分窗）
    judge_l2        L2 每轮窗口够不够
    search_l3       L3 独立全文检索（ChunkIndex）
    judge_l3        L3 够不够
    generate_answer 分档 answer（L0/L1/L2/L3）
    answer_unknown  诚实收尾
"""
from paperpilot.agents.nodes.answer import answer_unknown, generate_answer
from paperpilot.agents.nodes.judge import (judge_l0, judge_l1, judge_l2,
                                           judge_l3)
from paperpilot.agents.nodes.pull_chunk import expand_l2, search_l3
from paperpilot.agents.nodes.report import report_l0
from paperpilot.agents.nodes.retrieve import retrieve_claims

__all__ = [
    "report_l0", "judge_l0", "retrieve_claims", "judge_l1",
    "expand_l2", "judge_l2", "search_l3", "judge_l3",
    "generate_answer", "answer_unknown",
]
