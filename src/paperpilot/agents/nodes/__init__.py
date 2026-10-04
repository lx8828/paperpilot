"""问答图节点（L0→L1→L2→L3 逐级加料漏斗）。

graph/ 只做接线（StateGraph + 条件边），节点实现全部在 nodes/。
每个节点是纯函数：签名 (state: QAState) -> PartialState。

节点清单：
    report_l0       L0 零检索上下文（overview + core_points）
    judge_l0        L0 够不够（全貌类）——由 components/router 调用
    read_full       **全文直读**（`PAPERPILOT_QA_READER=fullctx` = 默认问答读取器）
    search_l3       L3 独立全文检索（RAG-2，保留：env 切回 / 内部重检索用）
    judge_l3        L3 够不够（仅 nol3j=0 的图使用）
    generate_answer 分档 answer（L0/L3/**FULLCTX**）
    answer_unknown  诚实收尾（仅 nol3j=0 的图使用）

**2026-09-27：问答读取器默认换为 `read_full`（全文直读）** —— 依据是 S/C 判据
（≤5 篇 ≈ 48k token，放得下 → 直读优于块级挑块）。**RAG-2 未删**：
`PAPERPILOT_QA_READER=retrieval` 切回；问答链内部两处独立重检索仍走它。

**2026-09-10：v2 漏斗节点（retrieve_claims / judge_l1 / expand_l2 / judge_l2）已下线归档**
（archive/qa_funnel_v2/），不再导出。
"""
from paperpilot.agents.nodes.answer import answer_unknown, generate_answer
from paperpilot.agents.nodes.judge import judge_l0, judge_l3
from paperpilot.agents.nodes.pull_chunk import search_l3
from paperpilot.agents.nodes.read_full import answer_fullctx
from paperpilot.agents.nodes.report import report_l0
from paperpilot.agents.nodes.set_answer import answer_set

__all__ = [
    "report_l0", "judge_l0", "answer_fullctx", "answer_set", "search_l3", "judge_l3",
    "generate_answer", "answer_unknown",
]
