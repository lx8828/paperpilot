"""QAState：论文问答图（L0→L1→L2→L3 逐级加料漏斗）的数据契约。

LangGraph 中 State 是整张图的共享字典：所有节点签名 (state) -> PartialState，
条件边读 state 字段分叉。先定 State = 定每个节点该产出什么、后续节点能消费什么。

分层漏斗（详见 QA_FUNNEL_DESIGN.md）：
    L0 Report     overview + core_points（零检索，不唤醒 embedding）
    L1 Claims     embedding top12（此时才加载模型）
    L2 Section    target_sections 定向圆心 + 同节增量扩展（分窗隔离 + 预算账本）
    L3 Global     ChunkIndex 独立全文检索（不继承 L0~L2 任何证据）

字段按图的数据流组织：
    输入    question / pdf
    L0      overview / core_points
    L1      retrieved（top12 ClaimHit，含 home_section）
    L2      chunks（当前圆心窗口）+ l2（圆心/半径/账本游标）
    L3      l3_chunks（独立检索结果，与 L2 隔离）
    判定    verdict（enough / target_sections / gap）
    输出    answer / cites / debug（route 履历）
"""
from __future__ import annotations

from typing import Any, TypedDict

# ═══════════════════════ 子结构 ═══════════════════════


class ClaimHit(TypedDict):
    """检索命中一条主张（来自 report.groups 的展示切片 + 直达原文字段）。"""

    gid: str
    rep_claim_id: str          # 引用链：直达 evidence_quote/page/chunk
    label: str                 # core_claim/result_primary/…
    importance: int            # 1-5
    text: str                  # 主张文本（中文 rep_text）
    sections: list[str]        # 出现的顶层章节
    pages: list[int]
    evidence: str              # 代表 claim 的原文证据句
    page: int                  # 证据页码
    chunk_id: str              # 升级定位原文（claim 实际出处 chunk）
    home_section: str          # claim 实际出处顶层节（L2 圆心定向用）
    score: float               # embedding 相似度


class Verdict(TypedDict):
    """judge 节点：LLM 判定当前证据是否足够 + 不够时下钻方向。"""

    enough: bool               # True → answer；False → 按 target_sections 下钻
    target_sections: list[str]  # 不够时：按"最可能藏答案"降序的有序候选（下一层取料方向）
    gap: str                   # 不够时缺什么（人话，供下钻/诚实收尾措辞）


class ChunkText(TypedDict):
    """升级拉取的正文原文片段。"""

    chunk_id: str
    page: int
    section: str               # 章节路径尾巴，供引用展示
    text: str


class L2Cursor(TypedDict):
    """L2 增量扩展的确定性游标（圆心顺序 / 半径 / 预算账本）。"""

    centers: list[str]         # 有序圆心 section（代码从 verdict.target_sections 截断）
    center_i: int              # 当前圆心下标
    radius: int                # 当前圆心已扩展到的半径（下一轮 +1）
    center_chunk: str          # 当前圆心的 chunk_id（已定位）
    pulled: list[str]          # 已拉过的 chunk_id（预算账本，防重叠重复）
    budget_used: int           # 按 pulled 新增唯一 chunk 计数


class Cite(TypedDict):
    """答案的可溯源引用（引用链：ref → gid/claim_id → 原文证据 + 页码）。"""

    ref: str                   # 外部引用格式 "<pdf>#<gid>" 或短 gid
    gid: str
    claim_id: str              # 直达原文的 claim（evidence_quote 在其上）
    evidence: str
    page: int


# ═══════════════════════ QAState ═══════════════════════


class QAMessage(TypedDict):
    """一轮对话消息（用户追问支持：judge/answer 参考历史理解指代）。"""

    role: str                  # "user" / "assistant"
    content: str               # 问题或回答文本


class QAState(TypedDict, total=False):
    # ── 输入（图调用方提供）──
    question: str              # 用户问题（本轮）
    pdf: str                   # storage/papers 下的论文文件名
    history: list[QAMessage]   # 之前轮次的对话（供 judge/answer 理解"这个方法/它"等指代）

    # ── L0 · Report（report_l0 节点产出，零检索）──
    title: str
    overview: str              # 论文概述（L0 answer 直接消费）
    core_points: list[dict[str, Any]]  # 必读主张（增强后含 gid/evidence/page 锚）

    # ── L1 · Claims（retrieve_claims 节点产出）──
    retrieved: list[ClaimHit]  # embedding 命中的 topK 主张（含 home_section 供 L2 定向）

    # ── L2 · Section（expand_l2 节点产出，当前圆心窗口）──
    chunks: list[ChunkText]    # 当前圆心累积窗口（切换圆心时清空，分窗隔离）
    l2: L2Cursor               # 圆心/半径/预算账本游标

    # ── L3 · Global（search_l3 节点产出，独立保底）──
    l3_chunks: list[ChunkText]  # 独立全文检索结果（与 L2 的 chunks 隔离，不混合）

    # ── 判定层（judge_* 节点产出）──
    verdict: Verdict           # enough / target_sections / gap

    # ── 输出（answer 节点产出）──
    answer: str                # 最终回答文本
    cites: list[Cite]          # 溯源引用（前端可渲染 + 将来跳 PDF）
    route: list[str]           # 履历：实际走过的层（debug）
    debug: dict[str, Any]      # 调试信息（n_retrieved / n_chunks / 预算账本等）
