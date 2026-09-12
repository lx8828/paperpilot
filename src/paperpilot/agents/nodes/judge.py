"""judge 节点：按档位的"够不够"判定（prompt 工厂）。

每层"够"的标准不同，模板各自说清"证据是什么 / 什么算够 / gap 怎么写"。

节点函数（v3 现役）：
    judge_l0   证据=overview+core_points      → 够：answer(L0)
    judge_l3   证据=l3_chunks（独立，不继承）  → 够：answer(L3) / 不够：answer_unknown

输出统一 Verdict 结构：{enough, target_sections, gap}。
target_sections 是 v2 L1 的遗留字段（v3 已无 L1/L2 层），现役节点输出空数组。

LLM 失败策略：L0 失败按 enough=True 降级用已拿证据回答（不中断）；
L3 失败按 enough=False 走诚实收尾（防凭空编造）。

**2026-09-10：v2 的 judge_l1 / judge_l2 / judge_l2_strict 已下线归档**
（快照见 archive/qa_funnel_v2/judge_snapshot_full.py）。
"""
from __future__ import annotations

import os
import re
from typing import Any

from paperpilot.agents.state import QAState
from paperpilot.tools import llm

_JSON = """只输出 JSON，不要多余文字。格式：
{
  "enough": true 或 false,
  "target_sections": [],
  "gap": "若 enough=false 一句话说明缺什么；enough=true 则留空"
}
"""

# ── L0：全貌类问题被 overview/core_points 覆盖吗？───────────────────────────

_SYS_L0 = (
    "你是严谨的信息充分性判定器。判定：仅凭『论文概述 + 核心要点』能否准确回答用户的全貌类问题"
    "（如论文讲了什么、核心贡献、主要结论）。只输出 JSON。"
)

_L0_TPL = """论文概述：
{overview}

核心要点（importance≥5 的主张）：
{core}
{history}
用户问题：{question}

请判断：仅凭上述信息能否**准确且完整**地回答该问题？
若问题是"全貌/总结"类且概述已覆盖 → enough=true。
若问题需要正文细节（具体方法、公式、数字、表格、附录）→ enough=false，gap 说明缺哪类。
{_json}"""


# ── 题型完整性自检（V2：治"部分相关就判够"）─────────────────────────────────
# 背景（QASPER_EVAL_LOG.md E1）：大量 fail 是列举/数字/对比/方法题拿到部分证据就收口。
# Judge 判 enough 前必须先按题型自查完整性——"相关"不等于"完整"。
_SYS_L3 = (
    "你是严谨的信息充分性判定器。给定针对用户问题检索到的若干正文段落（全文范围），"
    "判断能否据此**实质性地回答**问题；只有确认全文检索结果与问题无关、确实无法作答时，"
    "才判 enough=false（将诚实告知用户）。只输出 JSON。"
)

_L3_TPL = """全文检索到的正文段落（独立全文检索，未继承前面的判定）：
{chunks}

用户问题：{question}

请判断：
- 若这些段落提供了回答所需的**实质信息**（含具体数字/定义/列举/对比依据，
  即便答案需要跨段落拼凑或做简单推断，如 4262/150≈27.4、多段分别定义不同概念）
  → enough=true。
- 是非/有无类问题（Do/Is/Are/Does/是否/有没有，预期 yes/no）：
  判 enough 的标准是"能否据检索段落作出**确定的**是/否判断"，**不要求**段落里出现
  "是/否/没有"字样，可按以下两种情况判够：
  · 段落正面描述了该做法/机制/对象 → 可答 yes；
  · 段落**确实在讨论问题所指的那个对象/主题**（如问"标注是否用众包"，段落就在讲
    该标注流程，但只提到两名人工标注员、全程未提任何众包平台）→ 可答 no
    （被讨论过却无该做法 = 论文没采用）。
  注意护栏：若段落讲的其实是**别的对象/别的实验**（如问模型结构消融，段落却在讲
  特征组合实验；问是否做某任务，段落只讲数据集用途），或段落与问题主题完全无关
  → **不得**据此判够，仍判 enough=false，gap 写"检索段落在讲别的，缺与问题直接
  相关的内容"。
- 只有当检索到的段落**与问题完全无关、不含任何可作答的实质内容**时 → enough=false，
  gap 用一句话说明缺什么（这段话会直接呈现给用户，请写得像人话）。

注意：你是最后一层兜底，判 enough=false 意味着放弃作答、告知用户"论文里没有"。
请只在确实检索不到任何相关内容时才这样做——不要因为"段落没直接写出答案句"、
"需要自己拼凑/推断"而误判不够，那会白白丢掉已经检索到的答案。
{_json}"""


def _fmt_history(state: QAState) -> str:
    """把之前轮次的对话格式化为"对话历史"块（追问时理解"这个方法/它"等指代）。"""
    hist = state.get("history") or []
    if not hist:
        return ""
    lines = ["", "对话历史（本轮问题可能指代其中的内容，判定时请结合理解）："]
    for m in hist[-6:]:
        who = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"  {who}: {(m.get('content') or '')[:400]}")
    return "\n".join(lines)


def _parse(raw: object, default_enough: bool) -> dict[str, Any]:
    """解析 judge JSON；非法输入退回默认 enough。"""
    if isinstance(raw, dict):
        enough = bool(raw.get("enough", default_enough))
        secs = raw.get("target_sections") or []
        if not isinstance(secs, list):
            secs = []
        return {
            "enough": enough,
            "target_sections": [str(s) for s in secs],
            "gap": str(raw.get("gap", "") or ""),
        }
    return {"enough": default_enough, "target_sections": [], "gap": ""}


def _appear(state: QAState, lvl: str) -> list[str]:
    route = list(state.get("route") or [])
    if lvl not in route:
        route.append(lvl)
    return route


def _fmt_core(points: list[Any]) -> str:
    lines = []
    for i, c in enumerate(points, 1):
        lines.append(f"[{i}] [{c.get('label','')}]({c.get('importance',0)}分) {c.get('text','')}")
    return "\n".join(lines) if lines else "（无核心要点）"




def _fmt_chunks(chunks: list[Any], max_show: int = 8) -> str:
    lines = []
    for i, c in enumerate(chunks[:max_show], 1):
        text = (c.get("text") or "").strip()
        # 完整展示本块文本（不截断）：chunk 分块期已按段落原子切到 ~4000 字符，
        # judge 再砍会丢证据（Run1 真漏检根因，900/3000 截断均已移除）。
        lines.append(f"[{i}] (p{c.get('page','?')} {c.get('section','')}) {text}")
    return "\n".join(lines) if lines else "（暂无正文）"


# ── 节点：judge_l0 ────────────────────────────────────────────────────────────


def judge_l0(state: QAState) -> dict[str, Any]:
    user = _L0_TPL.format(
        overview=state.get("overview") or "（无概述）",
        core=_fmt_core(state.get("core_points") or []),
        history=_fmt_history(state),
        question=state.get("question", ""),
        _json=_JSON,
    )
    try:
        raw = llm.chat_json(_SYS_L0, user, temperature=0.0)
        v: dict[str, Any] = _parse(raw, default_enough=True)
    except llm.LLMError as e:
        v = {"enough": True, "target_sections": [], "gap": f"judge_l0 失败: {e}"[:120]}
    debug = dict(state.get("debug") or {})
    debug["judge_l0"] = {"gap": (v["gap"] or "")[:120], "enough": v["enough"]}
    return {
        "verdict": v,
        "route": _appear(state, "L0"),
        "debug": debug,
    }


# ── 节点：judge_l3（独立保底）────────────────────────────────────────────────


def judge_l3(state: QAState) -> dict[str, Any]:
    l3 = state.get("l3_chunks") or []
    # 检索到了的块 judge 要**全看得到**（旧实现写死 12，与 L3_TOP_K 对齐）。
    # ⚠️ 开 `PAPERPILOT_EXT_QUOTA` 后候选会多于 12（并集追加的表块在末尾），写死 12 会把
    #    追加块截掉 → judge 判"不够"直接走 answer_unknown，**并集收益被闸门吃掉**。
    #    故改为按实际块数对齐；quota=0 时 len(l3)==12，行为与旧实现逐位相同。
    user = _L3_TPL.format(
        chunks=_fmt_chunks(l3, max_show=len(l3) or 1),
        question=state.get("question", ""),
        _json=_JSON,
    )
    try:
        raw = llm.chat_json(_SYS_L3, user, temperature=0.0)
        v: dict[str, Any] = _parse(raw, default_enough=False)
    except llm.LLMError as e:
        # 保底层判定失败 → 诚实收尾，不凭空编造
        v = {"enough": False, "target_sections": [], "gap": f"判定失败: {e}"[:120]}
    debug = dict(state.get("debug") or {})
    debug["judge_l3"] = {"gap": (v["gap"] or "")[:120], "enough": v["enough"],
                         "n_chunks": len(l3)}
    return {
        "verdict": v,
        "route": _appear(state, "L3"),
        "debug": debug,
    }
