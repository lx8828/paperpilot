"""judge 节点：四层分模板（prompt 工厂）。

核心原则（QA_FUNNEL_DESIGN.md §3.5）：每层"够"的标准不同，模板各自说清
"证据是什么 / 什么算够 / gap 怎么写 / target_sections 从哪选"。

节点函数：
    judge_l0   证据=overview+core_points          → 够：answer_l0
    judge_l1   证据=L0+retrieved claims           → 够：answer_l1 / 不够：选 home_section → L2
    judge_l2   证据=L0+claims+当前窗口 chunks      → 够：answer_l2 / 不够：代码推进窗口或上 L3
    judge_l3   证据=l3_chunks（独立，不继承）       → 够：answer_l3 / 不够：answer_unknown

所有层输出同一 Verdict 结构：{enough, target_sections, gap}。
target_sections 只在 L1 有意义（候选=命中 claims 的 home_section，有序去重），
L0/L2/L3 输出空数组即可（由确定性逻辑决定下一步，judge 不做微观定位）。

LLM 失败策略：L0/L1/L2 失败按 enough=True 降级用已拿证据回答（不中断）；
L3 失败按 enough=False 走诚实收尾（防凭空编造）。
"""
from __future__ import annotations

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
_COMPLETENESS = """【判 enough 前必须做题型完整性自查】
先识别问题意图属于哪类，再检查眼前证据是否满足该类的"完整"要求：

- 列举类（问"有哪些/哪些X/包括什么/用什么方法/什么特征"）：
  证据必须列出**全部**成员。若只见 "such as / e.g. / including / 部分示例 / 等"
  或只提到零星几项而未穷举 → **不够**，gap 写"还缺哪些成员"。
- 数字类（问"多少/多大/多快/多高/多少数据/具体值"）：
  证据必须含**具体数值**。只有"更好/更快/显著优于/大幅"等定性比较而无数字 → **不够**，
  gap 写"需要具体数值"。
- 对比类（问"A与B比 / 与X相比 / 哪个更好 / difference"）：
  证据必须覆盖**所有被比对象和所有对比维度**。只见单边或部分指标 → **不够**，
  gap 写"缺哪一方/哪一维度的对比"。
- 方法/过程类（问"怎么做/如何/流程/pipeline"）：
  证据须覆盖**完整步骤或机制**。只有动机/结论/开头段、缺中间关键步骤 → **不够**，
  gap 写"缺哪一步"。

关键原则：证据"提到了相关内容"≠"能完整准确作答"。
若属于上述任一类且证据不全 → enough=false，即使眼前内容看起来相关。
"""


# ── L1：主题级问题 claims 够吗？不够则定向到哪节？───────────────────────────

_SYS_L1 = (
    "你是严谨的信息充分性判定器。给定论文概述与检索到的若干主张（claim），"
    "判断它们是否足以**完整准确**回答用户问题；若不足，从候选章节里指出最可能藏答案的 1~2 节。"
    "只输出 JSON。"
)

_L1_TPL = """论文概述：
{overview}

检索到的相关主张（每条末行是它的**出处章节** home_section）：
{items}
{history}
用户问题：{question}

请判断：
1. 仅凭上述主张能否**完整准确**回答？能 → enough=true，target_sections=[]。
2. 若不能 → enough=false，并从上方条目出现过的 home_section 里选 1~2 个
   **最可能包含答案正文**的章节填入 target_sections（必须原样使用上方出现过的名字，
   不要新造）；若你完全没有把握任何一节，则 target_sections=[]（会触发全文检索兜底）。
   gap 用一句话说明缺什么。
{_json}

{completeness}"""


# ── L2：当前窗口正文够了吗？─────────────────────────────────────────────────

_SYS_L2 = (
    "你是严谨的信息充分性判定器。给定论文概述与已拉取的**正文段落窗口**，"
    "判断这些原文是否足以**精确**回答用户问题（含数字/公式/表格细节）。只输出 JSON。"
)

_L2_TPL = """论文概述：
{overview}

当前拉取的正文段落（每段标了出处章节与页码）：
{chunks}
{history}
用户问题：{question}

请判断：
- 若这些正文**直接包含能完整作答的全部原句** → enough=true。
- 若明显缺失（答案在这段之外 / 数值对不上 / 枚举不全 / 该节不含此内容）→ enough=false，gap 说明还缺什么。
不要因为"理论上原文更权威"而无依据地判不够——若眼前正文已给出完整答案就判够。
{_json}

{completeness}"""


# ── L3：全文检索的独立保底───────────────────────────────────────────────────

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


def _fmt_claims(retrieved: list[Any]) -> str:
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(
            f"[{i}] [{r.get('label','')}]({r.get('importance',0)}分) {r.get('text','')}\n"
            f"    证据: {r.get('evidence','')}\n"
            f"    home_section: {r.get('home_section','') or '(未知)'}"
        )
    return "\n".join(lines) if lines else "（未检索到相关主张）"


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


# ── 节点：judge_l1 ────────────────────────────────────────────────────────────


def judge_l1(state: QAState) -> dict[str, Any]:
    retrieved = state.get("retrieved") or []
    # 候选 home_section：按 retrieved 顺序去重
    seen: set[str] = set()
    candidates: list[str] = []
    for r in retrieved:
        s = r.get("home_section") or ""
        if s and s not in seen:
            seen.add(s)
            candidates.append(s)
    user = _L1_TPL.format(
        overview=state.get("overview") or "（无概述）",
        items=_fmt_claims(retrieved),
        history=_fmt_history(state),
        question=state.get("question", ""),
        _json=_JSON,
        completeness=_COMPLETENESS,
    )
    try:
        raw = llm.chat_json(_SYS_L1, user, temperature=0.0)
        v: dict[str, Any] = _parse(raw, default_enough=True)
    except llm.LLMError as e:
        v = {"enough": True, "target_sections": [], "gap": f"judge_l1 失败: {e}"[:120]}
    # 只保留候选里真实存在过的 section（防模型自造名字）
    if not v["enough"]:
        valid = [s for s in v["target_sections"] if s in seen]
        v["target_sections"] = valid
    debug = dict(state.get("debug") or {})
    debug["judge_l1"] = {
        "gap": (v["gap"] or "")[:120],
        "enough": v["enough"],
        "target_sections": v["target_sections"],
        "candidates": candidates[:6],
    }
    return {
        "verdict": v,
        "route": _appear(state, "L1"),
        "debug": debug,
    }


# ── 节点：judge_l2（L2 自环中每轮扩完判一次）───────────────────────────────


def judge_l2(state: QAState) -> dict[str, Any]:
    chunks = state.get("chunks") or []
    user = _L2_TPL.format(
        overview=state.get("overview") or "（无概述）",
        chunks=_fmt_chunks(chunks),
        history=_fmt_history(state),
        question=state.get("question", ""),
        _json=_JSON,
        completeness=_COMPLETENESS,
    )
    try:
        raw = llm.chat_json(_SYS_L2, user, temperature=0.0)
        v: dict[str, Any] = _parse(raw, default_enough=True)
    except llm.LLMError as e:
        v = {"enough": True, "target_sections": [], "gap": f"judge_l2 失败: {e}"[:120]}
    debug = dict(state.get("debug") or {})
    debug["judge_l2"] = {"gap": (v["gap"] or "")[:120], "enough": v["enough"],
                         "n_chunks": len(chunks)}
    return {
        "verdict": v,
        "route": _appear(state, "L2"),
        "debug": debug,
    }


# ── 节点：judge_l3（独立保底）────────────────────────────────────────────────


def judge_l3(state: QAState) -> dict[str, Any]:
    l3 = state.get("l3_chunks") or []
    user = _L3_TPL.format(
        chunks=_fmt_chunks(l3, max_show=10),
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
