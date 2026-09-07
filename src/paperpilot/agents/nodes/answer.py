"""answer 节点：基于"最终足够的那层证据"生成回答与 cites。

分档上下文（QA_FUNNEL_DESIGN.md §3.5，answer 依据 judge 判够的那档取料）：
    L0 answer：overview + core_points（claims 样式条目）
    L1 answer：overview + retrieved claims
    L2 answer：overview + 当前窗口 chunks（judge_l2 判够的依据）
    L3 answer：只用 l3_chunks（独立保底，不继承 overview 等上层料）

引用方案：上下文条目编号 [1..n]，LLM 在回答中标注 [n]；
后端把 [n] 映射回真实命中（gid/claim_id/evidence/page 或 chunk_id/page）生成 cites——
LLM 不自由编造证据文本，引用锚定在检索结果上。

answer_unknown：L3 仍不足时的诚实收尾（"我不知道，缺 XX"），不编造。
"""
from __future__ import annotations

import re
from typing import Any

from paperpilot.agents.embedder import ChunkIndex
from paperpilot.agents.nodes.judge import judge_l3
from paperpilot.agents.state import QAState
from paperpilot.tools import llm

SYSTEM = (
    "你是严谨的论文问答助手，只能依据提供的『论文概述 / 检索主张 / 正文段落』作答。"
    "回答须标注引用编号 [n]（指向对应信息条目）。用中文回答，控制在 3~6 句（仅当问题要求"
    "『列举全部成员』时才可超过句数，以列全为准）。"
    "\n\n回答准则："
    "1. 若问题可用『是/否』回答（如 'Is/Are/Do/Does ... ?'），必须基于证据给出明确的『是』或『否』"
    "   结论，不得用『无法确认/不确定/无法判断』等含糊话回避——证据支持哪个就答哪个。"
    "2. 对象裁决：作答前先明确问题问的到底是哪个对象。若证据/概述中出现多个同域候选"
    "   （如：多个 baseline 或 SOTA、训练 vs 评测数据、检索方法与分类方法、源语言 vs 目标语言、"
    "   '论文提出的方法' vs '对比的基线'、不同实验），只回答与问题字面**直接对应**的那个，"
    "   并在开头点明'你问的 X 对应论文中的 Y'；不要因某候选在概述或段首更显眼就答它。"
    "3. 不要滥用『信息不足/论文未提及』：仅当给出的证据里**确实没有任何**相关内容时才这样说；"
    "   若证据已包含答案（哪怕只有一句），必须直接作答，不能因'想更全面'而推说信息不足。"
    "4. 问『哪些/什么/多少』类问题时，从给定证据中尽可能完整地列出所有成员或给出具体数值。"
    "5. 禁止编造数字、方法或结论——一切说法必须有上方证据支撑。"
    "6. 作答前必须先**通读全部条目**（从 [1] 到最后一个），把与问题相关的片段找齐；"
    "   答案若分散在多个条目里，必须合并引用（如 [1][4]），禁止只引第一条命中的条目。"
    "7. 问题要『数值/幅度/差量/提升多少』时，只要条目中出现具体数字（分数、百分比、倍率、"
    "   时长等），就必须原样引用该数字作答；**严禁**因'没在第一眼看到'就说『文中未给出数值』——"
    "   除非你把全部条目逐条读完后确认无任何数字。"
    "8. 若答案需由多个数值推算（如平均值=总数÷样本数、差量=A−B），请写出算式并给出计算结果，"
    "   不要只停留在'约/较大/显著'这类定性描述。"
    "9. 列表题（问名称/成员/清单）若证据中逐项出现，必须全部列出并给每项来源；"
    "   若证据只给出部分，如实说明'证据中出现了……'并列出所见，不得断言'未列出/无清单'。"
    "10. 成员归并：同一实体若以不同形式出现（如同一语言对写为 EN-ES 与 ES-EN 两种方向、"
    "   同一方法的全称与缩写），应归并成**一个**成员，按论文整体口径统计数量，"
    "   不要把方向变体/同义重复算成多个成员。"
    "11. 防自毁：若你已在上方条目或候选事实清单中转述/引用了与答案直接对应的具体内容"
    "   （名称/数值/机制/来源/成员），就必须以此作答并给出明确结论，"
    "   **禁止**再用『论文未提供/无法确认/信息不足』之类的话收尾否定自己已找到的内容；"
    "   只有当候选事实清单为空、且逐条核对原文确实无相关内容时，才可如实说明证据不足。"
    "12. 概念/方法题必须点名：若答案对应的是某个**具体**模型/方法/数据集/成员"
    "   （如 word2vec、SUMBT、Meta-LSTM、MFD、HDSA），必须写出该专名本身，"
    "   禁止只给类别描述（如'基于词嵌入的模型''某基线'）而不给名字。"
    "13. 收尾核对（列表/数值/成员题必做）：成稿后把你要列的每一项，回到上方条目逐项核对"
    "   ——问题若带数量（'6 个语言对'『全部 12 个方法』），列出的项数必须与引用段实际出现"
    "   的成员一致；发现还有同行成员未列出（常藏在描述性长句/后半段/相邻条目里）就补上，"
    "   不要因'答案已够长'而省略。"
)

# 两步法第一步：从给定条目中穷举式提取"与问题直接相关的事实清单"。
# 目的：强制模型通读全部条目、把所有相关成员/数值/方法逐项捞出，再让最终 answer
# 基于清单作答——避免"只看第一条就断言没有 / 漏掉分散在其他条目里的答案"。
SYSTEM_EXTRACT = (
    "你是论文证据提取器。会给你一篇论文的若干编号正文条目 [1..n] 与一个问题。\n"
    "任务：逐条通读**全部条目**，穷举式提取所有与问题**直接相关**的具体事实，"
    "例如：具体名称/成员/清单项、具体数值/百分比/得分/时长/差值、方法名、机制描述、"
    "数据来源与构建方式、实验设置等。\n"
    "规则：\n"
    "1. 宁可多列、不可漏列：只要某个片段能部分回答问题（给出任一成员、任一数值、任一名称），"
    "   就把它列为一条事实；不要因'答案好像已经够了'或'这段主题更像方法介绍'而跳过——"
    "   相关成员常常藏在主题不同的段落里（如实验对比段提到低资源语言对）。\n"
    "2. 事实必须**原样保留数字、专名与术语**（如 74.34、EMPATHETICDIALOGUES、"
    "   dialogue simulator），禁止概括成'若干数值/相关方法/某种来源'。\n"
    "3. 每条事实标注它出自的条目编号 n（可能多条出自同一 n）。\n"
    "4. 列举题（问'有哪些X/哪几个X/什么X'且问题中带数量，如 6 个语言对）：必须把散落在"
    "   **不同条目**中的全部成员都捞出来，并尽量让不同成员的数量与问题所示数量一致；"
    "   不得只从一个最相关的条目里取几个就收手。\n"
    "5. 只做提取，不要作答、不要推理、不要下'是否足以回答'的结论。\n"
    '输出为 JSON：{"facts": [{"n": 1, "text": "含原样数字/名称的事实"}, ...]}；'
    "若没有任何相关事实，facts 输出空数组。只输出 JSON，不要其他文字。"
)

SYSTEM_UNKNOWN = (
    "你是严谨的论文问答助手。面对一个可能无法百分之百确定的问题，按以下顺序处理：\n"
    "1. 若给出的原文能支持一个明确的结论（包括是/否、数值、做法有无），请**直接给结论**"
    "并标注对应条目 [n]，不要因为'想更全面'而含糊。\n"
    "2. 若确实无法给出确定结论：**不要只说'找不到/无法回答'**——请把检索到的最相关原文片段"
    "逐条提炼/转述给用户（保留关键数字、专名与措辞，标注 [n]），让用户能直接看到论文怎么说、"
    "自行判断；同时在结尾用一句话诚实说明'缺哪类信息导致无法下定论'。\n"
    "3. 禁止编造：提炼必须来自上方条目。用中文回答。"
)

CITE_RE = re.compile(r"\[(\d{1,2})\]")

# ── 缺失断言复核闸门 ──────────────────────────────────────────────────────────
# 背景（QA_V2_NEW10_20260906_REPORT.md F1）：浅层(L0/L1/L2) answer 证据不足时，
# 模型常断言"论文没写/未给出/未说明"，而数值/原因其实在 Abstract、正文表或 Discussion
# 里——浅层上下文看不到就误判全文缺失，且不触发下钻。
# 闸门：浅层答案若带"缺失口吻"，强制做一次 L3 独立全文复核；确有可答内容 → 以 L3 重答；
# 确认全文也没有 → 保留原答案（此刻"缺失结论"才算站得住）。L3 终答本身不触发。
AUDIT_TOP_K = 10
_ABSENCE_RE = re.compile(
    r"(未给出|未提供|未提及|未列出|并未说明|并未给出|没有给出|没有提供|找不到|"
    r"未找到|没有找到|无法找到|无法判断|无法确认|信息不足|不包含[^，。；]{0,10}信息|"
    r"未(?:明确|直接|具体|详细)?(?:说明|给出|提供|提及|指出|解释|报告)|"
    r"没有(?:明确|直接|具体)?(?:说明|给出|提供|提及|解释|报告)|"
    r"(?:论文|文中|文献|原文|文章)[^，。；]{0,16}(?:没有|未|不曾|并未)"
    r"(?:明确|直接|具体)?(?:说明|给出|提供|提及|指出|解释|给出过|存在|报告)|"
    r"(?:论文|文中|文献|原文|文章)[^，。；]{0,8}(?:未|没有|并未)[^，。；]{0,12}"
    r"(?:给出|提供|报告|说明))",
    re.I)


# ── 条目格式化（claim 样式 / chunk 样式统一编号）─────────────────────────────


def _fmt_claim_entry(i: int, e: dict[str, Any]) -> str:
    sec = (e.get("sections") or e.get("title_path") or [""])
    sec = sec[0] if isinstance(sec, list) and sec else ""
    return (
        f"[{i}] 主张[{e.get('label','')}]（{e.get('importance',0)}分，{sec}，"
        f"p{e.get('page','?')}）：{e.get('text','')}\n"
        f"    原文证据：{e.get('evidence','')}"
    )


def _fmt_chunk_entry(i: int, c: dict[str, Any]) -> str:
    return (f"[{i}] 正文段落（p{c.get('page','?')} {c.get('section','')}）："
            f"{c.get('text','')}")


# ── cites 解析 ────────────────────────────────────────────────────────────────


def _cites_from(text: str, entries: list[dict[str, Any]], pdf: str) -> list[dict[str, Any]]:
    stem = pdf.rsplit(".", 1)[0]
    cites: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for n in sorted({int(m) for m in CITE_RE.findall(text)}):
        if not (1 <= n <= len(entries)):
            continue
        e = entries[n - 1]
        if e["kind"] == "claim":
            key = ("g", e.get("gid", ""))
            if key in seen:
                continue
            seen.add(key)
            cites.append({
                "ref": f"{stem}#{e.get('gid','')}",
                "gid": e.get("gid", ""),
                "claim_id": e.get("rep_claim_id", ""),
                "evidence": e.get("evidence", ""),
                "page": e.get("page", 0),
            })
        else:  # chunk
            key = ("c", e.get("chunk_id", ""))
            if key in seen:
                continue
            seen.add(key)
            cites.append({
                "ref": f"{stem}#{e.get('chunk_id','')}",
                "gid": "",
                "claim_id": "",
                "evidence": e.get("text", ""),
                "page": e.get("page", 0),
            })
    return cites


# ── 各档证据取料 ──────────────────────────────────────────────────────────────


def _claim_entries(items: list[Any]) -> list[dict[str, Any]]:
    """retrieved claims / core_points 都转成 claim 样式条目。"""
    out = []
    for it in items:
        entry = {**it, "kind": "claim"}
        entry.setdefault("sections", [])
        out.append(entry)
    return out


def _chunk_entries(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        out.append({**it, "kind": "chunk"})
    return out


def _fmt_history(state: QAState) -> str:
    """把之前轮次的对话格式化为"对话历史"块（追问时理解"这个方法/它"等指代）。"""
    hist = state.get("history") or []
    if not hist:
        return ""
    lines = ["", "对话历史（本轮问题可能指代其中的内容，回答时请结合理解）："]
    for m in hist[-6:]:
        who = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"  {who}: {(m.get('content') or '')[:500]}")
    return "\n".join(lines)


def _build_context(state: QAState) -> tuple[str, list[dict[str, Any]], str]:
    """返回 (导语/概述区, 编号条目列表, 档位)。"""
    overview = state.get("overview") or ""
    l3 = state.get("l3_chunks")
    if l3:
        entries = _chunk_entries(l3)
        header = f"标题：{state.get('title','')}\n\n=== 全文检索到的正文（独立检索） ==="
        return header, entries, "L3"
    chunks = state.get("chunks")
    if chunks:
        entries = _chunk_entries(chunks)
        header = f"论文概述：{overview or '（无）'}\n\n=== 正文窗口 ==="
        return header, entries, "L2"
    retrieved = state.get("retrieved")
    if retrieved:
        entries = _claim_entries(retrieved)
        header = f"论文概述：{overview or '（无）'}\n\n=== 检索到的相关主张 ==="
        return header, entries, "L1"
    entries = _claim_entries(state.get("core_points") or [])
    header = f"论文概述：{overview or '（无）'}\n\n=== 核心要点 ==="
    return header, entries, "L0"


def _extract_facts(question: str, context: str) -> list[dict[str, Any]]:
    """两步法第一步：通读全部条目，穷举与问题相关的事实清单。

    返回形如 [{"n": 2, "text": "..."}] 的列表；解析/调用失败时返回 []。
    失败不阻断主流程（降级为单步 answer，仍带防自毁准则）。
    """
    user = (f"{context}\n\n用户问题：{question}\n\n"
            "请逐条通读上方全部条目，穷举提取与问题直接相关的事实，并只输出指定 JSON。")
    try:
        obj = llm.chat_json(SYSTEM_EXTRACT, user, temperature=0.0)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(obj, dict):
        return []
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return []
    out = []
    for f in facts:
        if isinstance(f, dict) and f.get("text"):
            out.append({"n": int(f.get("n") or 0), "text": str(f["text"])})
    return out


def _fmt_facts_block(facts: list[dict[str, Any]]) -> str:
    lines = [
        f"- 条目[{f['n']}]：{f['text']}" if f["n"] else f"- {f['text']}"
        for f in facts
    ]
    return ("\n\n第 1 步穷举出的候选事实清单（供你参考的'可能有关'线索，**不是必须全用**）：\n"
            + ("\n".join(lines) if lines else "（第 1 步未提取到任何候选事实）")
            + "\n\n使用规则：\n"
            "  a) 判断哪些候选事实**直接回答用户问题**，把它们组织进答案并标注对应 [n]。\n"
            "  b) 清单里可能混有**背景/对比/反向方向**等相邻信息（如训练数据的反向语言对、"
            "     与其他基线对比的数值）——这类只在你认为它们能帮助回答时才选用，"
            "     不要为了'覆盖清单'而把它们当作答案主体。\n"
            "  c) 若清单遗漏了直接答案，可再回看上方条目补充并标注 [n]；但禁止因"
            "     '清单之外没有别的'就断言信息不足。")


# ── 节点：generate_answer ─────────────────────────────────────────────────────


def _sec_tail(paths: list[str]) -> str:
    """chunk title_path → 展示节名（与 pull_chunk.search_l3 同款取法）。"""
    for p in reversed(paths or []):
        if " · " in p:
            return p.split(" · ", 1)[1].strip()
    return (paths[-1] if paths else "")


def _compose(question: str, pdf: str, header: str,
             entries: list[dict[str, Any]], level: str,
             state: QAState) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """给定上下文条目生成 answer + cites + facts（无 route/debug 副作用）。"""
    parts = [header]
    fmt = _fmt_claim_entry if level in ("L1", "L0") else _fmt_chunk_entry
    for i, e in enumerate(entries, 1):
        parts.append(fmt(i, e))
    if not entries:
        parts.append("（未检索到相关条目）")
    context = "\n\n".join(parts)

    # 两步法：L2/L3（chunk 上下文长、答案分散）先穷举相关事实清单，再带清单作答。
    facts: list[dict[str, Any]] = []
    if level in ("L2", "L3") and entries:
        facts = _extract_facts(question, context)
    facts_block = _fmt_facts_block(facts)

    user = (f"{context}{facts_block}\n\n{_fmt_history(state)}\n\n用户问题：{question}\n\n"
            f"（若信息不足以回答，请说明缺什么，不要编造。）")
    answer = llm.chat_text(SYSTEM, user, temperature=0.0)
    cites = _cites_from(answer, entries, pdf)
    return answer, cites, facts


def _l3_audit_entries(pdf: str, question: str) -> list[dict[str, Any]]:
    """缺失断言复核：独立全文检索（向量+BM25 融合，提升表格/数值段召回）。"""
    try:
        hits = ChunkIndex(pdf).search_hybrid(question, top_k=AUDIT_TOP_K)
    except Exception:  # noqa: BLE001（复核失败不阻断，按"未找到"处理）
        return []
    out: list[dict[str, Any]] = []
    for h in hits:
        out.append({
            "kind": "chunk",
            "chunk_id": h.get("chunk_id", ""),
            "page": h.get("page", 0),
            "section": _sec_tail(list(h.get("title_path") or [])),
            "text": h.get("text", ""),
        })
    return out


def _audit_absence(question: str, pdf: str) -> dict[str, Any]:
    """复核一条浅层"缺失断言"：L3 全文若能作答 → found=True，由调用方重答。"""
    entries = _l3_audit_entries(pdf, question)
    if not entries:
        return {"found": False, "entries": [], "reason": "无全文检索命中"}
    try:
        v = judge_l3({"question": question, "l3_chunks": entries})
    except Exception:  # noqa: BLE001
        v = {"verdict": {"enough": False}}
    enough = bool((v.get("verdict") or {}).get("enough", False))
    return {"found": enough, "entries": entries,
            "reason": "" if enough else "全文复核判定仍不足"}


def generate_answer(state: QAState) -> dict[str, Any]:
    question = state.get("question", "")
    pdf = state.get("pdf", "")
    header, entries, level = _build_context(state)
    answer, cites, facts = _compose(question, pdf, header, entries, level, state)

    # 缺失断言复核闸门：浅层答案若带"论文没写/未给出/找不到"口吻，
    # 强制 L3 全文复核（修正"证据没送到就断言全文没有"的误判）。
    audit: dict[str, Any] | None = None
    if level in ("L0", "L1", "L2") and _ABSENCE_RE.search(answer or ""):
        audit = _audit_absence(question, pdf)
        if audit["found"]:
            l3_entries = audit["entries"]
            header3 = (f"标题：{state.get('title','') or ''}\n\n"
                       f"=== 全文检索正文（缺失断言复核） ===")
            answer, cites, facts = _compose(question, pdf, header3,
                                            l3_entries, "L3", state)
            level = "L3"
            entries = l3_entries

    verdict = state.get("verdict") or {}
    route = [x for x in (state.get("route") or []) if not x.startswith("answer_")]
    if level == "L3" and "L3" not in route and audit and audit["found"]:
        route.append("L3")
    if f"answer_{level}" not in route:
        route.append(f"answer_{level}")
    debug = dict(state.get("debug") or {})
    debug["answer"] = {
        "level": level,
        "n_entries": len(entries),
        "n_facts": len(facts),
        "facts": [f"{f.get('n')}:{(f.get('text') or '')[:80]}" for f in facts][:12],
        "enough": verdict.get("enough"),
        "gap": (verdict.get("gap") or "")[:120],
    }
    if audit is not None:
        debug["absence_audit"] = {
            "triggered": True,
            "found": audit["found"],
            "dove_l3": audit["found"],
            "n_l3": len(audit["entries"]),
            "reason": audit["reason"],
        }
    return {
        "answer": answer,
        "cites": cites,
        "route": route,
        "debug": debug,
    }


# ── 节点：answer_unknown（诚实收尾）─────────────────────────────────────────


def answer_unknown(state: QAState) -> dict[str, Any]:
    """诚实收尾（L3 仍不足）。2026-09-07：不再只说"找不到"——
    能明确就给结论；确实无法确定时，把检索到的相关原文提炼给用户自行判断。"""
    question = state.get("question", "")
    pdf = state.get("pdf", "")
    gap = (state.get("verdict") or {}).get("gap", "")
    entries: list[dict[str, Any]] = []
    for c in (state.get("l3_chunks") or [])[:6]:
        entries.append({**c, "kind": "chunk"})

    if entries:
        parts = ["=== 全文检索到的相关原文（供你提炼给用户） ==="]
        for i, e in enumerate(entries, 1):
            parts.append(_fmt_chunk_entry(i, e))
        context = "\n\n".join(parts)
        user = (f"{context}\n\n用户问题：{question}\n\n"
                f"缺口信息（为什么下不了定论）：{gap or '未能找到决定性证据'}\n\n"
                f"请按系统准则处理：能明确就明确（标[n]）；不能明确就把最相关片段提炼给用户（标[n]），"
                f"并说明缺什么。")
    else:
        user = (f"用户问题：{question}\n\n缺口信息：{gap or '未知'}\n\n"
                f"检索没有任何相关原文，请诚实说明并指出缺哪类信息。")
    answer = llm.chat_text(SYSTEM_UNKNOWN, user, temperature=0.0)
    route = list(state.get("route") or [])
    if "answer_unknown" not in route:
        route.append("answer_unknown")
    cites = _cites_from(answer, entries, pdf) if entries else []
    debug = dict(state.get("debug") or {})
    debug["answer"] = {"level": "unknown", "gap": gap[:120], "n_evidence": len(entries)}
    return {
        "answer": answer,
        "cites": cites,
        "route": route,
        "debug": debug,
    }
