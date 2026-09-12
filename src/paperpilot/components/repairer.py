"""答案修复器：输出闸门（validator.gate）检出 high 问题后的对症重试。

挂在 gate(repair=...) 上，一次调用做"判定失败类型 → 对症重试 → 复检"，
仍不过返回 None → gate 走兜底（拒答话术，不用 answer_unknown 路径）。

两条修复路径（业界 Self-Refine / CRAG 的最小落地）：
  ① Self-Refine（generation 型问题：off_topic / contradiction / vague）：
     把质检意见注入，让 LLM 基于**已给引用原文**只重写问题句（不引入新来源，单次 LLM）。
  ② CRAG（evidence 型问题：编数 / 引用越界 / unsupported 句）：
     从问题句/无据句构造"缺口定向查询"，走多查询×向量+BM25 混合检索（topK 加大），
     拿回的新证据作 L3 上下文重新生成（复用主链 compose 两步法）。
复检：重试结果再过一次 validator.check（机器+LLM 同口径），仍 high → None（触发兜底）。

设计约束：
  - 不无限循环：每次 gate 至多一次 repair 调用；repair 内每型至多试一次。
  - 新答案必须仍走 [n] 引用（cites 由 _cites_from 与对应 entries 重新对齐）。
  - 任何一步异常 → None（让 gate 按原逻辑兜底），绝不抛断输出。
"""
from __future__ import annotations

import re
from typing import Any

from paperpilot.tools import llm

_EVIDENCE_TYPES = {"number", "citation", "unsupported"}
_GENERATION_TYPES = {"off_topic", "contradiction", "vague"}

# 重试时引用片段最多取前 N 条、每条截断，控上下文
_MAX_EV = 10
_EV_CAP = 600


def _entries_from_cites(cites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """cites（答案实际引用，每条带 evidence 原文）→ gate 风格 chunk 条目。"""
    out: list[dict[str, Any]] = []
    for c in cites or []:
        ev = str(c.get("evidence") or "")
        if ev:
            out.append({"kind": "chunk", "text": ev,
                        "page": int(c.get("page") or 0),
                        "chunk_id": str(c.get("ref") or "")})
    return out


def _fmt_entries(entries: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i+1}] {str(e.get('text') or '')[: _EV_CAP]}"
                     for i, e in enumerate(entries[:_MAX_EV])) or "（无）"


# ── ① Self-Refine ──────────────────────────────────────────────────────────

_SYS_REWRITE = (
    "你是论文问答的改写员。系统初稿被质检标出问题。你的任务：基于【问题】【被引用原文片段】"
    "与【质检意见】，重写答案——只保留/修正有原文支撑的表述；删除或改写无支撑/偏题/含糊/"
    "自相矛盾的部分（自相矛盾处一律以原文为准）；可以补充被你遗漏、但**原文片段里明显有据**"
    "的要点。引用编号 [n] 必须真实存在于下方原文片段（编号超出或悬空会判失败）。"
    "禁止引入原文片段里没有的内容。直接输出重写后的完整答案文本，不要任何解释。"
)

_REFINE_TPL = """【问题】
{question}

【质检意见】
{issues}

【初稿答案】
{answer}

【被引用原文片段（[n] 对应引用编号；只是检索到的子集，非全文）】
{evidence}

请输出重写后的完整答案："""


def _refine_answer(question: str, answer: str, issues: list[dict[str, Any]],
                   cites: list[dict[str, Any]]) -> str | None:
    entries = _entries_from_cites(cites)
    if not entries:
        return None
    iss = [i for i in issues if i["sev"] in ("high", "mid") and i.get("detail")]
    lines = "\n".join(f"- [{i.get('type')}] {str(i.get('sentence', ''))[:120]}"
                      f" :: {str(i.get('detail', ''))[:200]}"
                      for i in iss[:8]) or "（无明确意见——请按'只依据原文重写'原则自查）"
    user = _REFINE_TPL.format(question=question, issues=lines,
                              answer=answer[:2500], evidence=_fmt_entries(entries))
    try:
        out = llm.chat_text(_SYS_REWRITE, user, temperature=0.0)
    except llm.LLMError:
        return None
    s = str(out or "").strip()
    return s if s and s != answer else None


# ── ② CRAG（缺口定向重检索） ────────────────────────────────────────────────

_GAP_SENT_RE = re.compile(r"\[\d+\]")


def _crag_regenerate(question: str, pdf: str, answer: str,
                     issues: list[dict[str, Any]], top_k: int = 16
                     ) -> tuple[str, list[dict[str, Any]]] | None:
    """对 evidence 型失败：用问题/无据句做多查询混合检索，新证据上 L3 重答。"""
    from paperpilot.agents.embedder import ChunkIndex
    from paperpilot.agents.nodes.pull_chunk import _section_tail
    from paperpilot.components.context_builder import compose

    gaps = []
    for i in issues:
        s = str(i.get("sentence") or "").strip()
        if i["type"] in _EVIDENCE_TYPES and len(s) > 6:
            q = _GAP_SENT_RE.sub("", s).strip()
            q = re.sub(r"\s+", " ", q)[:160]
            if q and q.lower() != question.lower() and q not in gaps:
                gaps.append(q)
    queries = ([question] + gaps)[:3]
    try:
        idx = ChunkIndex(pdf)
        hits = (idx.search_multi_hybrid(queries, top_k=top_k) if len(queries) > 1
                else idx.search_hybrid(question, top_k=top_k))
    except Exception:  # noqa: BLE001
        return None
    l3: list[dict[str, Any]] = []
    for h in hits:
        txt = h.get("text")
        if not txt:
            continue
        l3.append({"chunk_id": str(h.get("chunk_id", "")),
                   "page": int(h.get("page") or 0),
                   "section": _section_tail(list(h.get("title_path") or [])),
                   "text": txt})
    if not l3:
        return None
    entries = [{**c, "kind": "chunk"} for c in l3]
    header = f"=== 全文检索正文（修复回路·缺口重检索，{len(l3)} 块）==="
    try:
        new, cites, _facts = compose(question, pdf, header, entries, "L3", {})
    except Exception:  # noqa: BLE001
        return None
    new = str(new or "").strip()
    if not new or new == answer:
        return None
    return new, list(cites or []), entries


# ── 复检 ─────────────────────────────────────────────────────────────────────


def _recheck_entries(question: str, answer: str, entries: list[dict[str, Any]]) -> bool:
    """重试结果再过一次机器+LLM 检查；high 仍存在 → 修复无效。

    entries 必须是 answer 引用编号 [n] 对应的**完整上下文条目集**
    （refine → 被引 cites 子集重编号；crag → 新检索全条目），否则会误报越界。
    """
    from paperpilot.components import validator
    try:
        res = validator.check(question, answer, entries,
                              n_entries=len(entries))
    except Exception:  # noqa: BLE001
        return False
    return not bool(res.get("high"))


def _try_refine(question: str, pdf: str, answer: str,
                issues: list[dict[str, Any]], cites: list[dict[str, Any]]
                ) -> tuple[str, list[dict[str, Any]]] | None:
    from paperpilot.agents.nodes.answer import _cites_from
    new = _refine_answer(question, answer, issues, cites)
    if not new:
        return None
    entries = _entries_from_cites(cites)   # refine 模型按 cites 子集编号作答 → 用同集复检
    new_cites = list(_cites_from(new, entries, pdf) or [])
    if not new_cites or not _recheck_entries(question, new, entries):
        return None
    return new, new_cites


def _try_crag(question: str, pdf: str, answer: str,
              issues: list[dict[str, Any]], cites: list[dict[str, Any]]
              ) -> tuple[str, list[dict[str, Any]]] | None:
    got = _crag_regenerate(question, pdf, answer, issues)
    if not got:
        return None
    new, new_cites, full_entries = got
    if not new_cites or not _recheck_entries(question, new, full_entries):
        return None
    return new, new_cites


# ── ③ 补充复核（2026-09-09）────────────────────────────────────────────────
# 触发：机器数字层发现"答案某数在被引证据缺失、但全篇某 chunk 找到"——不静默放行、
# 也不判对错，把命中块作为 Supplementary Context 喂回 Generator，由它判断初稿
# 是否需要修改（如补充块的 7.5 是章节号而非天数 → LLM 识别无关，维持原答案或删断言）。
# 与 CRAG 区别：CRAG 是"证据缺失需重检索"（真编数）；这里证据已定位到具体 chunk。

_SYS_SUPPLEMENT = (
    "你是严谨的论文问答复核员。质检发现初稿中若干数字在被引用片段中找不到依据，"
    "只在论文其他位置的【补充证据】里出现同名数字。下方【待核项】已给出机器定位："
    "哪个数字、出自初稿哪句、在补充证据哪个条目出现。\n"
    "请对每个待核项裁决其数字断言是否成立，输出最终完整答案：\n"
    "1. 若补充证据条目与初稿断言是**同一语义语境**（如都是训练数据量/同一指标）→ "
    "断言成立：保留，并把该句引用改为指向支撑它的条目 [n]。\n"
    "2. 若补充证据只是**同名不同义**（如初稿说'耗时 7.5 天'，补充条目是'章节 7.5'/"
    "别处另一个指标）→ 该条目不支撑断言；且被引片段本就没有该数 → **必须删除**该"
    "数字断言，或改写为'论文中无法确认这一点'，禁止让无据数字留在最终答案（宁缺毋编）。\n"
    "3. 只能依据下方条目改写；未受影响的其余内容与引用保持原样，不得新增条目外内容。\n"
    "4. 若某待核项其实能由被引片段直接支撑 → 保留并保持其引用即可。\n"
    "直接输出最终答案文本，不要任何解释。"
)

_SUPP_TPL = """【问题】
{question}

【初稿答案】
{answer}

【待核项】（机器已定位：数字 → 初稿子句 → 补充证据条目）
{items}

【完整条目（[n] 对应下方编号；1..{k} 为初稿被引用片段，其余为补充证据）】
{entries}

请逐项裁决后输出最终答案："""


_SUB_SENT_RE = re.compile(r"(?<=[，。！？；;!?\n])\s*|(?<=[，。！？；;!?\n])")
_NUM_STRIP_RE = re.compile(r"\[\d+\]")


def _num_subclauses(answer: str, num: float) -> list[str]:
    """从初稿中摘出含指定归一数字的**子句**（逗号/句号级，聚焦单个数字断言）。"""
    from paperpilot.components.validator import _canonical_nums
    out = []
    for part in _SUB_SENT_RE.split(answer or ""):
        s = part.strip().strip("，。；")
        if not s or len(s) < 4:
            continue
        if num in _canonical_nums(_NUM_STRIP_RE.sub("", s)):
            out.append(s)
        if len(out) >= 8:
            break
    return out


def _supplement_entries(cites: list[dict[str, Any]],
                        supplements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并初稿被引条目 + 补充证据 chunk 条目（按 chunk_id 去重，补引在后）。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in cites or []:
        ev = str(c.get("evidence") or "")
        cid = str(c.get("chunk_id") or c.get("ref") or "")
        if not ev:
            continue
        key = cid or ev[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": "chunk", "chunk_id": cid, "page": int(c.get("page") or 0),
                    "text": ev})
    for s in supplements or []:
        for ch in (s.get("chunks") or [])[:3]:
            txt = str(ch.get("text") or "") if isinstance(ch, dict) else str(getattr(ch, "text", ""))
            cid = (ch.get("chunk_id") or "") if isinstance(ch, dict) else str(getattr(ch, "chunk_id", "") or "")
            page = ch.get("page") if isinstance(ch, dict) else getattr(ch, "page", 0)
            if not txt:
                continue
            key = cid or txt[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": "chunk", "chunk_id": cid, "page": int(page or 0), "text": txt})
    return out


def supplement(question: str, pdf: str, answer: str,
               supplements: list[dict[str, Any]],
               cites: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]] | None:
    """补充复核：数字在被引证据缺失、但已在全篇定位到命中 chunk 时调用。

    把命中块与初稿一并给 Generator，由其判断初稿是否需改（识别"此数非彼数"）。
    Returns: (new_answer, new_cites)；Generator 判定无需修改 → None（gate 放行原答案）。
    """
    entries = _supplement_entries(cites, supplements)
    if not entries:
        return None
    # 被引片段计数（后续条目才是补充证据；用于模板提示）
    k = sum(1 for c in cites or [] if str(c.get("evidence") or "").strip())
    # 待核项：机器已定位 数字 → 初稿子句 → 该数所在补充条目编号
    # （条目 1..k 是初稿被引片段——这些片段里本就找不到该数，重点看补充条目 k+1..n）
    entry_by_cid: dict[str, int] = {}
    for i, e in enumerate(entries, 1):
        if e.get("chunk_id"):
            entry_by_cid[str(e["chunk_id"])] = i
    items: list[str] = []
    for s in supplements:
        num_txt = str(s.get("num_text") or s.get("num") or "")
        for sub in _num_subclauses(answer, float(s.get("num") or 0)):
            hit_ids = []
            for ch in (s.get("chunks") or [])[:3]:
                cid = ch.get("chunk_id") if isinstance(ch, dict) else getattr(ch, "chunk_id", "")
                idx = entry_by_cid.get(str(cid or ""))
                if idx is not None:
                    hit_ids.append(f"[{idx}]")
            items.append(f"- 数字 {num_txt}｜初稿句：{sub[:140]}｜"
                         f"补充证据条目：{','.join(hit_ids) or '（未并入条目）'}")
    if not items:
        return None
    user = _SUPP_TPL.format(
        question=question, answer=answer[:2500],
        items="\n".join(items),
        k=k, entries=_fmt_entries(entries))
    try:
        new = llm.chat_text(_SYS_SUPPLEMENT, user, temperature=0.0)
    except llm.LLMError:
        return None
    new = str(new or "").strip()
    if not new or new == answer:
        return None  # Generator 判定无需修改
    # 重对齐引用 + 机器复检（数字须落回被引或补充块，防复核时引入新编数）
    from paperpilot.agents.nodes.answer import _cites_from
    from paperpilot.components import validator
    try:
        new_cites = list(_cites_from(new, entries, pdf) or [])
        res = validator.check(question, new, entries, n_entries=len(entries), use_llm=False)
    except Exception:  # noqa: BLE001
        return None
    if res.get("high"):
        return None
    return new, new_cites


# ── ④ 数值裁决（2026-09-11）──────────────────────────────────────────────────
# 触发：机器数字层发现"答案某数在被引证据与**全篇**都找不到"。
# 关键认知：**"找不到" ≠ "编造"** —— parser 覆盖不全（LaTeX 残留 / 单位词界 /
# 年份剥离）、派生运算、写法差异都会导致找不到（2026-09-11 实测 18/18 全是误杀）。
# 因此机器层只出**信号**，最终裁决交给 LLM 在真实上下文里做 —— 机器不再单方面判死。

_SYS_ADJUDICATE = (
    "你是论文问答的**数值复核员**。初稿里有几个显著数字，在被引用的原文片段中"
    "**没有直接匹配到**。请判断：这些数字是否**真的没有依据**。\n"
    "【重要】原文常用**不同写法**表达同一数值，遇到下列任一情况都判为**有依据**：\n"
    "  - LaTeX 残留：`$8.5 \\times 10^{11}$`、`$7.30 * 10^{115}$`、`54$\\%$`\n"
    "  - 千分位 / 欧式小数：`15,000`、`7,5`\n"
    "  - 科学计数：`7.3e115`、`3.6×10^4`\n"
    "  - 单位后缀与中文数词：`36 million`、`1.5b`、`3.6w`、`2亿`、`三万六千`\n"
    "  - 由片段中已有数字**经四则运算**得出的结果（初稿若已写出算式与操作数，"
    "如 `2000−300=1700`、`293÷4262≈0.069`，视为合规）\n"
    "只有当**确实找不到该数值的任何写法、也推不出来**时，才判 supported=false"
    "（此时该数字属编造，必须拦下）。\n"
    '只输出 JSON：{"supported": true/false, "reason": "一句话理由"}。'
)

_ADJ_TPL = """【问题】
{question}

【待核数字】（机器在被引片段与全文中都没匹配到）
{items}

【被引用的原文片段（[1..{k}] 对应初稿引用编号；只是检索到的子集，非全文）】
{evidence}

请逐项判断这些数字是否被上述片段支持："""


def adjudicate_numbers(question: str, answer: str, nums: list[float],
                       cites: list[dict[str, Any]]) -> tuple[bool, str]:
    """数值裁决：机器"全篇找不到"的显著数字 → LLM 在真实上下文里判是否支持。

    Returns: (supported, reason)。复核失败/异常 → (False, 原因)，保守（宁缺毋编）。
    """
    from paperpilot.components.validator import _fmt_num

    entries = _entries_from_cites(cites)
    items: list[str] = []
    for v in list(nums)[:6]:
        subs = _num_subclauses(answer, float(v))
        line = f"- {_fmt_num(float(v))}"
        if subs:
            line += f"｜初稿句：{subs[0][:120]}"
        items.append(line)
    if not items:
        return False, "无可核查子句"
    user = _ADJ_TPL.format(question=question, items="\n".join(items),
                           k=len(entries), evidence=_fmt_entries(entries))
    try:
        raw = llm.judge_json(_SYS_ADJUDICATE, user, temperature=0.0)
    except (llm.LLMError, ValueError) as e:
        return False, f"复核调用失败: {type(e).__name__}"
    if not isinstance(raw, dict):
        return False, "复核输出无法解析"
    return bool(raw.get("supported")), str(raw.get("reason") or "")[:200]


# ── 主入口 ───────────────────────────────────────────────────────────────────


def repair(question: str, pdf: str, answer: str,
           issues: list[dict[str, Any]], cites: list[dict[str, Any]]
           ) -> tuple[str, list[dict[str, Any]]] | None:
    """输出闸门检出问题后的一次性对症修复。

    Returns: (new_answer, new_cites)；修不动/无对应类型问题 → None（gate 走兜底拒答）。
    """
    tgt = [i for i in issues if i["sev"] in ("high", "mid") and i.get("type") != "missing"]
    if not tgt or not (answer or "").strip():
        return None
    has_ev = any(i["type"] in _EVIDENCE_TYPES for i in tgt)
    has_ge = any(i["type"] in _GENERATION_TYPES for i in tgt)
    # 生成问题（证据在没用对）先 Self-Refine（便宜）；仍有 evidence 型失败再做 CRAG。
    if has_ge:
        got = _try_refine(question, pdf, answer, tgt, cites)
        if got:
            return got
    if has_ev:
        got = _try_crag(question, pdf, answer, tgt, cites)
        if got:
            return got
    return None
