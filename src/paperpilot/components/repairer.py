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
    """cites（答案实际引用，每条带 evidence 原文）→ gate 风格 chunk 条目。

    ⚠️ **必须把「第几篇 / 文件名 / 裸 chunk_id」分开存**（2026-10-03 修，评审指出）：

    `chunk_id` **只在篇内唯一** —— 每篇论文的块都从 `c1` 开始。所以任何**只按 cid**
    的匹配都会串篇。旧实现把 `ref`（形如 `"<stem>#<chunk_id>"`）塞进 `chunk_id`，
    而且**完全不产出 `pdf`**，后果有两个（都实测坐实过）：

      ① `_cites_from` 的**锚点分支永远匹配不上**（`"2408.09273#c9"` ≠ 锚点里的 `c9`）
         → fullctx 答案一旦被修复，引用被**清空**（"修好了"其实是没修上）；
      ② `_cites_from` 里 `e.get("pdf")` 恒空 → 回退到调用方传的 `pdfs[0]` →
         **多篇下所有 cite 都被标成第 1 篇**（比"没有 pdf"更糟：给了个错的）。

    所以这里拆成三件：`p`（第几篇，1-based，0 = 未知）/ `pdf`（文件名）/
    `chunk_id`（**裸 cid**）。多篇下的消歧全靠 `(p, chunk_id)` 这一对。
    """
    out: list[dict[str, Any]] = []
    for c in cites or []:
        ev = str(c.get("evidence") or "")
        if not ev:
            continue
        ref = str(c.get("ref") or "")
        cid = str(c.get("chunk_id") or "")
        pdf = str(c.get("pdf") or "")
        if "#" in ref:                      # `ref = "<pdf(stem)>#<chunk_id>"`
            head, _, tail = ref.partition("#")
            cid = cid or tail
            pdf = pdf or head
        out.append({"kind": "chunk", "text": ev,
                    "page": int(c.get("page") or 0),
                    "chunk_id": cid, "pdf": pdf,
                    "p": int(c.get("p") or 0),
                    "ref": ref})
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


def _crag_regenerate(question: str, pdfs: list[str], answer: str,
                     issues: list[dict[str, Any]], top_k: int = 16
                     ) -> tuple[str, list[dict[str, Any]]] | None:
    """对 evidence 型失败：用问题/无据句做多查询混合检索，新证据上 L3 重答。

    ⚠️ 参数是**语料（多篇）**：检索走 `MultiChunkIndex`；`pdfs[0]` 只作 cites 的兜底标签
    （每条证据自带 `pdf`，cites 不会串篇）。本系统没有单篇路径（2026-09-23）。
    """
    from paperpilot.agents.embedder import MultiChunkIndex
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
        idx = MultiChunkIndex([str(p) for p in pdfs if p])
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
        new, cites, _facts = compose(question, pdfs[0] if pdfs else "",
                                     header, entries, "L3", {})
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


def _try_refine(question: str, pdfs: list[str], answer: str,
                issues: list[dict[str, Any]], cites: list[dict[str, Any]]
                ) -> tuple[str, list[dict[str, Any]]] | None:
    from paperpilot.agents.nodes.answer import _cites_from
    new = _refine_answer(question, answer, issues, cites)
    if not new:
        return None
    entries = _entries_from_cites(cites)   # refine 模型按 cites 子集编号作答 → 用同集复检
    new_cites = list(_cites_from(new, entries, pdfs[0] if pdfs else "") or [])
    if not new_cites or not _recheck_entries(question, new, entries):
        return None
    return new, new_cites


def _try_crag(question: str, pdfs: list[str], answer: str,
              issues: list[dict[str, Any]], cites: list[dict[str, Any]]
              ) -> tuple[str, list[dict[str, Any]]] | None:
    got = _crag_regenerate(question, pdfs, answer, issues)
    if not got:
        return None
    new, new_cites, full_entries = got
    if not new_cites or not _recheck_entries(question, new, full_entries):
        return None
    return new, new_cites


# ── ①′ fullctx 专属修复：**同一份全上下文上再答一次**（2026-10-03）────────────
#
# 为什么不能沿用 CRAG（实测踩到）：
#   fullctx 的提示词里**本来就有全部语料** —— "缺口证据"不用去检索，它已经在上下文里。
#   CRAG 却去 `MultiChunkIndex.search_multi_hybrid(top_k=16)` **重新全局取块**（无篇配额），
#   于是：① 字面命中多的那一篇占满 16 个名额 → 跨篇问题被修成"只讲第 1 篇"
#   （实测修复后 5 条 cites 全来自 `2408.09273.pdf`）；② 答案改写成 `[n]` 编号 →
#   **整套 `[P…·§…·¶…]` 锚点没了**，前端"点引用跳原文"跟着失效。
#
# 本路径的做法就是"**把初稿 + 质检意见塞回提示词，第二遍生成**"：
#   · 语料一字未动 → 覆盖不丢；编号体系未换 → 锚点仍是 `[P…]`；
#   · 成本可控：`sys + ctx` 是**稳定前缀**，第二遍**命中 prompt 前缀缓存**
#     （这正是 fullctx 能成立的设计前提，见其模块头第 1 条）。
#
# 为何检索路径（RAG-2）仍保留 CRAG：那条路径的上下文**只有 top-k 个块**，
# 缺口证据确实不在里面 → 必须重检索。两条路径的前提不同，所以修法不同。

_REASK_TPL = """【你的初稿（未通过内部事实校验）】
{draft}

【校验发现的问题】
{issues}

【修订要求】
1) 只依据**上文语料**修订：把没有原文支撑的断言**删除或改写**（宁缺毋编），其余内容保持原样；
2) **保留 `[P…·§…·¶…]` 形式的引用锚点**（编号照上文语料，不要自造）；需要换依据时也用它；
3) 某个断言在语料里确实找不到支撑 → 明确写成「语料中未提及」，不要含糊带过；
4) 直接输出**修订后的完整答案**，不要解释你改了什么。
"""


def _is_fullctx() -> bool:
    """当前问答读取器是不是 fullctx（决定修复走"同上下文再答"还是"重检索"）。"""
    try:
        from paperpilot.agents.nodes.read_full import reader
        return reader() == "fullctx"
    except Exception:  # noqa: BLE001  判不出来就当不是（退回旧路径，不冒险）
        return False


def _try_fullctx_reask(question: str, pdfs: list[str], answer: str,
                       issues: list[dict[str, Any]]
                       ) -> tuple[str, list[dict[str, Any]]] | None:
    """fullctx 修复：**同一份全上下文** + 初稿 + 意见 → 再生成（保留 `[P…]` 锚点）。"""
    from paperpilot.components import fullctx

    if not pdfs:
        return None
    iss = [i for i in issues if i["sev"] in ("high", "mid")]
    lines = "\n".join(
        f"- [{i.get('type')}] {str(i.get('sentence') or '')[:120]}"
        f" :: {str(i.get('detail') or '')[:200]}" for i in iss[:8]) or "（无明确意见——按'只依据原文'自查）"
    q2 = _REASK_TPL.format(draft=str(answer or "")[:2500], issues=lines)
    try:
        out = fullctx.answer([str(p) for p in pdfs], q2)
    except Exception:  # noqa: BLE001  调不通就交给下一条路径
        return None
    s = str(out.get("answer") or "").strip()
    new_cites = list(out.get("cites") or [])
    if not s or s == answer:
        return None
    # 复检：用**新版答案自己的**被引证据（允许它换依据），仍然 HIGH → 视为没修好
    if not new_cites or not _recheck_entries(question, s, _entries_from_cites(new_cites)):
        return None
    return s, new_cites


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


def supplement(question: str, pdfs: list[str], answer: str,
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
        new_cites = list(_cites_from(new, entries, pdfs[0] if pdfs else "") or [])
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


def repair(question: str, pdfs: list[str], answer: str,
           issues: list[dict[str, Any]], cites: list[dict[str, Any]]
           ) -> tuple[str, list[dict[str, Any]]] | None:
    """输出闸门检出问题后的一次性对症修复。

    **按读取器分流**（2026-10-03）：
      · **fullctx**（默认）→ `_try_fullctx_reask`：**同一份全上下文** + 初稿 + 质检意见
        再生成一次。语料不用重检索（本来就在提示词里）、覆盖不丢、`[P…]` 锚点不换，
        而且 `sys+ctx` 是稳定前缀 → 第二遍**命中 prompt 前缀缓存**，成本可控。
      · **retrieval**（RAG-2）→ 老路：Self-Refine（生成型）→ CRAG（证据型）。
        那条路径的上下文**只有 top-k 个块**，缺口证据确实不在里面 → **必须**重检索。

    ⚠️ fullctx 下**不再退到 CRAG**：`_crag_regenerate` 是全局取块（无篇配额）→ 跨篇问题
    会被修成"只讲第 1 篇"，且答案改写成 `[n]` 编号 → 锚点体系整体丢失。第一遍"同上下文
    再答"若没修好，就交给 `gate` 的 `salvage` 出**标注过的补充说明** —— 那比一份
    "覆盖残缺 + 引用跳不动"的答案更好。

    Returns: (new_answer, new_cites)；修不动/无对应类型问题 → None（gate 走兜底拒答）。
    """
    tgt = [i for i in issues if i["sev"] in ("high", "mid") and i.get("type") != "missing"]
    if not tgt or not (answer or "").strip():
        return None
    if _is_fullctx():
        return _try_fullctx_reask(question, pdfs, answer, tgt)
    has_ev = any(i["type"] in _EVIDENCE_TYPES for i in tgt)
    has_ge = any(i["type"] in _GENERATION_TYPES for i in tgt)
    # 生成问题（证据在没用对）先 Self-Refine（便宜）；仍有 evidence 型失败再做 CRAG。
    if has_ge:
        got = _try_refine(question, pdfs, answer, tgt, cites)
        if got:
            return got
    if has_ev:
        got = _try_crag(question, pdfs, answer, tgt, cites)
        if got:
            return got
    return None


# ── ③ 兜底前的「LLM 补充说明」（2026-10-02）────────────────────────────────────
#
# 动机：闸门修复失败后原本**直接拒答**（`FALLBACK_MSG`）—— 把"证据不足"一刀切成
# "什么都不说"。但多数失败是**部分**可确证的：原答案里有的说法有据、有的没据。
# 全拒答会把**有据的那部分也一起丢掉**。
#
# 改法：叫**一次** LLM 产出**补充说明**，把「能确证 / 不能确证」分开写清，
# 并**强制打上标注**（`SALVAGE_BANNER`）—— 前端与用户都必须能一眼看出
# **这不是系统作答**，而是系统给出的"补充"。
#
# 与 ①② 的区别：①② 试图**修好原答案**（仍算系统作答）；③ 不修，
# 而是**降级为"补充"**、并把这个降级事实**显式写在答案里**。

def supplement_mark() -> str:
    """「非系统作答」标注的**唯一来源**（定义在 `validator.SUPPLEMENT_MARK`）。

    延迟导入避开模块级循环。⚠️ `gate` 那一层还会**再兜一次** ——
    所以即使换了 salvage 实现、或模型不照做，标注也不会丢。
    """
    from paperpilot.components import validator as _V
    return _V.SUPPLEMENT_MARK


_SYS_SALVAGE_T = (
    "你是论文问答助手。**系统先前给出的答案未通过证据校验**，现在需要你产出一段"
    "**补充说明** —— 不是重写答案、也不是替系统作答。请严格遵守：\n"
    "1) 输出第一行必须原样写：{mark}\n"
    "2) 正文分两部分，标题原样使用：\n"
    "   **能从证据确证的部分**：只写【可引用原文】里能直接读出的结论，逐条标 [n]。\n"
    "   **无法确证的部分**：逐条指出系统原答案里哪些说法在【可引用原文】中找不到支撑，"
    "写明「未找到支撑」。\n"
    "3) **只依据【可引用原文】**：不得引入外部知识、不得推测补齐；"
    "确实没有就写「给定文本中未提及」。\n"
    "4) 数字、指标名、方法名、专有名词必须**逐字照抄原文**（原文是英文就保留英文）。\n"
    "5) 直接给补充说明，不要复述题目、不要解释你的流程。"
)


def salvage(question: str, answer: str, issues: list[dict[str, Any]],
            cites: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]] | None:
    """闸门修复失败后，产出**明确标注为"补充"**的说明（一次 LLM 调用）。

    与 `repair` 的分工：`repair` 试图**修好**原答案（仍算系统作答，失败才到这）；
    本函数**不修**，改为"降级成补充 + 显式标注"，让用户能拿到"部分可确证"的信息，
    而不是一纸拒答。

    Returns: `(补充文本, cites)`；无被引证据 / 调用失败 / 空输出 → `None`（gate 仍走拒答）。
    """
    entries = _entries_from_cites(cites)
    if not entries:
        return None                       # 连被引原文都没有 → 没什么可补充的，交给兜底
    iss = [i for i in (issues or []) if i.get("sev") in ("high", "mid")]
    lines = "\n".join(
        f"- [{i.get('type')}] {str(i.get('detail') or i.get('sentence') or '')[:200]}"
        for i in iss[:8]) or "（未给出具体意见）"
    user = (f"【问题】{question}\n\n"
            f"【系统原答案（未通过校验）】\n{str(answer or '')[:2000]}\n\n"
            f"【校验发现的问题】\n{lines[:1200]}\n\n"
            f"【可引用原文（[n] 对应编号；只是被引片段，非全文）】\n{_fmt_entries(entries)}\n\n"
            "请输出补充说明：")
    mark = supplement_mark()
    try:
        out = llm.chat_text(_SYS_SALVAGE_T.format(mark=mark), user,
                            temperature=0.0, max_tokens=900)
    except llm.LLMError:
        return None
    s = str(out or "").strip()
    if not s:
        return None
    # ★ 模型没照做也必须带上标注 —— 这是"不得冒充系统作答"的硬保证（不依赖模型自觉）
    if mark not in s:
        s = f"{mark}\n{s}"
    # 补充说明依据的仍是**同一批被引证据**，故 cites 原样带过（前端引用跳转不受影响）
    return s, list(cites or [])
