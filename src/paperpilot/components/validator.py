"""Validator 门面：最终输出前的答案检查（检出问题 → 上层标注 / 打回重答）。

定位（2026-09-09 定）：**输出前闸门**，不是离线判分器（判分 = T 层外部裁判，带 gold）。
不判"漏了哪个"，只检出"明显有问题"：引用悬空 / 编数 / 无证据断言 / 疑似漏项 / 自相矛盾 /
指代模糊 / 偏题。任何问题 → 标注给用户（"此句可能无原文支撑"）或打回 Generator 重答。

分层与分级（2026-09-09 校准收敛）：
  机器件 = 硬 gate（稳定、0 LLM 成本）：A 引用越界、C 漏数/编数（答案新给的显著数字
    须能在被引原文找到）→ HIGH → 兜底/打回。
  两条误杀豁免（2026-09-10，MinerU hard 自测 4/30 假阳性定位）：
    ① 原文自带引用：越界编号若在被引证据原文里以 "[n]" 原样出现，是论文参考文献序号
       被模型照抄（如 `existing research [29]`），不是悬空的上下文引用 → 不判越界。
    ② 派生数：答案同时给出操作数与计算结果（如 `66.0% − 35.8% = 30.2`），且操作数均有
       证据支撑时，差值/和值（30.2）不算编数——SYSTEM 准则 8 明确要求算出差量。
  LLM 体检 = 软标注（免费 judge 二元判定不稳，不当硬 gate 依据）：unsupported / off_topic /
    missing(env 开) / contradiction / vague → MID → 前端标注"AI 复核"，不触发兜底。
  判断题（"X 对吗/是否…"）只走机器守护、不做 LLM 体检（negqa 校准：免费 judge 在
    "复述题干数值并否定"句法上误报 unsupported ~30%，few-shot/豁免难根治；机器 C+A 已足够）。
证据与铁律见 docs/RAG_COMPONENT_NOTES.md §2-⑧。
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

# ── 类型与严重度 ─────────────────────────────────────────────────────────────
HIGH = "high"
MID = "mid"
LOW = "low"
TYPES = {"citation", "number", "unsupported", "missing", "contradiction", "vague", "off_topic"}

_NUMERIC_RE = re.compile(
    r"how much|by how much|how many|how (fast|quick(ly)?|large|big|high|long|often)|"
    r"what (is|are) the (accuracy|f1|score|percentage|percent|performance|value|number|"
    r"size|rate|amount)|improvement|outperform|increase|reduc|drop|gain|"
    r"score|f1|acc(uracy)?\b|%|precision|recall", re.I)
_JUDE_RE = re.compile(
    r"对吗|是否正确|是否属实|是否准确|正确吗|准确吗|是不是|能否|是否存在|是否提到|是否使用|"
    r"\b(is|are|did|does|was|were|do)\b.*\?", re.I)
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?%?")
_CITE_RE = re.compile(r"\[(\d+)\]")


def is_numeric_question(question: str) -> bool:
    """问题是否要求具体数值/量化结果。"""
    return bool(_NUMERIC_RE.search(question))


def is_judgement_question(question: str) -> bool:
    """是否判断题（问"X 对吗/是否…"）。

    判断题走机器守护（C 数值 + A 引用），不做 LLM 语义体检——校准（2026-09-09 negqa 20）：
    免费 judge 对"复述题干数值并否定"的判断题句法误报 unsupported 达 ~30%，few-shot/豁免
    均难根治；而机器层（题干数剔除后核对真值 + 引用完整性）已能守护这类答案（negqa 20/20）。
    """
    return bool(_JUDE_RE.search(question))


def numbers_in(text: str) -> list[str]:
    """抽取文本中的数值 token（含小数/百分号），顺序保留。"""
    return _NUM_RE.findall(text)


def check_cites(text: str, entries: list[dict[str, Any]], pdf: str) -> list[dict[str, Any]]:
    """解析并校验答案 [n] 引用 → 合法 cites 列表（越界引用丢弃）。委派 answer._cites_from。"""
    from paperpilot.agents.nodes.answer import _cites_from
    return _cites_from(text, entries, pdf)


def _entry_text(e: dict[str, Any]) -> str:
    """条目 → 可作证据的原文文本（chunk 用全文，claim 用 evidence 句）。"""
    return str(e.get("text") or e.get("evidence") or "")


def _chunk_text(c: Any) -> str:
    """extra_chunks 元素（Chunk / dict / str）→ 文本。"""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("text") or c.get("evidence") or "")
    return str(getattr(c, "text", "") or "")


def _chunk_meta(c: Any, text: str) -> dict[str, Any]:
    """extra_chunks 元素 → 溯源元数据（chunk_id/page/section/text），供补充上下文用。"""
    if isinstance(c, dict):
        return {"chunk_id": str(c.get("chunk_id") or ""),
                "page": c.get("page") or (c.get("page_span") or (0,))[0],
                "text": text}
    return {"chunk_id": str(getattr(c, "chunk_id", "") or ""),
            "page": getattr(c, "page", None) or (getattr(c, "page_span", None) or (0,))[0],
            "text": text}


# ── 机器层（0 成本）──────────────────────────────────────────────────────────

def _machine_checks(question: str, answer: str, entries: list[dict[str, Any]],
                    n_entries: int | None = None,
                    extra_chunks: list[Any] | None = None,
                    adjudicate: bool = False
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    issues: list[dict[str, Any]] = []
    supplements: list[dict[str, Any]] = []
    # answer 引用编号对应"生成时的完整上下文条目数"，未必等于 len(entries)
    # （entries 可能是 cites 引用子集）。调用方可传 n_entries 指定真实上下文数。
    bound = n_entries if n_entries is not None else len(entries)
    cited = sorted({int(m) for m in _CITE_RE.findall(answer)})
    # A 引用结构：越界 / 无引用
    if cited:
        bad = [n for n in cited if n < 1 or n > bound]
        # 豁免①：越界编号若在被引证据原文中以 "[n]" 原样出现 → 论文自带参考文献序号
        # （模型照抄原文，如 `existing research [29]`），非悬空的上下文引用（2026-09-10）。
        ev_all = "\n".join(_entry_text(e) for e in entries)
        bad = [n for n in bad if f"[{n}]" not in ev_all]
        for n in bad:
            issues.append({"sev": HIGH, "type": "citation",
                           "sentence": answer[:120],
                           "detail": f"引用 [{n}] 越界（共 {bound} 个证据条目）"})
    elif n_entries and len(answer) > 20:
        issues.append({"sev": LOW, "type": "citation", "sentence": "",
                       "detail": "答案未给出任何 [n] 引用（若为概述直答可忽略）"})
    # C 数值支撑：有证据即核对（任何题型；漏数/编数内部判断）
    ev = [_entry_text(e) for e in entries]
    unverified: list[float] = []
    if ev or extra_chunks:
        hard, supplements, unverified = validate_numbers(question, answer, ev,
                                                         extra_chunks=extra_chunks)
        for msg in hard:
            issues.append({"sev": HIGH, "type": "number", "sentence": "", "detail": msg})
        if unverified and not adjudicate:
            # 无裁决通道（如修复回路内的复检）→ 保持旧行为：机器直接判 high
            issues.append({"sev": HIGH, "type": "number", "sentence": "",
                           "detail": "疑似无证据支撑的数字: "
                                     + ", ".join(_fmt_num(x) for x in unverified[:6])})
    return issues, supplements, unverified


# ── LLM 体检（一次调用/答案）────────────────────────────────────────────────

_SYS_VALIDATE = (
    "你是答案质检员。输入：论文中的问题、待发布答案、以及答案引用的原文片段（[n] 编号）。"
    "你的职责是找出**答案自身可自检的问题**，不要和标准答案对比（没有标准答案）。只输出 JSON。")

_VALIDATE_TPL = """【问题】
{question}

【答案】
{answer}

【答案引用的原文片段（编号与答案里的 [n] 对应）】
{evidence}

请逐句检查答案，只输出 JSON（找不到就空数组/false）：
{{
  "unsupported": [{{"sentence": "原句摘录", "detail": "为什么判定原文不支撑"}}],
    // 句子里有原文找不到的断言/数值/对象（编造、幻觉）。原文片段没提到就算。
  "missing": true 或 false,
    // 原文片段里明显还有与问题直接相关、但答案没提的要点（只提示存在，不用列举是哪个）
  "contradiction": [{{"sentence": "原句", "detail": "与哪句矛盾"}}],
  "vague": [{{"sentence": "原句", "detail": "哪个指代/表述不清"}}],
  "off_topic": true 或 false   // 答案是否答非所问/偏题
}}
【示例·判断题否定题干（正确答案，不得标 unsupported）】
问题：论文的模型在 ImageNet 上 Top-1 达到 99%，对吗？
答案：该说法不正确——原文报告的是 78.6% 而非 99% [1]。
原文片段：[1] we report 78.6% Top-1 accuracy on ImageNet...
正确输出：{{"unsupported": [], "missing": false, "contradiction": [], "vague": [], "off_topic": false}}
理由：78.6% 在原文有据；99% 是题干假设被否定，复述它不算编造。

【示例·真编造（必须标 unsupported）】
问题：这篇论文用了哪些数据集？
答案：用了 ImageNet-21k、COCO [1]，以及一个 900 万张的私有数据集 [1]。
原文片段：[1] we use ImageNet-21k and COCO...
正确输出：{{"unsupported": [{{"sentence": "以及一个 900 万张的私有数据集", "detail": "原文片段未提到该私有数据集"}}], ...}}
理由：ImageNet/COCO 有据；私有数据集是编造。

注意：
- 引用真实、内容在原文有据的好答案应得到空数组与 false。拿不准别乱标。
- 答案**否定题干**（"X 不对/并非 X，真值是 Y"）时：Y 必须有据、X 是题干假设——这类"判断题"正常答案不得标 unsupported。
- 原文片段只是检索子集（不全），不要因"片段里没看到"就断言"原文没有"；只有与已给片段明显冲突/片段明确不含时才标。"""


# 判断题·否定题干句的机器豁免（校准 2026-09-09：免费 judge 对"复述题干数值再否定"句法不稳，
# 这类句子的数值真伪已由机器层 C 校验覆盖——不送 LLM，避免把"否定题干"误标成 unsupported）。
_NEG_POLAR = ("不是", "并非", "不对", "不正确", "错误", "没有", "并未", "未提及",
              "未出现", "未提供", "未给出", "不含", "不存在", "无法确认", "无法验证",
              "无法判断", "无法肯定", "no", "not", "never", "incorrect", "wrong",
              "did not", "does not")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")


def _mask_judgement_sentences(text: str) -> str:
    """把"判断题/题干引用"句替换为占位（其数字真伪走机器 C 层校验，不送 LLM）。

    触发：句含题干转述线索（"你问的/论文称/该说法/题干…"）且含数字或百分比，或含否定词。
    这类句 judge 易误判（把"复述题干假值并否定"当 unsupported）；数字真假已由 C 层核对。
    """
    parts = _SENT_SPLIT.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        has_neg = any(w in p.lower() for w in _NEG_POLAR)
        claim_q = any(k in p for k in (
            "你问的", "你提到", "您问的", "您提到", "论文称", "论文声称", "论文报告",
            "论文中报告", "题干", "该说法", "该数值", "这一数值", "问题中", "所问"))
        if has_neg or (claim_q and ("%" in p or re.search(r"\d", p))):
            out.append("〔题干判断题句：复述题干/给真值，数值已机器核验，跳过语义判定〕")
        else:
            out.append(p)
    return "".join(out)


def _llm_physical(question: str, answer: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from paperpilot.tools import llm
    evidence = "\n".join(f"[{i+1}] {_entry_text(e)[:600]}"
                         for i, e in enumerate(entries[:10])) or "（无引用原文）"
    answer_body = _mask_judgement_sentences(answer)
    user = _VALIDATE_TPL.format(question=question, answer=answer_body[:2000], evidence=evidence)
    issues: list[dict[str, Any]] = []
    try:
        raw = llm.judge_json(_SYS_VALIDATE, user, temperature=0.0)
    except Exception:
        return issues
    if not isinstance(raw, dict):
        return issues
    for sent in raw.get("unsupported") or []:
        issues.append({"sev": MID, "type": "unsupported",
                       "sentence": str(sent.get("sentence", ""))[:160],
                       "detail": str(sent.get("detail", ""))[:160]})
    if raw.get("off_topic"):
        issues.append({"sev": MID, "type": "off_topic", "sentence": "", "detail": "AI 复核：答案可能答非所问/偏题"})
    if raw.get("missing") and os.environ.get("PAPERPILOT_VALIDATOR_MISSING") == "1":
        # missing 提示默认关（好答案上也频繁误报，降噪）；需要时 env 开启
        issues.append({"sev": MID, "type": "missing", "sentence": "",
                       "detail": "被引原文中可能还有与问题直接相关但未作答的要点（仅提示复核）"})
    for sent in raw.get("contradiction") or []:
        issues.append({"sev": MID, "type": "contradiction",
                       "sentence": str(sent.get("sentence", ""))[:160],
                       "detail": str(sent.get("detail", ""))[:160]})
    for sent in raw.get("vague") or []:
        issues.append({"sev": MID, "type": "vague",
                       "sentence": str(sent.get("sentence", ""))[:160],
                       "detail": str(sent.get("detail", ""))[:160]})
    return issues


# ── 主入口 ───────────────────────────────────────────────────────────────────

_SIG_RE = re.compile(r"-?\d+(?:\.\d+)?%?[KMB]?")
# 显著数字：含小数 / 带百分号 / ≥100（不含 4 位年份）
# —— 小序号(2/3/5)与年份高频且无判别力，剔除可避免普通句误报（校准 2026-09-09）
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# 年份**只在明确语境**下才剥离（2026-09-11）：旧实现无条件删 19xx/20xx，
# 把 `Switchboard-2000`、`2000 hours`、`2000 − 300` 里的数据值一并抹掉，
# 使差值/倍数豁免找不到操作数（实测误杀根因④）。
_YEAR_PRE = re.compile(r"(?i)\b(?:in|since|during|until|from|by|year|years|"
                       r"copyright|©)\b[^0-9.\n，。；]{1,16}$")
_YEAR_POST = re.compile(r"^\s*(?:年|年代|s\b|'s|’s)")


def _strip_year_context(text: str) -> str:
    """仅剥离"明显在当日期用"的 4 位年；其余 4 位数保留参与数值匹配。

    判据（封闭语法，不维护成员表）：前有 in/since/during/until/from/by/year/©，
    或后有 年/年代/'s。
    """
    if not text:
        return text
    out: list[str] = []
    last = 0
    for m in _YEAR_RE.finditer(text):
        pre = text[max(0, m.start() - 24):m.start()]
        post = text[m.end():m.end() + 8]
        if (_YEAR_PRE.search(pre) or _YEAR_POST.search(post)
                or "年" in post[:3] or "年" in pre[-3:]):
            out.append(text[last:m.start()])
            out.append(" " * (m.end() - m.start()))
            last = m.end()
    out.append(text[last:])
    return "".join(out)


def _significant_numbers(text: str) -> set[str]:
    """旧字面实现（保留兼容外部调用）；新逻辑走 _canonical_nums。"""
    sig: set[str] = set()
    body = _YEAR_RE.sub("", text)
    for m in _SIG_RE.finditer(body):
        t = m.group()
        if t.endswith("K") or t.endswith("M") or t.endswith("B"):
            sig.add(t)
            continue
        num = float(t.rstrip("%"))
        if "%" in t or "." in t or num >= 100:
            sig.add(t)
    return sig


def _canonical_nums(text: str) -> set[float]:
    """显著数值（语义归一）集合——把"同一数值、不同写法"归到同一量纲再比较。

    归一化实现在 components/numbers.py（Normalizer：parse_number / scan_numbers），
    本函数只保留**显著判定**这一策略层：谁值得进比对集合。
    显著判据（沿用 2026-09-09 校准口径）：百分比写法 / 原文含"." / 值≥100 /
    非整数值（欧式小数 7,5→7.5）；剔除 4 位年份与引用编号 [n]。
    归一覆盖由 numbers.py 提供：欧式小数、千分位、中英单位、科学计数
    （3.6×10^4 / 3.6e4 / LaTeX `$8.5 \\times 10^{11}$`）、中文网络单位（3.6w）、
    中文数词（三万六千）；单位吸附须两侧词界（`15,000 most` 不吃 m、`F1 百分点` 不造 100）。
    年份仅在明确日期语境下剥离（2026-09-11 改，见 _strip_year_context）。
    """
    from paperpilot.components.numbers import scan_numbers
    body = _strip_year_context(text)
    body = re.sub(r"\[\d+\]", "", body)
    out: set[float] = set()
    for t in scan_numbers(body):
        v = t.value
        if v is None:
            continue
        sig = t.percent or "." in t.raw or v != round(v) or v >= 100
        if sig:
            out.add(v)
    return out


def _fmt_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _is_derived(v: float, operands: list[float]) -> bool:
    """v 是否可由两个"答案中已给出、且有证据支撑"的操作数**经四则运算**推得。

    豁免②（2026-09-10 立；2026-09-11 扩展到乘/除 + 四舍五入容差）：
    SYSTEM 准则 8 要求答案写出算式与结果（`66.0 − 35.8 = 30.2`、
    `2000 ÷ 300 ≈ 6.67`、`293 ÷ 4262 ≈ 0.069`）——这些结果全文不存在却是**合规计算**，
    不能判编数。旧版只认差/和、容差 1e-3 → 除法/乘法与"写成两位小数"的结果全被误杀。

    5% 容差是给"答案把 0.0687 写成 0.069 / 0.07"这类四舍五入留的；
    两个操作数都必须已在 operands（= 答案与证据共有）内，避免把无关数字误豁免。
    """
    def near(x: float, y: float) -> bool:
        return (math.isclose(x, y, rel_tol=1e-3, abs_tol=1e-6)
                or math.isclose(x, y, rel_tol=5e-2, abs_tol=1e-6))

    for i, a in enumerate(operands):
        for b in operands[i + 1:]:
            if (near(abs(a - b), v) or near(a + b, v)
                    or (b and (near(a * b, v) or near(a / b, v)))
                    or (a and near(b / a, v))):
                return True
    return False


def validate_numbers(question: str, answer: str,
                     evidence_texts: list[str] | None = None,
                     extra_chunks: list[Any] | None = None
                     ) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """数值支撑（机器，0 成本）→ (hard_issues, supplements, unverified)。

    hard_issues：与"数字是否有据"无关的硬结论（如"数值型问题但答案没给数字"）。
    supplements：被引证据没有、但**全篇某 chunk 找到了**的数——不判对错，只发信号，
            由闸门把命中块作补充上下文喂回复核（可能是别处同值、语义不同的数）。
    unverified：被引证据没有、**全篇也找不到**的显著数字——**这不是结论**：
        "机器没找到" ≠ "论文没有" ≠ "模型编造"。parser 覆盖不全（LaTeX 残留、
        单位后缀词界、年份剥离）、派生运算、写法差异都会导致找不到 ——
        2026-09-11 实测这批 18/18 全是误杀。→ 交给上层 LLM 裁决
        （见 `gate` 的 `adjudicate`）；机器层只做**高召回的存在性检查**，不再单方面判死。
    防误报：剔除题干自带数（复述/否定题干假值）、引用编号 [n]、年份（仅明确语境）、小序号。
    """
    issues: list[str] = []
    supplements: list[dict[str, Any]] = []
    if is_numeric_question(question) and not numbers_in(answer):
        issues.append("数值型问题但答案未给出任何具体数字")
    if (evidence_texts or extra_chunks) and answer:
        q_sig = _canonical_nums(question)
        body = re.sub(r"\[\d+\]", "", answer)
        ans_all = _canonical_nums(body)  # 未扣题干：派生操作数要用**全集**
        ans_sig = ans_all - q_sig
        if not ans_sig:
            return issues, supplements, []
        ev_sig: set[float] = set()
        for ev_ in evidence_texts or []:
            ev_sig |= _canonical_nums(ev_)
        unmatched = ans_sig - ev_sig
        # 豁免②：答案里给出、且**证据侧也有**的操作数，其四则运算结果不算编数。
        # 操作数取 ans_all（**不扣题干**）——否则 `Switchboard-2000 − Switchboard-300`
        # 这类题的操作数会被 q_sig 一并扣掉，豁免直接失效（2026-09-11 实测根因）。
        operands = sorted(ans_all & ev_sig)
        if operands and unmatched:
            unmatched = {v for v in unmatched if not _is_derived(v, operands)}
        if not unmatched:
            return issues, supplements, []
        if not extra_chunks:
            # 无全篇可用：无法区分"跨块"与"编数" → 全部交裁决（不再机器判死）
            return issues, supplements, sorted(unmatched)
        # 按 chunk 溯源：找到被引证据缺失数字所在的具体 chunk
        sup_map: dict[float, list[dict[str, Any]]] = {}
        for ch in extra_chunks:
            text = _chunk_text(ch)
            if not text:
                continue
            for v in (_canonical_nums(text) & unmatched):
                sup_map.setdefault(v, []).append(_chunk_meta(ch, text))
        real_unmatched = unmatched - set(sup_map)
        for v in sorted(sup_map):
            supplements.append({
                "num": v,
                "num_text": _fmt_num(v),
                "chunks": sup_map[v][:3],  # 最多 3 个命中块作补充上下文
            })
        return issues, supplements, sorted(real_unmatched)
    return issues, supplements, []


def check(question: str, answer: str,
          entries: list[dict[str, Any]] | None = None,
          use_llm: bool | None = None,
          n_entries: int | None = None,
          extra_chunks: list[Any] | None = None,
          adjudicate: bool = False) -> dict[str, Any]:
    """输出前答案检查。返回 {"ok", "high", "issues", "supplements", "unverified"}。

    issues 每项 {sev: high|mid|low, type, sentence, detail}。
    supplements 每项 {num, num_text, chunks:[{chunk_id,page,text}]}：
      被引证据里没有、但全篇某 chunk 找到的数字——不是对错结论，是"待补充复核"信号
      （数字可能是别处同值、语义不同的另一个数，不能静默放行）。调用方拿它做
      补充上下文喂回 Generator（见 gate 的 supplement 参数）。
    调用方策略：high → 打回重答/拒答；mid → 标注给用户复核；low → 忽略/提示。
    entries = Generator 用的上下文条目（带 kind/text 或 evidence）；缺省只跑文本机器检查。
    n_entries = 答案引用编号对应的真实上下文条目数（越界检查基准；缺省用 len(entries)）。
    extra_chunks = 全篇 chunks（Chunk/dict），用于区分"跨块合理"与"真编数"并溯源。
    use_llm 默认开（可 PAPERPILOT_VALIDATOR_LLM=0 关）。
    """
    entries = list(entries or [])
    use_llm = (os.environ.get("PAPERPILOT_VALIDATOR_LLM", "1") == "1") if use_llm is None else use_llm
    machine_issues, supplements, unverified = _machine_checks(
        question, answer, entries, n_entries=n_entries,
        extra_chunks=extra_chunks, adjudicate=adjudicate)
    issues: list[dict[str, Any]] = list(machine_issues)
    # 判断题走机器守护（C+A），不做 LLM 语义体检（见 is_judgement_question docstring）
    if use_llm and not is_judgement_question(question) and entries and (answer or "").strip():
        issues += _llm_physical(question, answer, entries)
    # 排序：high 在前
    issues.sort(key=lambda x: 0 if x["sev"] == HIGH else 1 if x["sev"] == MID else 2)
    has_high = any(i["sev"] == HIGH for i in issues)
    return {"ok": not has_high, "high": has_high, "issues": issues,
            "supplements": supplements, "unverified": unverified}


# ── 输出闸门（2026-09-09）──────────────────────────────────────────────────

FALLBACK_MSG = "抱歉，我可能无法准确回答这个问题——该答案未能通过内部事实校验。"


def gate(question: str, answer: str, cites: list[dict[str, Any]],
         *, repair=None, supplement=None, adjudicate=None, use_llm: bool | None = None,
         n_entries: int | None = None,
         extra_chunks: list[Any] | None = None) -> dict[str, Any]:
    """输出前闸门（接入 graph 的 answer 之后）。

    策略（2026-09-09 定，repair 2026-09-09 接上；supplement 2026-09-09 接上）：
      high 级问题 → 交给 repair 对症修复（Self-Refine/CRAG）；修好 → repaired；
      修不好/无 repair → 统一输出兜底话术（拒答，不走 unknown 路径）。
      mid/low → 原样输出（issues 随 debug 返回，供前端标注）。
      env PAPERPILOT_VALIDATOR_REPAIR_MID=1 时 mid 的 unsupported/off_topic/contradiction/
      vague 也触发 repair（默认关：软标注不硬修）。
      **supplement（补充复核）**：无 high 时，若机器数字层发现"被引证据缺失、
      但全篇某 chunk 找到该数"（supplements 非空）→ 不静默放行、也不判对错，
      调 supplement(question, answer, supplements, cites) 把命中块作补充上下文
      喂回 Generator 决定是否需改；Generator 判定无需改 → pass，改 → repaired。
      复核仅一次（补出的数字若仍无被引证据，交给后续校验，不无限循环）。
      **adjudicate（数值裁决，2026-09-11 接上）**：机器层报"被引证据没有、**全篇也
      找不到**"的显著数字时（`unverified`），**不再由机器直接判 high**——这类
      "找不到"大部分是 parser 覆盖不全/派生运算/写法差异造成的误杀（实测 18/18）。
      改调 adjudicate(question, answer, nums, cites) → (supported, reason)：
      判"支持" → 放行；判"不支持"/复核异常 → 才升为 high 走 repair/兜底。
      未传 adjudicate（如修复回路内的复检）时保持旧行为（机器判 high）。

    Args:
        question: 用户问题
        answer: Generator 产出的答案（含 [n] 引用）
        cites: answer 的引用列表（每项带 evidence 原文文本）
        repair: 修复器 (question, answer, issues, cites)
                -> (new_answer, new_cites) | None。None → 走兜底。
        supplement: 补充复核器 (question, answer, supplements, cites)
                -> (new_answer, new_cites) | None。None → Generator 判定无需修改。
        adjudicate: 数值裁决器 (question, answer, nums, cites) -> (supported: bool, reason)。
                对"全篇找不到"的数字给出最终裁决（机器只出信号）。
        use_llm: 体检是否走 LLM（默认读 PAPERPILOT_VALIDATOR_LLM）
        extra_chunks: 全篇 chunks（Chunk/dict），数字层溯源用。
    Returns:
        {"action": "pass"|"fallback"|"repaired", "answer": 最终输出文本,
         "cites": 最终引用（repaired 时为新对齐引用）, "issues": [...],
         "supplements": [...]}（供 debug/前端标注）
    """
    entries = [{"kind": "chunk", "text": str(c.get("evidence") or ""), "page": c.get("page", 0)}
               for c in cites if c.get("evidence")]
    res = check(question, answer, entries, use_llm=use_llm, n_entries=n_entries,
                extra_chunks=extra_chunks, adjudicate=adjudicate is not None)
    issues = res["issues"]
    supplements = list(res.get("supplements") or [])
    # 机器"全篇找不到"的显著数字 → **交裁决**，不再由机器单方面判死。
    # 机器层只做高召回的存在性检查：parser 覆盖不全（LaTeX/单位词界/年份）、派生运算、
    # 写法差异都会导致"找不到"（2026-09-11 实测这批 18/18 全是误杀）。
    # 只有复核判"不支持"（或复核异常）才升为 high，走既有的 repair → 兜底链路。
    unverified = list(res.get("unverified") or [])
    if unverified:
        ok, why = (False, "无裁决通道（保守处理）")
        if adjudicate is not None:
            try:
                ok, why = adjudicate(question, answer, unverified, cites)
            except Exception as e:  # noqa: BLE001  复核失败不抛断，按不支持保守处理
                ok, why = False, f"复核异常: {type(e).__name__}"
        if not ok:
            issues.append({"sev": HIGH, "type": "number", "sentence": "",
                           "detail": "疑似无证据支撑的数字（复核判定不支持）: "
                                     + ", ".join(_fmt_num(x) for x in unverified[:6])
                                     + f"（{why}）"})
    high = [i for i in issues if i["sev"] == HIGH]
    trigger = bool(high)
    if repair is not None and not trigger and os.environ.get("PAPERPILOT_VALIDATOR_REPAIR_MID") == "1":
        trigger = any(i["sev"] == MID and i["type"] in
                      ("unsupported", "off_topic", "contradiction", "vague") for i in issues)
    if trigger:
        if repair is not None:
            try:
                got = repair(question, answer, issues, cites)
            except Exception:  # noqa: BLE001  修复失败不抛断，按兜底处理
                got = None
            if isinstance(got, tuple) and len(got) == 2:
                new, new_cites = got
                if str(new or "").strip() and str(new).strip() != answer:
                    return {"action": "repaired", "answer": str(new).strip(),
                            "cites": list(new_cites or []), "issues": issues,
                            "supplements": supplements}
        return {"action": "fallback", "answer": FALLBACK_MSG, "cites": [],
                "issues": issues, "supplements": supplements}
    # 无 high：有 supplement 信号 → 补充复核一次（Generator 判断是否需改）
    if supplements and supplement is not None:
        try:
            got = supplement(question, answer, supplements, cites)
        except Exception:  # noqa: BLE001  复核失败不抛断，按"无需修改"放行
            got = None
        if isinstance(got, tuple) and len(got) == 2:
            new, new_cites = got
            if str(new or "").strip() and str(new).strip() != answer:
                return {"action": "repaired", "answer": str(new).strip(),
                        "cites": list(new_cites or []), "issues": issues,
                        "supplements": supplements}
    return {"action": "pass", "answer": answer, "issues": issues,
            "supplements": supplements}
