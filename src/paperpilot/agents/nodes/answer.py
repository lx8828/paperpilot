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

from paperpilot.agents.state import QAState
from paperpilot.tools import llm

SYSTEM = (
    "你是严谨的论文问答助手，只能依据提供的『论文概述 / 检索主张 / 正文段落』作答。"
    "回答须标注引用编号 [n]（指向对应信息条目）。若信息不足以准确回答，"
    "请如实说明缺什么，不要编造数字、方法或结论。用中文回答，控制在 3~6 句。"
)

SYSTEM_UNKNOWN = (
    "你是严谨的论文问答助手。用户的问题在论文中找不到可支撑的内容，"
    "请诚实地说明这一点，并简要说清楚缺少的是哪类信息。不要编造。用中文，1~3 句。"
)

CITE_RE = re.compile(r"\[(\d{1,2})\]")


# ── 条目格式化（claim 样式 / chunk 样式统一编号）─────────────────────────────


def _fmt_claim_entry(i: int, e: dict) -> str:
    sec = (e.get("sections") or e.get("title_path") or [""])
    sec = sec[0] if isinstance(sec, list) and sec else ""
    return (
        f"[{i}] 主张[{e.get('label','')}]（{e.get('importance',0)}分，{sec}，"
        f"p{e.get('page','?')}）：{e.get('text','')}\n"
        f"    原文证据：{e.get('evidence','')}"
    )


def _fmt_chunk_entry(i: int, c: dict) -> str:
    return (f"[{i}] 正文段落（p{c.get('page','?')} {c.get('section','')}）："
            f"{c.get('text','')}")


# ── cites 解析 ────────────────────────────────────────────────────────────────


def _cites_from(text: str, entries: list[dict], pdf: str) -> list[dict[str, Any]]:
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
                "evidence": e.get("text", "")[:120],
                "page": e.get("page", 0),
            })
    return cites


# ── 各档证据取料 ──────────────────────────────────────────────────────────────


def _claim_entries(items: list[dict]) -> list[dict[str, Any]]:
    """retrieved claims / core_points 都转成 claim 样式条目。"""
    out = []
    for it in items:
        entry = {**it, "kind": "claim"}
        entry.setdefault("sections", [])
        out.append(entry)
    return out


def _chunk_entries(items: list[dict]) -> list[dict[str, Any]]:
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
    if state.get("l3_chunks"):
        entries = _chunk_entries(state["l3_chunks"])
        header = f"标题：{state.get('title','')}\n\n=== 全文检索到的正文（独立检索） ==="
        return header, entries, "L3"
    if state.get("chunks"):
        entries = _chunk_entries(state["chunks"])
        header = f"论文概述：{overview or '（无）'}\n\n=== 正文窗口 ==="
        return header, entries, "L2"
    if state.get("retrieved"):
        entries = _claim_entries(state["retrieved"])
        header = f"论文概述：{overview or '（无）'}\n\n=== 检索到的相关主张 ==="
        return header, entries, "L1"
    entries = _claim_entries(state.get("core_points") or [])
    header = f"论文概述：{overview or '（无）'}\n\n=== 核心要点 ==="
    return header, entries, "L0"


# ── 节点：generate_answer ─────────────────────────────────────────────────────


def generate_answer(state: QAState) -> dict[str, Any]:
    question = state.get("question", "")
    pdf = state.get("pdf", "")
    header, entries, level = _build_context(state)

    parts = [header]
    if level in ("L1", "L0"):
        fmt = _fmt_claim_entry
    else:
        fmt = _fmt_chunk_entry
    for i, e in enumerate(entries, 1):
        parts.append(fmt(i, e))
    if not entries:
        parts.append("（未检索到相关条目）")
    context = "\n\n".join(parts)

    user = (f"{context}\n\n{_fmt_history(state)}\n\n用户问题：{question}\n\n"
            f"（若信息不足以回答，请说明缺什么，不要编造。）")
    answer = llm.chat_text(SYSTEM, user, temperature=0.0)
    cites = _cites_from(answer, entries, pdf)
    verdict = state.get("verdict") or {}

    route = list(state.get("route") or [])
    if f"answer_{level}" not in route:
        route.append(f"answer_{level}")
    debug = dict(state.get("debug") or {})
    debug["answer"] = {
        "level": level,
        "n_entries": len(entries),
        "enough": verdict.get("enough"),
        "gap": (verdict.get("gap") or "")[:120],
    }
    return {
        "answer": answer,
        "cites": cites,
        "route": route,
        "debug": debug,
    }


# ── 节点：answer_unknown（诚实收尾）─────────────────────────────────────────


def answer_unknown(state: QAState) -> dict[str, Any]:
    question = state.get("question", "")
    gap = (state.get("verdict") or {}).get("gap", "")
    answer = llm.chat_text(
        SYSTEM_UNKNOWN,
        f"{_fmt_history(state)}\n\n问题：{question}\n\n"
        f"补充：检索到的内容不足以回答，缺口信息：{gap or '未知'}\n\n"
        f"请诚实告知用户无法从该论文中找到答案。",
        temperature=0.0,
    )
    route = list(state.get("route") or [])
    if "answer_unknown" not in route:
        route.append("answer_unknown")
    debug = dict(state.get("debug") or {})
    debug["answer"] = {"level": "unknown", "gap": gap[:120]}
    return {
        "answer": answer,
        "cites": [],
        "route": route,
        "debug": debug,
    }
