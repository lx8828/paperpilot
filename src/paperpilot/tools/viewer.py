"""Viewer：claims → 去重主张组 → 角色打标 → 规则算分 → Markdown 视图。

数据流（第一版）：
  claims(json) → LLM 去重归并 → ClaimGroup[] → LLM 角色打标
              → 本地规则算分（importance 1-5，带分量解释）
              → render_markdown() / summary json

设计要点：
  - rep 文本直接取自组内原 claim 的 text（可溯源），LLM 不新写
  - LLM 只做两类离散判断：归并、角色 label；分数全部本地规则计算
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from paperpilot.models.schema import Claim, ClaimGroup, ClaimLabel
from paperpilot.prompts.viewer import (DEDUPE_SYSTEM, LABEL_SYSTEM,
                                       build_dedupe_user, build_label_user)
from paperpilot.tools import llm

LABEL_BASE: dict[str, float] = {
    "core_claim": 5.0,
    "result_primary": 4.5,
    "result_supporting": 3.5,
    "method_core": 4.0,
    "ablation": 3.0,
    "detail": 2.5,
    "limitation": 3.5,
    "related": 2.0,
}
_LABELS: set[str] = set(ClaimLabel.__args__)  # type: ignore[attr-defined]

# 5 分资格上限：只有"卖点级"角色可到 5，次级方法/结果/局限 cap 在 4，
# 实现细节/相关工作 cap 在 3（实验结论：否则 5 分被次级发现稀释，必读臃肿）
LABEL_CAP: dict[str, int] = {
    "core_claim": 5,
    "result_primary": 5,
    "method_core": 4,
    "result_supporting": 4,
    "ablation": 4,
    "limitation": 4,
    "detail": 3,
    "related": 3,
}

# ── section / 位置 ──────────────────────────────────────


def top_path(claim: Claim) -> str:
    return claim.title_path[0] if claim.title_path else "(无标题)"


def section_tail(path0: str) -> str:
    """顶层标题的可读名，去掉 KIND/编号 前缀，如 '1 Introduction'。"""
    return path0.split("·")[-1].strip()


# 位置权重（激进档，可在实验中调整）
# 实验依据：Abstract/Conclusion 出现的组几乎必然重点（两篇验证 100% 必读）；
# 但方法类论文的核心主张大量分布在正文，位置不可作为唯一判据。
POS_ABSTRACT = 2.0    # 摘要/结论/讨论：约等于必读
POS_INTRO = 1.0       # 引言：重述重点
POS_BODY = 0.0        # 正文实验/方法：中性（核心可能在此）
POS_APPENDIX = -1.0   # 附录/字母章节：支撑材料


def position_adj(path0: str) -> float:
    """该顶层章节的位置权重。"""
    up = path0.upper()
    tail = section_tail(path0)
    if "ABSTRACT" in up:
        return POS_ABSTRACT
    low = tail.casefold()
    if any(k in low for k in ("conclusion", "discussion", "concluding", "summary")):
        return POS_ABSTRACT
    if "INTRODUCTION" in up or low.startswith("introduction"):
        return POS_INTRO
    if up.startswith("APPENDIX") or re.match(r"^[A-Za-z]\s", tail):
        return POS_APPENDIX  # 附录 / 字母标题章节
    return POS_BODY


def _ordered_sections(claims: list[Claim]) -> list[str]:
    """正文顺序的顶层 section 展示名（去重）。"""
    seen: OrderedDict[str, None] = OrderedDict()
    for c in claims:
        p = top_path(c)
        seen.setdefault(section_tail(p), None)
    return list(seen)


# ── ① LLM 去重归并 ──────────────────────────────────────


def _dedupe_call(claims: list[Claim]) -> list[dict]:
    sec_order = _ordered_sections(claims)
    by_sec: "OrderedDict[str, list[dict]]" = OrderedDict((s, []) for s in sec_order)
    for c in claims:
        by_sec[section_tail(top_path(c))].append(
            {"claim_id": c.claim_id, "type": c.type, "text": c.text})
    payload = list(by_sec.items())
    rows = llm.chat_json(DEDUPE_SYSTEM, build_dedupe_user(payload),
                         temperature=0.0)
    if not isinstance(rows, list):
        raise ValueError("去重返回非数组")
    return [r for r in rows if isinstance(r, dict)]


def dedupe_groups(claims: list[Claim]) -> list[ClaimGroup]:
    """LLM 全局归并 → ClaimGroup 列表。LLM 失败/不完整时回退为单组，保证全量覆盖。"""
    by_id = {c.claim_id: c for c in claims}
    order = {c.claim_id: i for i, c in enumerate(claims)}
    groups: list[ClaimGroup] = []
    used: set[str] = set()

    def add(best: str, members: list[str]) -> None:
        mem = [m for m in members if m in by_id and m not in used]
        if best not in mem:
            mem = [best] + mem
        for m in mem:
            used.add(m)
        group_claims = [by_id[m] for m in mem]
        # 去重 sections（按正文顺序）与 pages
        sections: "OrderedDict[str, None]" = OrderedDict()
        for c in sorted(group_claims, key=lambda c: order[c.claim_id]):
            sections.setdefault(section_tail(top_path(c)), None)
        pages = sorted({c.page for c in group_claims})
        rep = by_id[best]
        groups.append(ClaimGroup(
            group_id=f"g{len(groups) + 1}",
            rep_claim_id=best,
            rep_text=rep.text.replace("\n", " ").strip(),
            claim_ids=mem,
            type=rep.type,
            sections=list(sections),
            pages=pages,
        ))

    try:
        rows = _dedupe_call(claims)
        for r in rows:
            best = str(r.get("best_id", "")).strip()
            ids = r.get("claim_ids")
            if not isinstance(ids, list):
                ids = []
            members = [str(x).strip() for x in ids if isinstance(x, str)]
            if best and best in by_id:
                add(best, members)
    except (llm.LLMError, ValueError) as e:
        print(f"  [去重] LLM 失败({e})，回退为单组")
        used.clear()

    # 覆盖校验：漏掉的补单组；所有 claim 恰好一组
    for c in claims:
        if c.claim_id not in used:
            add(c.claim_id, [c.claim_id])
    return groups


# ── ② LLM 角色打标 ──────────────────────────────────────


def label_groups(groups: list[ClaimGroup]) -> None:
    """LLM 给每个组判 1 个角色 label（就地写入）。失败则该组保持 detail。"""
    payload = [{"group_id": g.group_id, "type": g.type,
                "sections": g.sections, "text": g.rep_text} for g in groups]
    result: dict[str, dict] = {}
    try:
        rows = llm.chat_json(LABEL_SYSTEM, build_label_user(payload),
                             temperature=0.0)
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict):
                    result[str(r.get("group_id", ""))] = r
    except (llm.LLMError, ValueError) as e:
        print(f"  [打标] LLM 失败({e})，全部保持 detail")
    for g in groups:
        r = result.get(g.group_id)
        if r and str(r.get("label", "")) in _LABELS:
            g.label = str(r["label"])          # type: ignore[assignment]
            g.label_why = str(r.get("why", "")).strip()


# ── ③ 本地规则算分 ──────────────────────────────────────


def _quant_signals(text: str) -> tuple[bool, bool, bool]:
    has_num = bool(re.search(r"\d", text))
    low = text.casefold()
    has_pct = bool(re.search(r"%|％|倍|翻", low)) or "percent" in low or "fold" in low
    has_cmp = bool(re.search(
        r"从.{0,15}(到|提升至|增长到)|高于|低于|优于|超过|提升|下降|翻倍|显著|"
        r"outperform|improve|reduce|outperform|surpass|exceed|greater|higher", low))
    return has_num, has_pct, has_cmp


def score_groups(groups: list[ClaimGroup], claim_map: dict[str, Claim],
                 ev_map: dict[str, str]) -> None:
    """规则算分。就地写 group.importance / score 分量。"""
    for g in groups:
        base = LABEL_BASE.get(g.label, LABEL_BASE["detail"])
        # 位置：取组内出现的顶层章节中最强档
        pos_adj = 0.0
        for cid in g.claim_ids:
            c = claim_map.get(cid)
            if c is None:
                continue
            a = position_adj(top_path(c))
            pos_adj = max(pos_adj, a)
        # occurrence：跨章节重复强调
        occ_adj = min(max(len(g.sections) - 1, 0), 2) * 0.3
        # 量化信号（仅 result 系参与；detail 的数字不提升重要性）
        has_num, has_pct, has_cmp = _quant_signals(g.rep_text)
        q_adj = 0.0
        if g.label == "result_primary":
            q_adj = 0.6 if (has_pct or has_cmp) else -0.5
        elif g.label == "result_supporting":
            q_adj = 0.3 if (has_pct or has_cmp) else 0.0
        # evidence 状态
        ev = ev_map.get(g.rep_claim_id, "miss")
        ev_adj = {"hit": 0.3, "loose": 0.0, "miss": -0.8}.get(ev, -0.8)
        g.ev_state = ev
        raw = round(base + pos_adj + occ_adj + q_adj + ev_adj)
        cap = LABEL_CAP.get(g.label, 5)
        g.importance = max(1, min(cap, raw))
        g.score = {"base": base, "pos": pos_adj, "occ": occ_adj,
                   "quant": q_adj, "ev": ev_adj, "total": raw}


# ── ④ Markdown 渲染 ──────────────────────────────────────

_EV_MARK = {"hit": "✓", "loose": "~", "miss": "⚠"}
_MAX_HEADLINE = 10  # "必读"区展示上限，其余 5 分仍出现在章节精读里


def render_markdown(groups: list[ClaimGroup], meta: dict[str, Any]) -> str:
    """渲染重点摘要 Markdown。"""
    n = meta.get("n_claims", 0)
    title = meta.get("title") or meta.get("pdf", "")
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    must_all = sorted([g for g in groups if g.importance >= 5],
                      key=lambda g: g.score.get("total", 0), reverse=True)
    must_top = must_all[:_MAX_HEADLINE]
    limits = [g for g in groups if g.label == "limitation"]
    lines.append(f"> 原始 claims {n} 条 → 去重后 {len(groups)} 个主张组 → "
                 f"必读 {len(must_all)} 条 ｜ {meta.get('pdf', '')}")
    lines.append("")

    # 必读（top 截断，其余仍见章节精读）
    lines.append(f"## 🎯 必读 Top{min(len(must_all), _MAX_HEADLINE)}"
                 f"（importance 5 共 {len(must_all)} 条）")
    if must_top:
        for g in must_top:
            lines.append(_fmt_item(g))
        if len(must_all) > len(must_top):
            lines.append(f"\n_另有 {len(must_all) - len(must_top)} 条 5 分主张"
                         f"见下方「章节精读」。_")
    else:
        lines.append("_（无）_")
    lines.append("")

    # 局限
    show_lim = [g for g in limits if g.importance >= 4]
    if show_lim:
        lines.append(f"## ⚠️ 局限（{len(show_lim)}/{len(limits)} 条 ≥4 分）")
        for g in sorted(show_lim, key=lambda g: g.importance, reverse=True):
            lines.append(_fmt_item(g))
        lines.append("")

    # 章节精读
    lines.append("## 📖 章节精读")
    by_sec: "OrderedDict[str, list[ClaimGroup]]" = OrderedDict()
    for g in groups:
        sec = g.sections[0] if g.sections else "(无标题)"
        by_sec.setdefault(sec, []).append(g)
    for sec, gs in by_sec.items():
        top_gs = [g for g in gs if g.importance >= 4]
        rest = [g for g in gs if g.importance < 4]
        head = f"### {sec}"
        if rest:
            head += f"　·　另 {len(rest)} 条细节按需查看"
        lines.append(head)
        for g in sorted(top_gs, key=lambda g: (g.importance,
                                                g.score.get("total", 0)),
                        reverse=True):
            lines.append(_fmt_item(g))
        if not top_gs:
            lines.append(f"_该节无 4 分以上主张（{len(gs)} 条均为细节）_")
        lines.append("")

    # 全部明细（低分折叠，供下钻）
    lines.append("## 🗂 全部主张（按重要性）")
    for g in sorted(groups, key=lambda g: -g.importance):
        lines.append(f"- `{g.group_id}` **{g.importance}** "
                     f"[{g.label}] {g.rep_text}　"
                     f"({g.ev_state}·p{g.pages[:3]}{'…' if len(g.pages) > 3 else ''})")
    return "\n".join(lines)


def _fmt_item(g: ClaimGroup) -> str:
    pages = g.pages[:3]
    page_str = ",".join(map(str, pages)) + ("…" if len(g.pages) > 3 else "")
    mark = _EV_MARK.get(g.ev_state, "⚠")
    why = f"　_{g.label_why}_" if g.label_why else ""
    return (f"- **{g.importance}** [{g.label}] {g.rep_text}"
            f"　`{mark} p{page_str}`{why}")
