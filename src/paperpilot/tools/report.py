"""单篇精读报告：把 claims(view) + skeleton(论证关系) + 概述 串成完整 Markdown。

报告结构：
    标题/元信息 → 📋 概述 → 🎯 核心要点 → 🕸️ 论证骨架
    → ⚠️ 局限 → 📖 章节精读 → 🗂 全部主张

读取产物（不重跑旧流程）：
    assets/artifacts/out_claims/<pdf>.claims.json   统计信息
    assets/artifacts/out_views/<pdf>.summary.json   分组/打标/打分
    assets/artifacts/out_views/<pdf>.skeleton.json  Hub 论证关系
"""
from __future__ import annotations

from typing import Any

from paperpilot.prompts.report import (GUIDE_SYSTEM, OVERVIEW_SYSTEM,
                                       build_guide_user, build_overview_user)
from paperpilot.tools import llm

_EV = {"hit": "✓", "loose": "~", "miss": "⚠"}
_REL_CN = {"implements": "实现机制", "supports": "证据支撑",
           "limits": "边界限制", "contrasts": "对照区分"}


# ── 概述 ───────────────────────────────────────────


def _narrative(g: dict[str, Any]) -> bool:
    """该主张是否位于叙述性章节（摘要/引言/结论/讨论）。"""
    secs = g.get("sections") or []
    for s in secs[:1]:
        low = s.casefold()
        if ("abstract" in low or "introduction" in low
                or "conclusion" in low or "discussion" in low
                or "summary" in low or "concluding" in low):
            return True
    return False


def collect_materials(groups: list[dict[str, Any]], max_n: int = 12) -> list[dict[str, Any]]:
    """概述材料：摘要/引言/结论里的高分核心主张（优先叙述层）。"""
    cand = [g for g in groups
            if g.get("label") in ("core_claim", "result_primary")
            and g.get("importance", 0) >= 4]
    cand.sort(key=lambda g: (not _narrative(g),
                             -(g.get("importance", 0)),
                             -g.get("score", {}).get("total", 0)))
    return [{"gid": g["group_id"], "label": g["label"], "text": g["rep_text"]}
            for g in cand[:max_n]]


def build_overview(groups: list[dict[str, Any]], title: str) -> str:
    """LLM 生成全文概述（研究者向，忠于论文）。"""
    mats = collect_materials(groups)
    rows = llm.chat_json(OVERVIEW_SYSTEM, build_overview_user(title, mats),
                         temperature=0.0)
    if isinstance(rows, dict):
        return str(rows.get("overview", "") or "").strip()
    return str(rows).strip()


def collect_guide_materials(groups: list[dict[str, Any]], max_n: int = 18) -> list[dict[str, Any]]:
    """导读材料：核心主张+关键方法/支撑结果，叙述层优先。"""
    cand = [g for g in groups
            if g.get("label") in ("core_claim", "result_primary",
                                  "result_supporting", "method_core")
            and g.get("importance", 0) >= 4]
    cand.sort(key=lambda g: (not _narrative(g),
                             -(g.get("importance", 0)),
                             -g.get("score", {}).get("total", 0)))
    return [{"gid": g["group_id"], "label": g["label"], "text": g["rep_text"]}
            for g in cand[:max_n]]


def build_guide(groups: list[dict[str, Any]], title: str) -> str:
    """LLM 生成面向小白的通俗导读。"""
    mats = collect_guide_materials(groups)
    rows = llm.chat_json(GUIDE_SYSTEM, build_guide_user(title, mats),
                         temperature=0.0)
    if isinstance(rows, dict):
        return str(rows.get("guide", "") or "").strip()
    return str(rows).strip()


# ── 渲染 ───────────────────────────────────────────


def _rep(g: dict[str, Any], n: int = 160) -> str:
    return g.get("rep_text", "").replace("\n", " ")[:n]


def _evmark(ev: str) -> str:
    return _EV.get(ev, "⚠")


def render_report(groups: list[dict[str, Any]], hubs: list[dict[str, Any]],
                  overview: str, guide: str, meta: dict[str, Any],
                  figures: list[dict[str, Any]] | None = None) -> str:
    title = meta.get("title") or meta.get("pdf", "")
    pdf = meta.get("pdf", "")
    n_claims = meta.get("n_claims", 0)
    n_edges = sum(len(h.get("edges", [])) for h in hubs)
    by_id = {g["group_id"]: g for g in groups}
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> 原始 claims {n_claims} 条 → 去重 {len(groups)} 组 → "
                 f"核心主张 {len(hubs)} 个（{n_edges} 条论证关系） ｜ {pdf}")
    lines.append("")

    # 导读（小白向）
    if guide:
        lines.append("## 👋 一分钟导读（面向新手）")
        lines.append("")
        lines.append(guide)
        lines.append("")
        lines.append("---")
        lines.append("")

    # 概述
    lines.append("## 📋 概述")
    lines.append(overview or "_（概述生成失败）_")
    lines.append("")

    # 核心要点（Top8 快速浏览）
    top = sorted([g for g in groups if g.get("importance", 0) >= 5],
                 key=lambda g: g["score"].get("total", 0), reverse=True)[:8]
    lines.append(f"## 🎯 核心要点 Top{len(top)}")
    for g in top:
        p = (g.get("pages") or [])[:1]
        lines.append(f"- **{g['importance']}** [{g['label']}] {_rep(g, 120)}"
                     f"　`{_evmark(g.get('ev_state', ''))} p{','.join(map(str, p))}`")
    lines.append("")

    # 论证骨架
    lines.append("## 🕸️ 论证骨架")
    lines.append("_每个核心主张展开它在这个论证里的位置：什么机制实现它、"
                 "哪些实验支撑它、受什么限制、与谁对照。_")
    lines.append("")
    for h in hubs:
        lines.append(f"### {h['rep_text'][:110]}")
        lines.append(f"*{h['label']} · {h['importance']} 分*")
        if not h.get("edges"):
            reason = h.get("isolated_reason", "")
            note = {
                "no_sources": "独立主张：论文中缺少可与之关联的方法/结果/局限等主张类型",
                "no_edges": "独立主张：多次尝试后仍无直接关联项（多为总述句或定理句）",
            }.get(reason, "独立主张 · 无直接关联项")
            lines.append(f"_{note}_")
        for rel in ("implements", "supports", "limits", "contrasts"):
            es = [e for e in h.get("edges", []) if e["relation"] == rel]
            if not es:
                continue
            lines.append(f"**{_REL_CN[rel]}（{rel}）**")
            for e in es:
                src = by_id.get(e["source"], {})
                st = src.get("rep_text", "").replace("\n", " ")[:90]
                why = f"—— {e['why']}" if e.get("why") else ""
                lines.append(f"- `{e['source']}` {st}　{why}"
                             f"（p{e.get('page', '?')} {_evmark(e.get('ev', ''))}）")
        lines.append("")
    lines.append("")

    # 图表一览
    if figures:
        lines.append(f"## 🖼 图表一览（{len(figures)} 个）")
        lines.append("_论文中的 Figure/Table 及其解读（视觉细节请查看原文对应页）_")
        lines.append("")
        for f in figures:
            cap = " ".join(f.get("caption", "").split())[:95]
            g = (f.get("guide") or "").strip()
            lines.append(f"- **{f['id']}**（p{f['page']}）：{cap}")
            if g:
                lines.append(f"　↳ {g}")
        lines.append("")

    # 局限
    lims = sorted([g for g in groups if g.get("label") == "limitation"],
                  key=lambda g: g.get("importance", 0), reverse=True)
    if lims:
        lines.append("## ⚠️ 局限")
        for g in lims:
            p = (g.get("pages") or [])[:2]
            lines.append(f"- **{g['importance']}** {_rep(g, 150)}"
                         f"　`p{','.join(map(str, p))}`")
        lines.append("")

    # 章节精读
    lines.append("## 📖 章节精读")
    sec_ord: list[str] = []
    for g in groups:
        s = g["sections"][0] if g.get("sections") else "(无标题)"
        if s not in sec_ord:
            sec_ord.append(s)
    for s in sec_ord:
        gs = [g for g in groups
              if (g["sections"][0] if g.get("sections") else "(无标题)") == s]
        show = [g for g in gs if g.get("importance", 0) >= 4]
        rest = [g for g in gs if g.get("importance", 0) < 4]
        head = f"### {s}" + (f"　·　另 {len(rest)} 条细节" if rest else "")
        lines.append(head)
        for g in sorted(show, key=lambda g: (g["importance"],
                                             g["score"].get("total", 0)),
                        reverse=True):
            lines.append(f"- **{g['importance']}** [{g['label']}] {_rep(g, 130)}"
                         f"　`{_evmark(g.get('ev_state', ''))}`")
        lines.append("")
    lines.append("")

    # 全部主张
    lines.append("## 🗂 全部主张（按重要性）")
    for g in sorted(groups, key=lambda g: -g.get("importance", 0)):
        lines.append(f"- `{g['group_id']}` **{g['importance']}** [{g['label']}] "
                     f"{_rep(g, 100)}")
    return "\n".join(lines)
