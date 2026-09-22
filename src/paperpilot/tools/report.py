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

import re
from typing import Any

from paperpilot.prompts.report import (GUIDE_SYSTEM, OVERVIEW_SYSTEM,
                                       build_guide_user, build_overview_user)
from paperpilot.tools import llm

_EV = {"hit": "✓", "loose": "~", "miss": "⚠"}
_REL_CN = {"implements": "实现机制", "supports": "证据支撑",
           "limits": "边界限制", "contrasts": "对照区分"}


class MaterialEmpty(RuntimeError):
    """生成材料为空（无 importance≥4 的核心主张）。

    多为**打标失败**导致全篇退化成 detail（见 `viewer.label_groups`）——
    此时 LLM 无据可依。必须让失败**显性**，因为两条路都是静默失败：
      · 概述：LLM 会直接拒答，而**拒答话术会被当成概述存盘**；
      · 导读：prompt 允许"背景可以来自常识"，LLM 会**凭标题编一段**
        看似正常、实则无出处的文字 —— 比拒答更隐蔽。
    """

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}材料为空：{detail}")
        self.kind = kind
        self.detail = detail


# LLM 在"没有材料"时的拒答话术特征（防线：旧产物/其他路径漏进来时也能识别）
_REFUSAL_PAT = re.compile(
    r"未提供|无法撰写|无法生成|无法完成|请补充|材料不足|缺少.{0,6}材料|"
    r"没有.{0,6}材料|无法据以|不足以撰写")
_MIN_OVERVIEW_LEN = 40      # 概述硬性要求 90~140 字；低于此值必是异常


def degraded_reason(text: str) -> str:
    """概述文本是否为**无效产物**（空 / 拒答话术 / 过短）。返回原因，正常返回 ""。"""
    t = (text or "").strip()
    if not t:
        return "概述为空"
    if _REFUSAL_PAT.search(t[:80]):
        return f"概述是拒答话术（非论文内容）：{t[:48]}…"
    if len(t) < _MIN_OVERVIEW_LEN:
        return f"概述过短（{len(t)} 字 < {_MIN_OVERVIEW_LEN}）"
    return ""


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


def _pick(groups: list[dict[str, Any]], labels: tuple[str, ...], floor: float,
          max_n: int) -> list[dict[str, Any]]:
    cand = [g for g in groups
            if g.get("label") in labels and g.get("importance", 0) >= floor]
    cand.sort(key=lambda g: (not _narrative(g),
                             -(g.get("importance", 0)),
                             -g.get("score", {}).get("total", 0)))
    return [{"gid": g["group_id"], "label": g["label"], "text": g["rep_text"]}
            for g in cand[:max_n]]


# 材料分级放宽（2026-09-22）：严格档为空时逐级放宽，避免"确实没有 headline
# 结论"的论文被误判为打标失败；走到最后一档仍为空，才判定是真失败
# （全篇 claims 都退化成 detail、importance≤3 —— 见 `viewer.label_groups`）。
_CORE = ("core_claim", "result_primary")
_CORE_METHOD = ("core_claim", "result_primary", "method_core", "result_supporting")
_STRICT6 = _CORE_METHOD + ("ablation", "limitation")

OVERVIEW_LADDER: list[tuple[str, tuple[str, ...], float]] = [
    ("严格", _CORE, 4),
    ("放宽·核心不限分", _CORE, 0),
    ("放宽·含方法/次级结果", _CORE_METHOD + ("ablation",), 3),
]
GUIDE_LADDER: list[tuple[str, tuple[str, ...], float]] = [
    ("严格", _CORE_METHOD, 4),
    ("放宽·含消融/局限", _STRICT6, 4),
    ("放宽·六类不限分", _STRICT6, 3),
]
# ⚠️ 两把梯子的**最后一档都不允许 detail / related**：它们按 prompt 定义是
# "说明性、非主张"，拿来写概述/导读会把背景句包装成论文结论。


def _walk(groups: list[dict[str, Any]], ladder: list[tuple[str, tuple[str, ...], float]],
          max_n: int) -> tuple[list[dict[str, Any]], str]:
    for name, labels, floor in ladder:
        mats = _pick(groups, labels, floor, max_n)
        if mats:
            return mats, name
    return [], ladder[-1][0]


def collect_materials(groups: list[dict[str, Any]], max_n: int = 12
                      ) -> tuple[list[dict[str, Any]], str]:
    """概述材料：摘要/引言/结论里的高分核心主张（优先叙述层）。

    Returns: (材料, 命中的档位名)。**材料为空 = 所有档位都取不到 → 打标失败。**
    """
    return _walk(groups, OVERVIEW_LADDER, max_n)


def build_overview(groups: list[dict[str, Any]], title: str) -> str:
    """LLM 生成全文概述（研究者向，忠于论文）。无可用材料 → 抛 `MaterialEmpty`。"""
    mats, level = collect_materials(groups)
    if not mats:
        raise MaterialEmpty(
            "概述", f"{len(groups)} 组在所有档位都取不到材料（全篇 label=detail、"
                    "importance≤3：既可能是打标失败，也可能是该篇确实无核心主张）")
    if level != "严格":
        print(f"  [概述] 材料按「{level}」档放宽取得 {len(mats)} 条")
    rows = llm.chat_json(OVERVIEW_SYSTEM, build_overview_user(title, mats),
                         temperature=0.0)
    if isinstance(rows, dict):
        return str(rows.get("overview", "") or "").strip()
    return str(rows).strip()


def collect_guide_materials(groups: list[dict[str, Any]], max_n: int = 18
                            ) -> tuple[list[dict[str, Any]], str]:
    """导读材料：核心主张+关键方法/支撑结果，叙述层优先。返回 (材料, 档位名)。"""
    return _walk(groups, GUIDE_LADDER, max_n)


def build_guide(groups: list[dict[str, Any]], title: str) -> str:
    """LLM 生成面向小白的通俗导读。

    ⚠️ 无材料时**抛错**而不是继续（2026-09-22）：`GUIDE_SYSTEM` 允许"背景知识
    来自常识"，所以空材料时 LLM 会**凭标题编一段看似正常的导读**（实测
    `2606.18837` 空材料仍产出 569 字）—— 比概述拒答更隐蔽，必须显性化。
    """
    mats, level = collect_guide_materials(groups)
    if not mats:
        raise MaterialEmpty(
            "导读", f"{len(groups)} 组在所有档位都取不到材料（全篇 label=detail、"
                    "importance≤3：既可能是打标失败，也可能是该篇确实无核心主张）")
    if level != "严格":
        print(f"  [导读] 材料按「{level}」档放宽取得 {len(mats)} 条")
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
