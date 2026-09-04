"""Figures：识别论文图表 caption 并打包"图上下文"（方案1）与读图指南（方案2 非多模态）。

方案1：从 PDF 文本块识别 Figure/Table caption，并把正文引用该图/表的段落收集起来，
        组成可检索的"图上下文"（RAG 检索单元 + 报告"图表一览"）。
方案2：LLM 不看图，基于 caption + 正文引用段生成 60~120 字"读图指南"。
"""
from __future__ import annotations

import re
from typing import Any

from paperpilot.prompts.figures import GUIDE_SYSTEM, build_guide_user
from paperpilot.tools import llm

def _clean(text: str) -> str:
    return " ".join(text.split())


# 编号允许：4a / 3.1 / A.1 / C.1 / 10（章节编号或子图后缀）
_CAP_HEAD_RE = re.compile(
    r"^(?:Table|Tab\.?|Figure|Fig\.?)\s+"
    + r"(?:[A-Za-z]{1,2}\.\d+|[A-Za-z]?\d+(?:\.\d+)*[a-z]?)\s*",
    re.I)


def _caption_kind(text: str) -> tuple[str, str] | None:
    """若块是 caption 返回 (kind, num)，否则 None。

    判定：编号后可接 冒号/句点（"Table 2: …"），或空格+大写标题
    （"Table 1 Million-scale…"）；正文句如 "Table 6 lists…" 后接小写
    动词，不算 caption。
    """
    raw = text.strip()
    m = _CAP_HEAD_RE.match(raw)
    if not m:
        return None
    kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
    num = m.group(2)
    rest = raw[m.end():].lstrip()
    if not rest:
        return None
    if rest[0] in ":.":
        return kind, num
    if rest[0].isupper() or rest[0].isdigit():
        return kind, num
    return None


def extract_figures(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 PDF 文本块识别图表 caption + 收集正文引用段。

    Returns: [{id, kind, page, caption, refs}]
      id 形如 "Figure 4a" / "Table 2" / "Figure C.1"
      refs 为正文中提到该编号的块文本（截断保留整段）
    """
    figs: list[dict[str, Any]] = []
    for b in blocks:
        got = _caption_kind(b["text"])
        if got is None:
            continue
        kind, num = got
        fig_id = f"{kind} {num}"
        figs.append({"id": fig_id, "kind": kind.lower(), "num": num,
                     "page": b["page"], "caption": _clean(b["text"]),
                     "refs": []})

    # 收集正文引用段（编号后允许子图后缀 a/b，如 Figure 4a 引用归到 Figure 4）
    for fig in figs:
        num = fig["num"]
        escaped = re.escape(num)
        pat = re.compile(
            rf"\b(?:Figure|Fig\.?|Table|Tab\.?)\s*{escaped}[a-z]?\b", re.I)
        prefix = f"{fig['kind'].title()} {fig['num']}"
        for b in blocks:
            if b["page"] < fig["page"] - 1:  # 引用一般在图后
                continue
            t = b["text"]
            if not pat.search(t):
                continue
            if t.startswith(prefix):  # 跳过 caption 自身块
                continue
            ref = _clean(t)
            if ref and len(ref) > 15 and ref not in fig["refs"]:
                fig["refs"].append(ref)
    return figs


def build_context(fig: dict[str, Any], max_refs: int = 3, ref_cap: int = 300) -> str:
    """图上下文：caption + 前几段正文引用（RAG 单元/读图指南原料）。"""
    parts = [f"caption: {fig['caption']}"]
    refs = fig.get("refs", [])
    if refs:
        parts.append("正文引用：")
        for r in refs[:max_refs]:
            parts.append(r[:ref_cap])
    return "\n".join(parts)


def generate_guides(figs: list[dict[str, Any]]) -> dict[str, str]:
    """LLM 基于 caption+正文引用生成每图"读图指南"（一次调用，非多模态）。"""
    if not figs:
        return {}
    rows = None
    for attempt in range(2):
        try:
            rows = llm.chat_json(GUIDE_SYSTEM, build_guide_user(figs),
                                 temperature=0.0)
            break
        except (llm.LLMError, ValueError) as e:
            if attempt == 0:
                print(f"  [figures] 读图指南调用失败，重试: {e}")
            else:
                print(f"  [figures] 读图指南重试仍失败: {e}")
    if not isinstance(rows, list):
        return {}
    known = {f["id"] for f in figs}
    out: dict[str, str] = {}
    for r in rows:
        if isinstance(r, dict) and str(r.get("id", "")) in known:
            g = str(r.get("guide", "")).strip()
            if g:
                out[str(r["id"])] = g
    return out
