"""跨篇分析：N 张**成果卡** → **1 次 LLM** → 分析报告。

## 为什么不做"对照表"（2026-09-21 决策）

N 篇（≤5）的成果卡合计约 30 KB ≈ 8K token，**全部放得进上下文** ——
所以不需要预先把它们对齐成表：按 `(指标, 数据集, 模型)` 分组的机器
**算出来的东西 LLM 自己就会做**，而且 LLM 还能看到表里没有的"路线分岔"。
（实测：3 篇同主题论文按 `(指标, 数据集)` 对齐的结果是 **0 行** —— 表连内容都填不满。）

**但成果层不能省**：原始 `report.json` 449 KB/篇，5 篇 **2.25 MB 塞不进**上下文；
成果卡压到 6 KB/篇 —— **这才是"放得进上下文"的原因**。

## 本模块的职责（只剩两件）

1. `render_cards()`：把成果卡装配成**给 LLM 的输入**（带可比性标记与锚点）；
2. `compare_papers()`：**一次** LLM 调用拿分析，后端校验引用合法性后渲染 Markdown。

> `to_markdown()` 里没有数值对照表 —— 数值比较由 LLM 在分析正文里做，
> 且 prompt 已明确"自创指标 `~` 不可跨篇比大小、不同数据集的数值不可直接比"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from paperpilot.outcome import PaperOutcome, build_outcome
from paperpilot.prompts.compare import COMPARE_SYSTEM, build_user
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[2]
COMPARE_DIR = ROOT / "assets/artifacts/out_compare"
MAX_ENTRY = 6               # 每个维度最多列几篇（防御 LLM 灌水）


# ── 装配输入 ─────────────────────────────────────────────────────────────────

def render_cards(outcomes: dict[str, PaperOutcome]) -> str:
    """N 张成果卡 → LLM 输入文本。

    每篇压成一小段：问题 / 方法族 / 数据集 / 指标（**带 `=` `~` 可比性标记、
    数据集、模型、锚点**）/ 主要结论 / 局限。数值只给**已核**的。
    """
    blocks = []
    for i, (pdf, o) in enumerate(outcomes.items(), 1):
        L = [f"### 论文 {i}：{o.title or pdf}", f"文件名（引用时用这个）：`{pdf}`", ""]
        if o.problem:
            L.append(f"- 研究问题：{o.problem}")
        if o.method_family:
            L.append(f"- 方法族：{', '.join(o.method_family)}")
        ds = [d.name for d in o.good_datasets] or [d.name for d in o.datasets]
        if ds:
            L.append(f"- 数据集：{', '.join(ds)}")
        if o.metrics:
            L.append("- 指标（`=` 公认可跨篇比 ｜ `~` 自创，**仅本篇内可比**）：")
            for m in o.metrics:
                if not m.verified:
                    continue                      # 未核实的数值不给 LLM 引用
                tag = "=" if m.comparable else "~"
                bits = [f"= {m.value}"]
                if m.baseline:
                    bits.append(f"（基线 {m.baseline}，Δ {m.delta or '?'}）")
                if m.dataset:
                    bits.append(f"@ {m.dataset}")
                if m.model:
                    bits.append(f"[{m.model}]")
                anchor = f"  锚⟨{m.anchor_chunk} p{m.anchor_page}⟩" if m.anchor_chunk else ""
                if m.verify == "values":       # 弱档：数值确在原文，但未能逐字
                    anchor += "（数值同块，未逐字命中）"
                L.append(f"    - `{tag}` {m.name} " + " ".join(bits) + anchor)
        if o.key_findings:
            L.append("- 主要结论：")
            L += [f"    - {f.text}" for f in o.key_findings]
        if o.limitations:
            L.append("- 局限：")
            L += [f"    - {f.text}" for f in o.limitations]
        blocks.append("\n".join(L))
    return "\n\n".join(blocks)


# ── 结果 ─────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    pdfs: list[str] = field(default_factory=list)
    outcomes: dict[str, PaperOutcome] = field(default_factory=dict)
    data: dict = field(default_factory=dict)         # LLM 原始结构（已校验）
    meta: dict = field(default_factory=dict)

    def label(self, pdf: str) -> str:
        """短标题（给正文引用用）。"""
        t = self.outcomes.get(pdf)
        return (t.title if t and t.title else pdf).replace(".pdf", "")[:48]

    def to_markdown(self) -> str:
        d = self.data
        L = [f"# 跨篇分析（{len(self.pdfs)} 篇）", ""]
        for p in self.pdfs:
            L.append(f"- `{p}` — {self.label(p)}")
        L += [
            "",
            "> 分析基于**成果卡**：卡片里每个数值都已逐字回溯到论文原文（严格校验）。",
            "> `~` 标记的论文自创指标**不能跨篇比大小**；不同数据集上的数值也不可直接比。",
            "",
        ]
        if self.meta.get("degraded"):
            L += [f"> ⚠️ {self.meta['degraded']}", ""]

        if d.get("overview"):
            L += ["## 总览", "", str(d["overview"]).strip(), ""]

        dims = d.get("dimensions") or []
        if dims:
            L += ["## 逐维度对比", ""]
            for dim in dims[:8]:
                if not isinstance(dim, dict):
                    continue
                L += [f"**{dim.get('dim') or '（未命名维度）'}**", ""]
                for e in (dim.get("entries") or [])[:MAX_ENTRY]:
                    if not isinstance(e, dict):
                        continue
                    pdf = str(e.get("pdf") or "")
                    who = self.label(pdf) if pdf in self.outcomes else (pdf or "?")
                    L.append(f"- **{who}**：{str(e.get('point') or '').strip()}")
                L.append("")

        st = d.get("strengths") or []
        if st:
            L += ["## 优势与劣势（相对其它篇）", ""]
            for s in st[:MAX_ENTRY]:
                if not isinstance(s, dict):
                    continue
                pdf = str(s.get("pdf") or "")
                L.append(f"**{self.label(pdf) if pdf in self.outcomes else (pdf or '?')}**"
                         f"  (`{pdf}`)")
                for x in (s.get("strong") or []):
                    L.append(f"- 👍 {str(x).strip()}")
                for x in (s.get("weak") or []):
                    L.append(f"- 👎 {str(x).strip()}")
                L.append("")

        if d.get("advice"):
            L += ["## 建议", "", str(d["advice"]).strip(), ""]

        meta = self.meta
        L += ["---",
              f"<sub>参与分析 {len(self.pdfs)} 篇 ｜ LLM 调用 {meta.get('calls', 0)} 次 "
              f"｜ {meta.get('prompt_tokens', 0):,} + {meta.get('completion_tokens', 0)} token "
              f"｜ {meta.get('seconds', 0)}s"
              + (f" ｜ 丢弃非法引用 {meta['dropped_refs']} 条" if meta.get("dropped_refs") else "")
              + "</sub>"]
        return "\n".join(L)


# ── 主入口 ───────────────────────────────────────────────────────────────────

def compare_papers(pdfs: list[str], *, question: str = "", build: bool = True,
                   force: bool = False, verbose: bool = True) -> AnalysisResult:
    """N 篇 → 一次 LLM 分析。

    Args:
        pdfs: `assets/papers/` 下的文件名（2~5 篇为宜）。
        question: 研究者关注点（可选，会写进 prompt 引导分析角度）。
        build: True 时缺成果层就生成（调 LLM）；False 缺了直接报错。

    Returns:
        `AnalysisResult`。LLM 失败时抛 `llm.LLMError`。
    """
    log = print if verbose else (lambda *a: None)
    a = AnalysisResult(pdfs=list(pdfs))
    for p in pdfs:
        a.outcomes[p] = (build_outcome(p, force=force, verbose=verbose) if build
                         else _require_cached(p))

    cards = render_cards(a.outcomes)
    log(f"[compare] {len(pdfs)} 篇 → 卡片 {len(cards):,} 字符（≈{len(cards) // 4:,} token）")

    llm.reset_usage()
    import time
    t0 = time.time()
    from paperpilot.tools import mock_llm
    raw = (_mock_analysis(a) if mock_llm.llm_enabled()
           else llm.chat_json(COMPARE_SYSTEM, build_user(cards, question),
                              temperature=0.2, max_tokens=3000))
    a.meta["seconds"] = round(time.time() - t0, 1)
    a.meta["dropped_refs"] = 0
    a.data = _coerce(raw, a)

    u = llm.usage_stats()
    a.meta.update(calls=u["calls"], prompt_tokens=u["prompt_tokens"],
                  completion_tokens=u["completion_tokens"])
    log(f"[compare] 完成 {a.meta['seconds']}s | 调用 {u['calls']} 次 | "
        f"{u['prompt_tokens']:,} + {u['completion_tokens']} token"
        + (f" | 丢弃非法引用 {a.meta['dropped_refs']}" if a.meta["dropped_refs"] else ""))
    return a


def _require_cached(pdf: str) -> PaperOutcome:
    from paperpilot.outcome import load_outcome
    o = load_outcome(pdf)
    if o is None:
        raise FileNotFoundError(f"缺成果层缓存：{pdf}（先跑 cli/run_outcome.py）")
    return o


def _coerce(raw: object, a: AnalysisResult) -> dict:
    """防御性收敛 LLM 输出：**丢弃引用了不存在论文的条目**并计数。"""
    d = raw if isinstance(raw, dict) else {}
    valid = set(a.pdfs)

    def fix_pdf(x: object) -> str | None:
        s = str(x or "").strip()
        if s in valid:
            return s
        # 宽容：允许不吃 `.pdf` 后缀 / 带路径
        for v in valid:
            if s and (s in v or v.replace(".pdf", "") in s):
                return v
        a.meta["dropped_refs"] = a.meta.get("dropped_refs", 0) + 1
        return None

    dims = []
    for dim in (d.get("dimensions") or []):
        if not isinstance(dim, dict):
            continue
        ent = []
        for e in (dim.get("entries") or []):
            if not isinstance(e, dict):
                continue
            pdf = fix_pdf(e.get("pdf"))
            if pdf and str(e.get("point") or "").strip():
                ent.append({"pdf": pdf, "point": str(e["point"]).strip()})
        if ent:
            dims.append({"dim": str(dim.get("dim") or "").strip(), "entries": ent})

    st = []
    for s in (d.get("strengths") or []):
        if not isinstance(s, dict):
            continue
        pdf = fix_pdf(s.get("pdf"))
        if not pdf:
            continue
        strong = [str(x).strip() for x in (s.get("strong") or []) if str(x).strip()]
        weak = [str(x).strip() for x in (s.get("weak") or []) if str(x).strip()]
        if strong or weak:
            st.append({"pdf": pdf, "strong": strong[:5], "weak": weak[:5]})

    out = {"overview": str(d.get("overview") or "").strip(),
           "dimensions": dims, "strengths": st,
           "advice": str(d.get("advice") or "").strip()}
    if not (out["overview"] or dims or st or out["advice"]):
        out["overview"] = "（LLM 未返回可解析的分析内容）"
        a.meta["degraded"] = "LLM 返回内容无法解析为预期结构，请检查 prompt 或重试"
    return out


def _mock_analysis(a: AnalysisResult) -> dict:
    """演示模式：不联网，按卡片结构拼一份占位分析。"""
    return {
        "overview": "（演示模式）本分析未调用 LLM，仅按成果卡结构占位。",
        "dimensions": [{
            "dim": "研究问题",
            "entries": [{"pdf": p, "point": o.problem or "—"}
                        for p, o in a.outcomes.items()],
        }],
        "strengths": [{"pdf": p, "strong": [o.key_findings[0].text] if o.key_findings else [],
                       "weak": [o.limitations[0].text] if o.limitations else []}
                      for p, o in a.outcomes.items()],
        "advice": "（演示模式）",
    }


def save(a: AnalysisResult, name: str = "") -> Path:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    name = name or f"analysis_{len(a.pdfs)}p.md"
    p = COMPARE_DIR / name
    p.write_text(a.to_markdown(), encoding="utf-8")
    (COMPARE_DIR / name.replace(".md", ".json")).write_text(
        json.dumps({"pdfs": a.pdfs, "meta": a.meta, "analysis": a.data},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return p


__all__ = ["AnalysisResult", "COMPARE_DIR", "compare_papers", "render_cards", "save"]
