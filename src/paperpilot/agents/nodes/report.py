"""节点 L0 report_l0：加载 report，产出零检索上下文（overview + core_points）。

L0 是本漏斗的"最便宜入口"：只读 report.json，不加载 embedding 模型。
Judge0 若判够（全貌类问题被 overview/core_points 覆盖），answer 直接用
core_points 作答并溯源（cites 锚在 gid / rep_claim_id → claim evidence/page）。

core_points 产出时增强：从全量 claims 补齐 evidence_quote/page/chunk_id/home_section，
让 L0 answer 的 cites 与 L1/L2 一样能直达原文证据。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperpilot.agents.state import QAState

# agents/nodes/report.py → 项目根
ROOT = Path(__file__).resolve().parents[4]
VIEW_DIR = ROOT / "assets/artifacts/out_views"


def _load_report(pdf: str) -> dict[str, Any]:
    stem = Path(pdf).stem
    path = VIEW_DIR / f"{stem}.report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺 {path.name}——请先跑 `uv run python cli/main.py {pdf}` 生成报告")
    return json.loads(path.read_text(encoding="utf-8"))


L0_PER_PAPER = 4      # 多篇时**每篇**取几条核心要点（合并后要控制上下文规模）


def _corpus(state: QAState) -> list[str]:
    """语料：**只有一条路径**（`state["pdfs"]` → 多篇语料，1~N 篇）。

    ⚠️ 2026-09-23 删掉单篇路径：原先 `pdfs` 为空时退回 `state["pdf"]`（一篇）。
    现在**没有第二条单篇代码路径** —— 1 篇或 N 篇都走同一个 `MultiChunkIndex`
    + 合并总览，不再有"单篇分支"。

    ⚠️ 2026-09-23 放宽（原为 `len < 2` 直接报错）：**1 篇是合法语料**，只拒绝**空**。
    理由：`len==1` 走的**仍是同一条路径**（`MultiChunkIndex([p])` 与多篇无差别）——
    "没有单篇路径"指的是**代码分支**，不是**语料长度**；而 `< 2` 会连带打死
    web 的单篇上传（`index.html` 只传一篇）与单篇端到端测试。
    """
    pdfs = [str(p) for p in (state.get("pdfs") or []) if p]
    if not pdfs:
        raise ValueError(f"语料不能为空（pdfs 至少 1 篇）；收到 {pdfs!r}")
    return pdfs


def _src_label(pdf: str, report: dict[str, Any]) -> str:
    """论文在提示里的**显示名**（多篇时让 judge/answer 能区分是哪篇）。"""
    t = str(report.get("title") or "").strip()
    return (t if len(t) <= 30 else t[:30] + "…") or Path(pdf).stem


def report_l0(state: QAState) -> dict[str, Any]:
    """L0 上下文（零检索）：**合并总览** overview + core_points。

    语料**只有多篇**（`state["pdfs"]`，≥2 篇）：
        · 每篇概述前加 `〈显示名〉`，开头声明"本次语料共 N 篇"；
        · 核心要点每篇取前 `L0_PER_PAPER` 条，条目带 `src`（显示名）/`pdf`（文件名）。
    ⚠️ 2026-09-23 删掉单篇形态（`multi` 分支）与"指代不明"声明：对着多篇提问必然
    指明篇名或编号，泛指"这篇论文"不是产品用法。
    """
    pdfs = _corpus(state)
    # 问题里**明确指明**的篇（`components.focus.resolve_focus`）→ 收窄 L0 上下文范围。
    # ⚠️ 这是"在多篇语料里定位到某一篇"，**不是**"单篇语料"（语料永远是多篇）。
    focus = [str(p) for p in (state.get("focus") or []) if p]
    if focus and len(focus) < len(pdfs):
        pdfs = [p for p in pdfs if p in focus]
    reports = [(p, _load_report(p)) for p in pdfs]

    overviews: list[str] = []
    core_points: list[dict[str, Any]] = []
    for p, report in reports:
        claims = {c["claim_id"]: c for c in report.get("claims", [])}
        src = _src_label(p, report)
        ov = str(report.get("overview") or "").strip()
        overviews.append(f"〈{src}〉{ov}")

        groups = report.get("core_points") or []
        for g in groups[:L0_PER_PAPER]:
            rep = claims.get(g.get("rep_claim_id", "")) or {}
            core_points.append({
                "gid": g.get("gid", ""),
                "rep_claim_id": g.get("rep_claim_id", ""),
                "label": g.get("label", ""),
                "importance": g.get("importance", 0),
                "text": g.get("text", ""),
                "pages": list(g.get("pages") or []),
                "evidence": rep.get("evidence_quote", ""),
                "page": rep.get("page", 0),
                "chunk_id": rep.get("chunk_id", ""),
                "title_path": list(rep.get("title_path") or []),
                "src": src,
                "pdf": p,
            })

    # ⚠️ 2026-09-23 删：原先这里还有「用户若用「这篇论文」而未指明篇名，属于**指代不明**」。
    # 为什么删：对着多篇语料提问**必然指明篇名或编号**（泛指"这篇论文"不是产品用法）；
    # 注入那句会让 judge_l0 必然判不够（实测 0/15）→ 每个问题都被迫下钻、L0 直答形同废掉。
    head = f"（本次语料共 {len(pdfs)} 篇论文。每条概述/要点前〈〉内标明它属于哪一篇。）"
    overview = head + "\n\n" + "\n\n".join(overviews)
    title = "；".join(_src_label(p, r) for p, r in reports)

    route = list(state.get("route") or [])
    if "L0" not in route:
        route.append("L0")
    debug = dict(state.get("debug") or {})
    debug["l0"] = {"n_core_points": len(core_points), "n_papers": len(pdfs)}
    return {
        "title": title,
        "overview": overview,
        "core_points": core_points,
        "route": route,
        "debug": debug,
    }
