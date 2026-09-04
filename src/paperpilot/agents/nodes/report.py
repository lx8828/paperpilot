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
VIEW_DIR = ROOT / "out_views"


def _load_report(pdf: str) -> dict[str, Any]:
    stem = Path(pdf).stem
    path = VIEW_DIR / f"{stem}.report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺 {path.name}——请先跑 `uv run python cli/main.py {pdf}` 生成报告")
    return json.loads(path.read_text(encoding="utf-8"))


def report_l0(state: QAState) -> dict[str, Any]:
    pdf = state.get("pdf", "")
    report = _load_report(pdf)
    claims = {c["claim_id"]: c for c in report.get("claims", [])}

    core_points: list[dict[str, Any]] = []
    for g in report.get("core_points", []) or []:
        rep = claims.get(g.get("rep_claim_id", "")) or {}
        item = {
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
        }
        core_points.append(item)

    route = list(state.get("route") or [])
    if "L0" not in route:
        route.append("L0")
    debug = dict(state.get("debug") or {})
    debug["l0"] = {"n_core_points": len(core_points)}
    return {
        "title": report.get("title", ""),
        "overview": report.get("overview", ""),
        "core_points": core_points,
        "route": route,
        "debug": debug,
    }
