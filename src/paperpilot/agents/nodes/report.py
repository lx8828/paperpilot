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


# 多篇时**每篇**取几条核心要点。= 8 即"**全取**"：全语料 466 篇的 core_points **上限就是 8**
# （分布 {3:58,4:66,5:68,6:74,7:53,8:108}），n≥8 与 n=8 材料完全相同 → 8 是有效上限。
#
# 为什么默认改成 8（2026-09-24 实测，两组 A 组 L0 题共 30 道）：
#   · **更准**：`全取+正常路由` 27/30 (90%) vs `全取+全部走检索` 24/30 (80%)
#     —— 13 道分叉题上直答更好 3 题、更差 **0** 题；L0 材料是"已验证的要点摘要"，
#     这类"全貌/总结"题它比检索回的原文块更命中。
#   · **更省**：直答 prompt ~12.3k/题 vs 检索 ~18.0k/题（省 32%）；整臂 476k vs 545k tokens。
#     → "全部去检索"并不便宜。
#   · **上下文变长无害**：材料 2.5k→4.1k 字（entries 20→64），直答 prompt 仍 ~12k，
#     远低于检索路径 —— 要点本来就短。
#   · **不误伤**：n=4→8 对 `single`/`table` 题（每组 50 道，本该走检索）判够数**完全不变**
#     （80 道里 0 道翻转），`negative` 仅 group1 有 1 道翻转（且有"缺失断言复核闸门"兜底）；
#     L0 类共 +2 道转为直答。
L0_PER_PAPER = 8
# 多篇时**每篇**取几条「局限 / 未来方向」（2026-09-24 新增）。
# 为什么必须有：单篇时代 L0 材料含**全部** limitations（平均 6.6 条/篇），
# 多篇合并此前**完全不带** → 凡"局限/未来方向"类题在多篇下**结构性缺料**
# （实测 `8837-L0-3`：判不够下钻检索后仍只答出"未来方向"、漏掉"局限"；
#  全量摸底「局限/未来」类题 11/180，其中 3 道是 L0 类）。
# 另：`retrieval/scripts/_validate_group_questions.py` 一直在读
# `report_l0(...)["limitations"]`，但该字段从来没被返回过（恒为空）—— 本改动同时接上它。
#
# 配额取值（group2 的 A 组 L0 题 15 道实测，2026-09-24）：
#   无局限  合计 14/15 ｜ 免检索 4 题(直答 4/4) ｜ 材料 2488 字
#   lim=4   合计 14/15 ｜ 免检索 7 题(直答 6/7) ｜ 材料 3057 字   ← `0934-L0-3` 的锚点
#                                                               `ALFWorld` 出自该篇第 6 条局限，漏取 → 直答答错
#   lim=8   合计 14/15 ｜ 免检索 7 题(直答 7/7) ｜ 材料 3389 字   ← **取默认**
# 结论：`8` ≈ 覆盖平均 6.6 条/篇的全部局限，免检索率 27%→47%（省 3 次检索）而正确率持平。
L0_LIM_PER_PAPER = 8


def _env_int(name: str, default: int) -> int:
    """`PAPERPILOT_<name>` 覆盖（扫描 / A-B 用）；非法值回落默认。"""
    import os
    try:
        return int(os.environ.get(f"PAPERPILOT_{name}") or default)
    except ValueError:
        return default


def _l0_per_paper() -> int:
    """每篇核心要点条数（`PAPERPILOT_L0_PER_PAPER`）。"""
    return _env_int("L0_PER_PAPER", L0_PER_PAPER)


def _l0_lim_per_paper() -> int:
    """每篇局限条数（`PAPERPILOT_L0_LIM_PER_PAPER`；=0 退回"不带局限"的旧行为）。"""
    return _env_int("L0_LIM_PER_PAPER", L0_LIM_PER_PAPER)


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


def _material_entry(g: dict[str, Any], claims: dict[str, Any],
                    src: str, pdf: str) -> dict[str, Any]:
    """`report.json` 的 core_points/limitations 条目 → L0 材料条目。

    两者字段**同构**（`gid/rep_claim_id/label/importance/text/pages`），差别只在 `label`
    （`core_claim` vs `limitation`）；这里统一从全量 `claims` 补齐
    `evidence/page/chunk_id/title_path`，并注入 `src`（显示名）与 `pdf`（cites 定位哪一篇）。
    """
    rep = claims.get(g.get("rep_claim_id", "")) or {}
    return {
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
        "pdf": pdf,
    }


def report_l0(state: QAState) -> dict[str, Any]:
    """L0 上下文（零检索）：**合并总览** overview + core_points + limitations。

    语料**只有多篇**（`state["pdfs"]`，≥2 篇）：
        · 每篇概述前加 `〈显示名〉`，开头声明"本次语料共 N 篇"；
        · 核心要点每篇取前 `L0_PER_PAPER` 条，条目带 `src`（显示名）/`pdf`（文件名）；
        · 局限/未来方向每篇取前 `L0_LIM_PER_PAPER` 条（2026-09-24 新增，见常量注释）。
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
    limitations: list[dict[str, Any]] = []
    for p, report in reports:
        claims = {c["claim_id"]: c for c in report.get("claims", [])}
        src = _src_label(p, report)
        ov = str(report.get("overview") or "").strip()
        overviews.append(f"〈{src}〉{ov}")

        groups = report.get("core_points") or []
        for g in groups[: _l0_per_paper()]:
            core_points.append(_material_entry(g, claims, src, p))

        # 局限 / 未来方向（字段与 core_points 同构，`label="limitation"`）。
        # 单篇时代 L0 材料含全部 limitations；多篇合并此前完全不带 → 见文件头常量注释。
        for g in (report.get("limitations") or [])[: _l0_lim_per_paper()]:
            limitations.append(_material_entry(g, claims, src, p))

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
    debug["l0"] = {"n_core_points": len(core_points),
                   "n_limitations": len(limitations), "n_papers": len(pdfs)}
    return {
        "title": title,
        "overview": overview,
        "core_points": core_points,
        "limitations": limitations,
        "route": route,
        "debug": debug,
    }
