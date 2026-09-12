"""Pipeline 编排层：一个 PDF → 各环节产物 → report.json 一步完成。

把散在 run_claims / run_view / run_skeleton / run_figures / run_report
里的流程逻辑收敛为可编程调用的 stage 函数，供前端与问答层复用。

用法（作为库）：
    from paperpilot.pipeline import process_pdf
    report = process_pdf("2609.00859v1.pdf")          # 有缓存则复用，缺则生成
    report = process_pdf("x.pdf", force=True)          # 全链路重跑（需 LLM）
    report = process_pdf("x.pdf", skip_llm=True)       # 只装配已有产物，缺则报错

产物：
    assets/artifacts/out_claims/<stem>.claims.json      claims + 溯源锚点
    assets/artifacts/out_views/<stem>.summary.json      主张组（label/importance/score）
    assets/artifacts/out_views/<stem>.skeleton.json     论证骨架
    assets/artifacts/out_views/<stem>.figures.json      图表 + 读图指南
    assets/artifacts/out_views/<stem>.overview.json     / .guide.json / .report.md / .md
    assets/artifacts/out_views/<stem>.report.json       最终结构化报告（PaperReport）
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.models.schema import (Claim, ClaimGroup, EdgeView, GroupBrief,
                                      HubView, PaperReport, SectionView)
from paperpilot.qasper_source import build_chunks, load_papers
from paperpilot.tools import analyzer, llm
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.evidence import (claim_to_dict, dict_to_claim,
                                       evidence_state, verify_evidence)
from paperpilot.tools.figures import extract_figures, generate_guides
from paperpilot.tools.mineru_bridge import (figures_from_mineru_dir,
                                            title_from_mineru_dir)
from paperpilot.tools.pdf_parser import parse_pdf
from paperpilot.tools.report import (build_guide, build_overview,
                                     render_report)
from paperpilot.tools.skeleton import build_skeleton
from paperpilot.tools.viewer import (dedupe_groups, label_groups,
                                     render_markdown, score_groups)

ROOT = Path(__file__).resolve().parents[2]  # src/paperpilot/pipeline.py → 项目根
PAPERS_DIR = ROOT / "assets" / "papers"
CLAIMS_DIR = ROOT / "assets/artifacts/out_claims"
VIEW_DIR = ROOT / "assets/artifacts/out_views"
MINERU_OUT = ROOT / "assets/artifacts/out_mineru"   # MinerU 解析产物根（current_source=='mineru' 时读取）

MAX_LEN = 4000


# ───────────────────────── 基础 IO ─────────────────────────


def _stem(pdf_name: str) -> str:
    return Path(pdf_name).stem


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _paper_title(pdf_path: Path) -> str:
    try:
        return str(parse_pdf(str(pdf_path)).get("metadata", {}).get("title", "")).strip()
    except Exception:  # noqa: BLE001
        return ""


def _payload_source(pdf_name: str) -> str:
    """产物打标：该 pdf 当前解析源（qasper/mineru/pymupdf）。"""
    from paperpilot.agents.document_cache import current_source
    return current_source(pdf_name)


def _claims_source(pdf_name: str) -> str | None:
    """读 claims.json 里记录的解析源；无产物/无字段返回 None。"""
    stem = _stem(pdf_name)
    claims_file = CLAIMS_DIR / f"{stem}.claims.json"
    if not claims_file.exists():
        return None
    d = _load(claims_file)
    return str(d.get("source")) if d and d.get("source") else None


# ───────────────────────── QASPER 文本源（无 PDF 版式） ─────────────────────────

QASPER_PREFIX = "qasper_"
QASPER_SUFFIX = ".qpdf"


def is_qasper(pdf_name: str) -> bool:
    return pdf_name.startswith(QASPER_PREFIX) and pdf_name.endswith(QASPER_SUFFIX)


def _qasper_pid(pdf_name: str) -> str:
    """qasper_<pid>.qpdf → <pid>。"""
    return pdf_name[len(QASPER_PREFIX):-len(QASPER_SUFFIX)]


def _qasper_paper(pdf_name: str) -> dict[str, Any]:
    pid = _qasper_pid(pdf_name)
    paper = load_papers().get(pid)
    if not paper:
        raise FileNotFoundError(f"QASPER 缺论文: {pid}")
    return paper


def _qasper_chunks(pdf_name: str) -> list[Any]:
    """QASPER 论文的 Chunk[]（与 document_cache / QA 共用，chunk_id 恒定）。"""
    return build_chunks(_qasper_paper(pdf_name))


def _qasper_title(pdf_name: str) -> str:
    return str(_qasper_paper(pdf_name).get("title", "") or "").strip()


# ───────────────────────── Stage：各环节（产物缓存优先） ─────────────────────────


def _source_chunks(pdf_name: str, *, max_len: int = MAX_LEN) -> list[Any]:
    """统一 chunk 源：委托 document_cache.ordered_chunks（QASPER / MinerU / pymupdf）。

    报告链与 QA 检索链从这里拿到**同一批 chunk**（同源同 chunk_id），
    保证 claims.chunk_id 能被 verify / 检索一致回核。max_len 忽略：
    chunk 切分阈值统一为 document_cache.MAX_CHUNK_LEN(=4000)，与 MAX_LEN 一致。
    """
    from paperpilot.agents.document_cache import ordered_chunks
    return ordered_chunks(pdf_name)


def _display_title(pdf_name: str) -> str:
    """标题源统一：QASPER 用数据集 title；MinerU 用 content_list 整篇题；否则 PDF 元数据。"""
    if is_qasper(pdf_name):
        return _qasper_title(pdf_name)
    from paperpilot.agents.document_cache import current_source
    if current_source(pdf_name) == "mineru":
        t = title_from_mineru_dir(MINERU_OUT / _stem(pdf_name))
        if t:
            return t
    return _paper_title(PAPERS_DIR / pdf_name)


def stage_claims(pdf_name: str, *, force: bool, workers: int,
                 verbose: bool = False) -> dict[str, Any]:
    """claims 提取。返回 claims.json 的 payload。"""
    out = CLAIMS_DIR / f"{_stem(pdf_name)}.claims.json"
    if out.exists() and not force:
        if verbose:
            print(f"  [claims] 缓存复用 {out.name}")
        return _load(out) or {}
    chunks = _source_chunks(pdf_name)
    target = analyzer.extractable(chunks)
    claims, errors = analyzer.extract_claims(target, workers=workers)
    chunk_map = {c.chunk_id: c.text for c in target}
    hit, loose, miss = verify_evidence(claims, chunk_map)
    by_type = Counter(c.type for c in claims)
    payload = {
        "pdf": pdf_name,
        "source": _payload_source(pdf_name),
        "model": os_model(),
        "n_chunks": len(target),
        "n_claims": len(claims),
        "by_type": dict(by_type),
        "errors": errors,
        "claims": [claim_to_dict(c) for c in claims],
    }
    _dump(out, payload)
    if verbose:
        print(f"  [claims] 提取 {len(claims)} 条 | 类型 {dict(by_type)}"
              f" | 溯源 ✓{len(hit)} ~{len(loose)} ✗{len(miss)}"
              f" | err {len(errors)}")
    return payload


def stage_view(pdf_name: str, *, force: bool, verbose: bool = False) -> list[dict[str, Any]]:
    """去重 → 打标 → 算分 → summary.json + 重点摘要 md。返回 groups dict 列表。"""
    stem = _stem(pdf_name)
    claims_file = CLAIMS_DIR / f"{stem}.claims.json"
    sum_file = VIEW_DIR / f"{stem}.summary.json"
    md_file = VIEW_DIR / f"{stem}.md"
    if sum_file.exists() and not force:
        if verbose:
            print(f"  [view] 缓存复用 {sum_file.name}")
        return (json.loads(sum_file.read_text(encoding="utf-8")).get("groups") or [])

    data = _load(claims_file)
    if not data:
        raise FileNotFoundError(f"缺 claims.json：{claims_file}（先跑 stage_claims）")
    claims = [dict_to_claim(d) for d in data["claims"]]
    claim_map = {c.claim_id: c for c in claims}

    if verbose:
        print(f"  [view] 去重 {len(claims)} 条 claims（LLM）…")
    groups = dedupe_groups(claims)
    if verbose:
        print(f"  [view] → {len(groups)} 组；打标（LLM）…")
    label_groups(groups)
    ev_map = {c.claim_id: evidence_state(c, _chunk_text_map(pdf_name))
              for c in claims}
    score_groups(groups, claim_map, ev_map)
    groups_dict = [g.model_dump(mode="json") for g in groups]

    _dump(sum_file, {"pdf": pdf_name, "n_claims": len(claims),
                     "groups": groups_dict})
    title = _display_title(pdf_name) or stem
    md = render_markdown(groups, {"pdf": pdf_name, "n_claims": len(claims),
                                  "title": title})
    md_file.write_text(md, encoding="utf-8")
    if verbose:
        print(f"  [view] 已保存 {sum_file.name} + {md_file.name}")
    return groups_dict


def stage_skeleton(pdf_name: str, *, force: bool,
                   verbose: bool = False) -> list[dict[str, Any]]:
    """论证骨架 → skeleton.json。返回 hubs dict 列表。"""
    stem = _stem(pdf_name)
    sum_file = VIEW_DIR / f"{stem}.summary.json"
    skel_file = VIEW_DIR / f"{stem}.skeleton.json"
    if skel_file.exists() and not force:
        if verbose:
            print(f"  [skeleton] 缓存复用 {skel_file.name}")
        return (json.loads(skel_file.read_text(encoding="utf-8")).get("hubs") or [])

    groups = _load(sum_file)
    claims = _load(CLAIMS_DIR / f"{stem}.claims.json")
    if not groups or not claims:
        raise FileNotFoundError(f"缺 summary/claims（先跑 stage_view / stage_claims）")
    gdicts = groups.get("groups") or []
    claim_map = {c["claim_id"]: c for c in claims.get("claims", [])}
    if verbose:
        print(f"  [skeleton] 建骨架（{len(gdicts)} 组）…")
    skel = build_skeleton(pdf_name, gdicts, claim_map)
    _dump(skel_file, skel)
    if verbose:
        n_edges = sum(len(h.get("edges") or []) for h in skel["hubs"])
        print(f"  [skeleton] {len(skel['hubs'])} Hub / {n_edges} 边")
    return skel["hubs"]


def stage_figures(pdf_name: str, *, force: bool,
                  verbose: bool = False) -> list[dict[str, Any]]:
    """图表识别 + 读图指南 → figures.json。返回 figures dict 列表。"""
    stem = _stem(pdf_name)
    fig_file = VIEW_DIR / f"{stem}.figures.json"
    if fig_file.exists() and not force:
        if verbose:
            print(f"  [figures] 缓存复用 {fig_file.name}")
        return (json.loads(fig_file.read_text(encoding="utf-8")).get("figures") or [])

    if is_qasper(pdf_name):
        # QASPER 文本源无版面图：写空 figures（QA/报告不依赖图表）
        _dump(fig_file, {"pdf": pdf_name, "figures": []})
        return []
    from paperpilot.agents.document_cache import current_source
    if current_source(pdf_name) == "mineru":
        # MinerU：直接从 content_list 的 table/chart/image 元素取 caption/页（含图内文字不污染）
        figs = figures_from_mineru_dir(MINERU_OUT / _stem(pdf_name)) or []
    else:
        result = parse_pdf(str(PAPERS_DIR / pdf_name))
        figs = extract_figures(result["blocks"])
    if figs:
        if verbose:
            print(f"  [figures] 识别 {len(figs)} 个图表，生成指南（LLM）…")
        guides = generate_guides(figs)
        for f in figs:
            f["guide"] = guides.get(f["id"], "")
    for f in figs:
        f["refs"] = f.get("refs", [])[:3]
    _dump(fig_file, {"pdf": pdf_name, "figures": figs})
    if verbose:
        n_guide = sum(1 for f in figs if f.get("guide"))
        print(f"  [figures] {len(figs)} 图 / {n_guide} 带指南")
    return figs


def stage_report_text(pdf_name: str, *, force: bool,
                      groups: list[dict[str, Any]], hubs: list[dict[str, Any]],
                      figures: list[dict[str, Any]],
                      verbose: bool = False) -> tuple[str, str]:
    """概述 + 导读 + report.md。返回 (overview, guide)。"""
    stem = _stem(pdf_name)
    ov_file = VIEW_DIR / f"{stem}.overview.json"
    gd_file = VIEW_DIR / f"{stem}.guide.json"
    md_file = VIEW_DIR / f"{stem}.report.md"

    title = _display_title(pdf_name) or stem
    overview = _load(ov_file) if not force else None
    guide = _load(gd_file) if not force else None
    if overview is None or guide is None:
        if not llm.is_configured():
            raise RuntimeError("LLM 未配置（生成概述/导读需要），或产物缺失。"
                               "请填写 .env / 设环境变量。")
        if overview is None:
            if verbose:
                print("  [report] 生成概述（LLM）…")
            ov = build_overview(groups, title)
            _dump(ov_file, {"overview": ov})
            overview = {"overview": ov}
        if guide is None:
            if verbose:
                print("  [report] 生成导读（LLM）…")
            gd = build_guide(groups, title)
            _dump(gd_file, {"guide": gd})
            guide = {"guide": gd}
    overview_text = str((overview or {}).get("overview", ""))
    guide_text = str((guide or {}).get("guide", ""))

    n_claims = sum(len(g.get("claim_ids", [])) for g in groups)
    md = render_report(groups, hubs, overview_text, guide_text,
                       {"pdf": pdf_name, "title": title, "n_claims": n_claims},
                       figures=figures)
    md_file.write_text(md, encoding="utf-8")
    if verbose:
        print(f"  [report] 已保存 {md_file.name}")
    return overview_text, guide_text


# ───────────────────────── 装配：最终 PaperReport ─────────────────────────


def load_claim_models(pdf_name: str) -> list[Claim]:
    """从产物读全量 claims → Claim[]（问答层/RAG 共用 loader）。"""
    data = _load(CLAIMS_DIR / f"{_stem(pdf_name)}.claims.json") or {}
    return [Claim.model_validate(d) for d in data.get("claims", [])]


def load_group_models(pdf_name: str) -> list[ClaimGroup]:
    """从产物读主张组 → ClaimGroup[]（问答层/RAG 共用 loader）。"""
    data = _load(VIEW_DIR / f"{_stem(pdf_name)}.summary.json") or {}
    return [ClaimGroup.model_validate(d) for d in data.get("groups", [])]


def _brief(g: ClaimGroup) -> GroupBrief:
    return GroupBrief(
        gid=g.group_id,
        rep_claim_id=g.rep_claim_id,   # 引用链：展示条目 → 代表 claim → 原文证据
        label=g.label,
        importance=g.importance,
        text=g.rep_text,
        ev=g.ev_state,
        pages=list(g.pages),
    )


def _assemble(pdf_name: str, *, verbose: bool = False) -> PaperReport:
    stem = _stem(pdf_name)
    claims = load_claim_models(pdf_name)
    groups = load_group_models(pdf_name)
    skel_data = _load(VIEW_DIR / f"{stem}.skeleton.json")
    fig_data = _load(VIEW_DIR / f"{stem}.figures.json")
    ov = _load(VIEW_DIR / f"{stem}.overview.json")
    gd = _load(VIEW_DIR / f"{stem}.guide.json")
    if not claims and not groups:
        raise FileNotFoundError(f"缺 claims/summary，无法装配 report（{pdf_name}）")

    hubs: list[dict[str, Any]] = (skel_data or {}).get("hubs", []) or []
    figures: list[dict[str, Any]] = (fig_data or {}).get("figures", []) or []

    by_id = {g.group_id: g for g in groups}
    # 骨架 edges 补 source 文本；hub 带 rep_claim_id（引用链直达原文证据）
    hub_views: list[HubView] = []
    for h in hubs:
        h_edges: list[EdgeView] = []
        for e in h.get("edges", []):
            src = by_id.get(e.get("source", ""))
            h_edges.append(EdgeView(
                relation=str(e.get("relation", "")),
                source=str(e.get("source", "")),
                text=src.rep_text if src else "",
                why=str(e.get("why", "") or ""),
                evidence=str(e.get("evidence", "") or ""),
                page=int(e.get("page") or 0),
                ev=str(e.get("ev", "") or ""),
            ))
        hub = by_id.get(h.get("hub", ""))
        hub_views.append(HubView(
            hub=h.get("hub", ""),
            label=h.get("label", ""),
            importance=int(h.get("importance", 0)),
            text=h.get("rep_text", ""),
            isolated_reason=h.get("isolated_reason", ""),
            edges=h_edges,
            rep_claim_id=hub.rep_claim_id if hub else "",
        ))

    # 核心要点：importance>=5 按总分取前 8
    def total(g: ClaimGroup) -> float:
        return float((g.score or {}).get("total", 0) or 0)

    top = sorted((g for g in groups if g.importance >= 5),
                 key=total, reverse=True)[:8]
    core_points = [_brief(g) for g in top]
    lims = sorted((g for g in groups if g.label == "limitation"),
                  key=lambda g: g.importance, reverse=True)
    limitations = [_brief(g) for g in lims]

    # 章节精读（与 report.md 的章节组织对齐）
    sec_order: list[str] = []
    for g in groups:
        s = (g.sections or ["(无标题)"])[0]
        if s not in sec_order:
            sec_order.append(s)
    sections: list[SectionView] = []
    for s in sec_order:
        gs = [g for g in groups if (g.sections or ["(无标题)"])[0] == s]
        show = sorted((g for g in gs if g.importance >= 4),
                      key=lambda g: (g.importance, total(g)), reverse=True)
        rest = sorted((g for g in gs if g.importance < 4),
                      key=lambda g: (g.importance, total(g)), reverse=True)
        sections.append(SectionView(
            title=s,
            groups=[_brief(g) for g in show],
            n_detail=len(rest),
            detail_groups=[_brief(g) for g in rest],
        ))

    n_edges = sum(len(h.get("edges") or []) for h in hubs)
    stats = {
        "n_claims": len(claims),
        "n_groups": len(groups),
        "n_hubs": len(hubs),
        "n_edges": n_edges,
        "n_must": len(core_points),
        "n_lim": len(limitations),
        "n_figures": len(figures),
    }

    report = PaperReport(
        pdf=pdf_name,
        title=_display_title(pdf_name) or stem,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        stats=stats,
        guide=str((gd or {}).get("guide", "") or ""),
        overview=str((ov or {}).get("overview", "") or ""),
        core_points=core_points,
        limitations=limitations,
        skeleton=hub_views,
        sections=sections,
        figures=figures,
        claims=claims,
        groups=groups,
    )
    return report


# ───────────────────────── 工具函数 ─────────────────────────


def os_model() -> str:
    import os
    return os.environ.get("PAPERPILOT_LLM_MODEL", "")


def _chunk_text_map(pdf_name: str) -> dict[str, str]:
    target = analyzer.extractable(_source_chunks(pdf_name))
    return {c.chunk_id: c.text for c in target}


def required_files(pdf_name: str) -> list[Path]:
    stem = _stem(pdf_name)
    return [CLAIMS_DIR / f"{stem}.claims.json",
            VIEW_DIR / f"{stem}.summary.json",
            VIEW_DIR / f"{stem}.skeleton.json",
            VIEW_DIR / f"{stem}.figures.json",
            VIEW_DIR / f"{stem}.overview.json",
            VIEW_DIR / f"{stem}.guide.json"]


# ───────────────────────── 主入口：一步完成 ─────────────────────────


def process_pdf(pdf_name: str, *, force: bool = False,
                skip_llm: bool = False, workers: int = 4,
                verbose: bool = True) -> PaperReport:
    """一个论文源 → 报告产物 → PaperReport（并落盘 report.json）。

    支持两种源：
      - PDF 文件（assets/papers/<pdf_name>）
      - QASPER 文本源（qasper_<paper_id>.qpdf，从 qasper_data 数据集构造 chunks，
        无 PDF 版面/页码——claims 提取、view、报告、QA 全链路一致）

    Args:
        pdf_name: PDF 文件名 或 QASPER 虚拟名
        force: True 时全链路重跑（调 LLM）；False 时有缓存则复用
        skip_llm: True 时只装配已有产物，缺失任一环节直接报错（不调 LLM）
        workers: claims 提取并发数
        verbose: 打印各环节进度
    """
    if not is_qasper(pdf_name):
        pdf_path = PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    stem = _stem(pdf_name)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    # 解析源一致性：claims 产物若来自不同源（pymupdf vs MinerU）→ 整链重建，
    # 防止旧 pymupdf claims/report 混入 MinerU 检索或反之（QA 向量缓存会自动重建）
    src = _payload_source(pdf_name)
    old_src = _claims_source(pdf_name)
    stale = old_src is not None and old_src != src
    eff_force = force or stale
    if stale and verbose:
        print(f"  [source] 解析源 {old_src} → {src}，整链重建产物")

    if skip_llm:
        if stale:
            raise FileNotFoundError(
                f"产物解析源不一致（claims={old_src}，当前={src}）："
                f"请用 process_pdf(..., force=True) 重建后再 --skip-llm 装配")
        missing = [p for p in required_files(pdf_name) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--skip-llm 需要全部产物已存在，缺：{[p.name for p in missing]}")
        print(f"[{pdf_name}] 装配已有产物（skip-llm）…")
        report = _assemble(pdf_name, verbose=verbose)
    else:
        # 1. claims 提取
        claims_payload = stage_claims(pdf_name, force=eff_force, workers=workers,
                                      verbose=verbose)
        # 2. 去重 / 打标 / 算分
        groups = stage_view(pdf_name, force=eff_force, verbose=verbose)
        # 3. 论证骨架
        hubs = stage_skeleton(pdf_name, force=eff_force, verbose=verbose)
        # 4. 图表
        figures = stage_figures(pdf_name, force=eff_force, verbose=verbose)
        # 5. 概述 / 导读 / 报告 md
        stage_report_text(pdf_name, force=eff_force, groups=groups, hubs=hubs,
                          figures=figures, verbose=verbose)
        # 6. 装配最终 report.json
        report = _assemble(pdf_name, verbose=verbose)

    out = VIEW_DIR / f"{stem}.report.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"✔ report.json 已保存: {out}")
    print(f"  {report.title or stem} ｜ stats={report.stats}")
    return report
