"""Bench 回归基准：扫全部论文，输出"模块勾叉矩阵"，并可与基线快照 diff。

设计意图（回归护栏，不是验收测试）：
  改了代码 → 重跑一次 → 对比 baseline 看勾叉变化：
    - 好论文"勾→叉" = 改出回归了，立刻定位是哪个模块的改动闯祸
    - 恶意论文"叉→勾" = 改动真的让系统变强了
  "叉"不在此处修复，只记录；修不修另议。

模块分两类：
  本地重算（重新 parse/chunk + evidence 规则验证，不调 LLM，检测解析层改动）：
    PARS 解析           PDF 能解析、blocks>0
    HDR  标题/分块       能切出正文 chunk 且带 title_path
    EVID 证据溯源        evidence 回原文命中率（miss 占比 ≤ 阈值）
  产物检查（只读 out_claims / out_views 的 JSON，检测下游环节产物健康）：
    CLA  Claims 提取     claims>0 且无提取错误
    NS   噪声筛查        可选环节：无 noise.json 时跳过（·），不判失败
    DEUP 去重归并        summary.json 存在、groups 数合理
    LAB  打标打分        label/importance/score 字段合法
    SKEL 论证骨架        hubs>0 且无"无边 hub"
    FIG  图表处理        figures.json 存在、图均带指南
    RPT  完整报告        overview/guide/report.md 三件齐

用法：
    uv run python run_bench.py                             # 全量检查 storage/papers 全部
    uv run python run_bench.py 2609.00859v1.pdf            # 指定论文
    uv run python run_bench.py --save-baseline             # 首次：把本次结果固化为基线
    uv run python run_bench.py --no-local                  # 跳过本地解析重算（只查产物层）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools import analyzer
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.evidence import dict_to_claim, verify_evidence
from paperpilot.tools.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "src" / "paperpilot" / "storage" / "papers"
CLAIMS_DIR = ROOT / "out_claims"
VIEWS_DIR = ROOT / "out_views"
BENCH_DIR = ROOT / "bench"

# ───────────────────────── 常量 ─────────────────────────

LABELS = {
    "core_claim", "result_primary", "result_supporting", "method_core",
    "ablation", "detail", "limitation", "related",
}
MODULES = ["PARS", "HDR", "CLA", "EVID", "NS", "DEUP", "LAB", "SKEL", "FIG", "RPT"]
GROUP = {
    "PARS": "本地重算", "HDR": "本地重算", "EVID": "本地重算",
    "CLA": "产物", "NS": "产物", "DEUP": "产物", "LAB": "产物",
    "SKEL": "产物", "FIG": "产物", "RPT": "产物",
}

NOISE_MAX = 0.15   # 与 quality.ABNORMAL 一致
MISS_MAX = 0.15    # evidence miss 占比上限
HDR_TITLE_RATIO = 0.25   # L1 标题污染占比 ≥ 此值判 HDR fail（抓系统性，容忍零星）
SKEL_ISOLATED_RATIO = 0.25  # 孤岛 hub 占比 > 此值判 SKEL fail

# 幽灵标题判据：正常章节标题是短短语，不含数学符号。
# 数学符号/超长句 → 极可能是正文句或符号碎片被误判成标题（如 "2 Z2⊕Z3 Z2"）。
_MATH_CHARS = set("⊕⊗⊖⊂⊃⊆⊇∈∉≤≥≠≈≡◦⊥∠∑∏∫√∞θφψαβγδΩΔ°→←π±×")


def _heading_body(label: str) -> str:
    """从 chunk.title_path 的 L1 标签（如 'L1 1 · 1 Introduction'）取标题正文。"""
    if " · " in label:
        return label.split(" · ", 1)[1].strip()
    return label.strip()


def _strip_number(body: str) -> str:
    """剥离标题编号前缀（'1 ' / '3.2 ' / 'A '），剩纯标题文本。"""
    m = re.match(r"^(?:[A-Za-z]?\d+(?:\.\d+)*[a-z]?|[IVX]+|[A-Za-z])\s*[.\-]?\s*", body)
    return body[m.end():] if m else body


def _suspicious_heading(label: str) -> bool:
    """幽灵标题特征：超长句（>70 字）或含数学结构符号。"""
    body = _strip_number(_heading_body(label))
    if len(body) > 70:
        return True
    return any(ch in _MATH_CHARS for ch in body)


def status(s: str, detail: str = "") -> dict[str, str]:
    return {"status": s, "detail": detail}


# ───────────────────────── 本地重算模块 ─────────────────────────


def _local_parse(pdf_path: Path) -> tuple[dict[str, Any], Any]:
    """重跑解析+分块。返回 (report, target_chunks)。"""
    out: dict[str, dict[str, str]] = {}
    try:
        result = parse_pdf(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        out["PARS"] = status("fail", f"解析异常: {type(e).__name__}: {e}")
        return out, None
    blocks = result["blocks"]
    if not blocks:
        out["PARS"] = status("fail", "blocks=0（无文本层？扫描件？）")
        return out, None
    out["PARS"] = status("pass", f"{result['page_count']} 页 / {len(blocks)} blocks")

    chunks = chunk_document(blocks, max_len=4000)
    target = analyzer.extractable(chunks)
    if not target:
        out["HDR"] = status("fail", "无正文 chunk（标题识别/过滤异常）")
        return out, None
    bare = [c.chunk_id for c in target if not c.title_path]
    if bare:
        # 缺 title_path 只影响标题组织，不影响 evidence 验证，继续
        out["HDR"] = status("fail", f"正文 chunk 缺 title_path: {bare[:5]}")
    # L1 标题质量：幽灵标题（正文句/符号碎片被误判成章节标题）
    l1_all: list[str] = []
    for c in target:
        if c.title_path and c.title_path[0].startswith("L1 "):
            txt = _heading_body(c.title_path[0])
            if txt and txt not in l1_all:
                l1_all.append(txt)
    susp = [t for t in l1_all if _suspicious_heading(t)]
    if "HDR" not in out:
        parts = [f"正文 {len(target)} chunk / {len(l1_all)} 个 L1 节"]
        if susp:
            ratio = len(susp) / max(len(l1_all), 1)
            parts.append(f"⚠可疑标题 {len(susp)}/{len(l1_all)}")
            if ratio >= HDR_TITLE_RATIO:
                out["HDR"] = status("fail", f"标题污染 {len(susp)}/{len(l1_all)}（样例: {susp[0][:60]}）")
            else:
                parts.append(f"零星: {susp[0][:40]}")
                out["HDR"] = status("pass", " / ".join(parts))
        else:
            out["HDR"] = status("pass", " / ".join(parts))
    return out, target


def _local_evidence(claims_json: dict[str, Any], target: Any) -> dict[str, Any]:
    """对产物 claims 重新做 evidence 规则验证。"""
    claims = [dict_to_claim(d) for d in claims_json["claims"]]
    chunk_map = {c.chunk_id: c.text for c in target}
    hit, loose, miss = verify_evidence(claims, chunk_map)
    n = len(claims) or 1
    miss_ratio = len(miss) / n
    known = sum(1 for c in claims if c.chunk_id in chunk_map)
    match_rate = (known / len(claims)) if claims else 1.0
    parts = [f"✓{len(hit)} ~{len(loose)} ✗{len(miss)} (miss {miss_ratio:.0%})"]
    if match_rate < 1.0:
        parts.append(f"⚠ chunk_id 匹配 {match_rate:.0%}（分块漂移？）")
    if miss_ratio <= MISS_MAX:
        return {"EVID": status("pass", " / ".join(parts))}
    return {"EVID": status("fail", " / ".join(parts) + f" > 阈值 {MISS_MAX:.0%}")}


# ───────────────────────── 产物检查模块 ─────────────────────────


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _prod_claims(stem: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """CLA。返回 (report, claims_dict)"""
    path = CLAIMS_DIR / f"{stem}.claims.json"
    d = _read(path)
    if d is None:
        return {"CLA": status("fail", "缺 claims.json")}, None
    n = d.get("n_claims", 0)
    errs = d.get("errors") or []
    if n <= 0:
        return {"CLA": status("fail", "n_claims=0")}, d
    if errs:
        return {"CLA": status("fail", f"提取失败 chunk {len(errs)} 个: {errs[0][:60]}…")}, d
    return {"CLA": status("pass", f"{n} 条 / 4 类={sorted((d.get('by_type') or {}).items())}")}, d


def _prod_noise(stem: str, n_claims: int) -> dict[str, Any]:
    d = _read(CLAIMS_DIR / f"{stem}.noise.json")
    if d is None:
        return {"NS": status("skip", "无 noise.json（未跑 run_scan_noise）")}
    noisy = d.get("noisy") or []
    ratio = len(noisy) / n_claims if n_claims else 0
    if ratio <= NOISE_MAX:
        return {"NS": status("pass", f"{len(noisy)} 条噪声 ({ratio:.0%})")}
    return {"NS": status("fail", f"{len(noisy)}/{n_claims} ({ratio:.0%}) > {NOISE_MAX:.0%}")}


def _prod_summary(stem: str, n_claims: int) -> dict[str, Any]:
    """DEUP + LAB。"""
    d = _read(VIEWS_DIR / f"{stem}.summary.json")
    if d is None:
        return {"DEUP": status("fail", "缺 summary.json"), "LAB": status("fail", "缺 summary.json")}
    groups = d.get("groups") or []
    g = len(groups)
    out: dict[str, Any] = {}
    if g <= 0:
        out["DEUP"] = status("fail", "groups=0")
    elif g > n_claims:
        out["DEUP"] = status("fail", f"groups {g} > claims {n_claims}（归并未生效？）")
    else:
        out["DEUP"] = status("pass", f"{n_claims}→{g} 组（压缩 {g / max(n_claims,1):.0%}）")

    bad_label = sorted({gr.get("label", "?") for gr in groups if gr.get("label") not in LABELS})
    bad_imp = sorted({gr.get("importance") for gr in groups
                      if not isinstance(gr.get("importance"), int) or not 1 <= gr["importance"] <= 5})
    bad_score = [gr.get("group_id") for gr in groups if not isinstance(gr.get("score"), dict)]
    if bad_label or bad_imp or bad_score:
        msg = []
        if bad_label:
            msg.append(f"非法 label={bad_label}")
        if bad_imp:
            msg.append(f"非法 importance={bad_imp}")
        if bad_score:
            msg.append(f"缺 score: {bad_score[:5]}")
        out["LAB"] = status("fail", "；".join(msg))
    else:
        imp = Counter(gr.get("importance", 0) for gr in groups)
        out["LAB"] = status("pass", f"label 合法 / importance {dict(sorted(imp.items()))}")
    return out


def _prod_skeleton(stem: str) -> dict[str, Any]:
    d = _read(VIEWS_DIR / f"{stem}.skeleton.json")
    if d is None:
        return {"SKEL": status("fail", "缺 skeleton.json")}
    hubs = d.get("hubs") or []
    if not hubs:
        return {"SKEL": status("fail", "hubs=0（无核心主张？）")}
    n_edges = sum(len(h.get("edges") or []) for h in hubs)
    isolated = [h.get("hub") for h in hubs if not h.get("edges")]
    if not isolated:
        return {"SKEL": status("pass", f"{len(hubs)} Hub / {n_edges} 边")}
    ratio = len(isolated) / len(hubs)
    if ratio > SKEL_ISOLATED_RATIO:
        return {"SKEL": status("fail", f"孤岛 hub {len(isolated)}/{len(hubs)}（{isolated[:5]}）")}
    return {"SKEL": status("pass", f"{len(hubs)} Hub / {n_edges} 边（{len(isolated)} 孤岛，容忍线内）")}


def _prod_figures(stem: str) -> dict[str, Any]:
    d = _read(VIEWS_DIR / f"{stem}.figures.json")
    if d is None:
        return {"FIG": status("fail", "缺 figures.json")}
    figs = d.get("figures") or []
    if not figs:
        return {"FIG": status("pass", "0 图（无图表论文）")}
    no_guide = [f.get("id") for f in figs if not (f.get("guide") or "").strip()]
    if no_guide:
        return {"FIG": status("fail", f"无指南图: {no_guide[:8]}")}
    kinds = Counter(f.get("kind") for f in figs)
    return {"FIG": status("pass", f"{len(figs)} 图全带指南 {dict(kinds)}")}


def _prod_report(stem: str) -> dict[str, Any]:
    need = {
        "overview": VIEWS_DIR / f"{stem}.overview.json",
        "guide": VIEWS_DIR / f"{stem}.guide.json",
        "report.md": VIEWS_DIR / f"{stem}.report.md",
    }
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        return {"RPT": status("fail", f"缺: {missing}")}
    return {"RPT": status("pass", "overview + guide + report.md")}


# ───────────────────────── 单篇检查 ─────────────────────────


def check_one(pdf_name: str, *, local: bool) -> dict[str, Any]:
    pdf_path = PAPERS_DIR / pdf_name
    stem = Path(pdf_name).stem
    if not pdf_path.exists():
        return {m: status("skip", "缺少 PDF 文件") for m in MODULES}

    out: dict[str, Any] = {}

    # 本地重算层
    if local:
        local_out, target = _local_parse(pdf_path)
        out.update(local_out)
        if target is not None:
            claims_json = _read(CLAIMS_DIR / f"{stem}.claims.json")
            if claims_json is not None:
                out.update(_local_evidence(claims_json, target))
            else:
                out["EVID"] = status("skip", "无 claims.json，跳过证据验证")
    else:
        out["PARS"] = status("skip", "--no-local")
        out["HDR"] = status("skip", "--no-local")
        out["EVID"] = status("skip", "--no-local")

    # 产物层
    cla, claims_json = _prod_claims(stem)
    out.update(cla)
    if claims_json is not None:
        out.update(_prod_noise(stem, claims_json.get("n_claims", 0)))
        out.update(_prod_summary(stem, claims_json.get("n_claims", 0)))
    else:
        out["NS"] = status("skip", "无 claims.json")
        out["DEUP"] = status("skip", "无 claims.json")
        out["LAB"] = status("skip", "无 claims.json")
    out.update(_prod_skeleton(stem))
    out.update(_prod_figures(stem))
    out.update(_prod_report(stem))

    # 补齐缺失模块（防御）
    for m in MODULES:
        out.setdefault(m, status("skip", "未执行"))
    return out


# ───────────────────────── 打印与基线 ─────────────────────────

_SYM = {"pass": "✓", "fail": "✗", "skip": "·"}


def show_matrix(pdfs: list[str], results: dict[str, dict[str, Any]]) -> None:
    print("=" * 100)
    print(f"{'pdf':<20s}" + "  ".join(m.rjust(4) for m in MODULES))
    print("-" * 100)
    for pdf in pdfs:
        r = results[pdf]
        row = "  ".join(_SYM[r[m]["status"]].rjust(4) for m in MODULES)
        fails = [m for m in MODULES if r[m]["status"] == "fail"]
        mark = f"  {len(fails)}✗" if fails else ""
        print(f"{pdf:<20s}{row}{mark}")
    print("=" * 100)


def show_details(pdfs: list[str], results: dict[str, dict[str, Any]]) -> None:
    for pdf in pdfs:
        r = results[pdf]
        for m in MODULES:
            if r[m]["status"] == "fail":
                print(f"  ✗ {pdf} [{m}] {r[m]['detail']}")


def diff_baseline(pdfs: list[str], results: dict[str, dict[str, Any]],
                  baseline: dict[str, dict[str, Any]]) -> None:
    regress, improve = [], []
    for pdf in pdfs:
        for m in MODULES:
            old = baseline.get(pdf, {}).get(m, {}).get("status")
            new = results[pdf][m]["status"]
            if old == "pass" and new == "fail":
                regress.append((pdf, m))
            elif old == "fail" and new == "pass":
                improve.append((pdf, m))
    if regress:
        print("回归（勾→叉）：")
        for pdf, m in regress:
            print(f"  ✗ {pdf} [{m}] {results[pdf][m]['detail']}")
    if improve:
        print("改善（叉→勾）：")
        for pdf, m in improve:
            print(f"  ✓ {pdf} [{m}] {results[pdf][m]['detail']}")
    if not regress and not improve:
        print("与基线一致，无变化。")


# ───────────────────────── 主流程 ─────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", help="指定论文文件名；默认 storage/papers 全部")
    ap.add_argument("--save-baseline", action="store_true",
                    help="把本次结果固化为 baseline.json")
    ap.add_argument("--no-local", action="store_true",
                    help="跳过本地解析重算（只查产物层，快但检测不到解析层改动）")
    ap.add_argument("--no-baseline-diff", action="store_true",
                    help="不读 baseline.json 对比")
    args = ap.parse_args()

    if args.pdfs:
        pdfs = args.pdfs
    else:
        pdfs = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print("没有可检查的论文（storage/papers 为空）")
        return 1

    results: dict[str, dict[str, Any]] = {}
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf} …", end=" ")
        r = check_one(pdf, local=not args.no_local)
        results[pdf] = r
        fails = [m for m in MODULES if r[m]["status"] == "fail"]
        print(f"{sum(1 for m in MODULES if r[m]['status'] == 'pass')}/{len(MODULES)} ✓, "
              f"{len(fails)} ✗")

    print()
    show_matrix(pdfs, results)
    show_details(pdfs, results)

    # 快照 + 基线对比
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap_file = BENCH_DIR / "snapshots" / f"{ts}.json"
    snap_file.parent.mkdir(parents=True, exist_ok=True)
    snap_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"快照已存: {snap_file}")

    if args.save_baseline:
        (BENCH_DIR / "baseline.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("本次结果已固化为基线: bench/baseline.json")
    elif not args.no_baseline_diff:
        base = BENCH_DIR / "baseline.json"
        if base.exists():
            print("\n--- 相对基线 diff ---")
            diff_baseline(pdfs, results, json.loads(base.read_text(encoding="utf-8")))
        else:
            print("\n（无 baseline.json；首次运行可加 --save-baseline 固化基线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
