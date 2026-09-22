"""修补「打标全失败」的篇：重跑 *打标 + 算分 + 概述/导读 + 装配*。

## 背景（2026-09-22）

`viewer.label_groups` 旧版**一次性**把全部主张组发给 LLM：355 组时 user 3.9 万字符、
需输出 355 个 JSON 对象 → **输出被截断** → `_parse_json` 兜底失败 → `except` 把
**全篇**保持默认 `label="detail"`（`LABEL_CAP["detail"]=3` 于是封顶 3 分）。

后果链：`core_points=0`（要 ≥5）→ `collect_materials=0`（要 core_claim/result_primary
且 ≥4）→ 概述 prompt 材料区空白 → LLM 拒答 → **拒答话术被当成概述存进 report.json**
（全库 465 篇里 7 篇命中）。根因已在 `label_groups`（分批 50）与
`build_overview`/`build_guide`（空材料抛 `MaterialEmpty`）修掉，本脚本负责**存量数据**。

## 只重跑什么

    ✅ 打标（label_groups，现已分批）→ 算分（score_groups）→ summary.json + .md
    ✅ 概述/导读/report.md（stage_report_text force）→ report.json（_assemble）
    ❌ 不重跑 claims      —— claims.json 原样
    ❌ 不重跑 dedupe      —— **分组与 group_id 不变**（cites 锚点/评测口径不受影响）
    ❌ 不重解析 PDF/MinerU —— 复用 summary.json 里已缓存的 ev_state

用法：
    uv run python cli/run_repair_labels.py --scan           # 只列出待修篇（不调 LLM）
    uv run python cli/run_repair_labels.py --all            # 修补全部待修篇
    uv run python cli/run_repair_labels.py 2606.18837 ...   # 指定篇（按 stem）
    uv run python cli/run_repair_labels.py --all --dry-run  # 只看将要做什么
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot import pipeline as P  # noqa: E402
from paperpilot.tools import llm, viewer  # noqa: E402
from paperpilot.tools.report import degraded_reason  # noqa: E402


def inspect(stem: str) -> tuple[str, str]:
    """返回 (类别, 说明)。类别 ∈ {"", "broken", "degraded"}。

    broken   —— 疑似**修复前**产生的坏产物：全篇 detail 且 report.json **无降级标记**
    degraded —— 已正确标记降级（该篇**确实**没有核心主张；不是数据损坏，不用修）
    """
    sm = P._load(P.VIEW_DIR / f"{stem}.summary.json")
    groups = (sm or {}).get("groups") or []
    if not groups:
        return "", ""
    rep = P._load(P.VIEW_DIR / f"{stem}.report.json") or {}
    deg = str(rep.get("degraded") or "").strip()
    labels = {str(g.get("label")) for g in groups}
    top = max((int(g.get("importance") or 0) for g in groups), default=0)
    if labels == {"detail"} and top < 4:
        if deg:
            return "degraded", f"确实无核心主张（{deg}）"
        return "broken", "打标全失败（全 detail 且无 ≥4 分，且产物无降级标记 → 修复前的坏产物）"
    ov = P._load(P.VIEW_DIR / f"{stem}.overview.json")
    if ov is not None and str(ov.get("overview") or "").strip():
        why = degraded_reason(str(ov.get("overview") or ""))
        if why and not deg:
            return "broken", f"概述无效（{why}）"
    return "", ""


def needs_repair(stem: str) -> str:
    """返回需要修补的原因（"" = 无需修补）。"""
    cat, why = inspect(stem)
    return why if cat == "broken" else ""


def scan() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """返回 (待修[broken], 已知降级[degraded]，后者**不是损坏**)。"""
    broken: list[tuple[str, str]] = []
    degraded: list[tuple[str, str]] = []
    for f in sorted(P.VIEW_DIR.glob("*.summary.json")):
        stem = f.name[: -len(".summary.json")]
        cat, why = inspect(stem)
        if cat == "broken":
            broken.append((stem, why))
        elif cat == "degraded":
            degraded.append((stem, why))
    return broken, degraded


def repair(stem: str, *, verbose: bool = True) -> dict[str, int]:
    """重跑打标+算分+概述+装配。返回修补后的 stats。"""
    sm = P._load(P.VIEW_DIR / f"{stem}.summary.json") or {}
    pdf = str(sm.get("pdf") or stem)      # 原始名（qasper_* 保留 .qpdf 后缀）
    old_groups_raw = sm.get("groups") or []

    claims = P.load_claim_models(pdf)
    claim_map = {c.claim_id: c for c in claims}
    groups = P.load_group_models(pdf)     # ← 分组来自现有 summary.json，group_id 不变
    if not groups:
        raise RuntimeError(f"{stem}: 读不到主张组，跳过")

    # 复用缓存的 ev_state：claims 未变 → 证据状态未变，**不重新解析 PDF**
    ev_map = {str(g.get("rep_claim_id") or ""): str(g.get("ev_state") or "miss")
              for g in old_groups_raw}
    before = _label_hist(old_groups_raw)

    print(f"  [1/4] 重新打标 {len(groups)} 组（分批 {viewer.LABEL_BATCH}）…")
    viewer.label_groups(groups)
    print(f"  [2/4] 重新算分…")
    viewer.score_groups(groups, claim_map, ev_map)

    groups_dict = [g.model_dump(mode="json") for g in groups]
    P._dump(P.VIEW_DIR / f"{stem}.summary.json",
            {"pdf": pdf, "n_claims": len(claims), "groups": groups_dict})
    rep_old = P._load(P.VIEW_DIR / f"{stem}.report.json") or {}
    title = str(rep_old.get("title") or "") or stem
    md = viewer.render_markdown(groups, {"pdf": pdf, "n_claims": len(claims),
                                         "title": title})
    (P.VIEW_DIR / f"{stem}.md").write_text(md, encoding="utf-8")

    print(f"  [3/4] 重跑概述/导读（force）…")
    hubs = (P._load(P.VIEW_DIR / f"{stem}.skeleton.json") or {}).get("hubs") or []
    figures = (P._load(P.VIEW_DIR / f"{stem}.figures.json") or {}).get("figures") or []
    P.stage_report_text(pdf, force=True, groups=groups_dict, hubs=hubs,
                        figures=figures, verbose=verbose)

    print(f"  [4/4] 重新装配 report.json…")
    report = P._assemble(pdf, verbose=False)
    (P.VIEW_DIR / f"{stem}.report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8")

    after = _label_hist(groups_dict)
    n_must = int(report.stats.get("n_must", 0))
    print(f"  □ 标签 {before} → {after}")
    print(f"  □ n_must(core_points) {sum(1 for g in old_groups_raw if int(g.get('importance') or 0) >= 5)}"
          f" → **{n_must}** ｜ 概述 {len(str(rep_old.get('overview') or ''))} 字 → "
          f"{len(report.overview)} 字"
          + (f"  ⚠️ degraded: {report.degraded}" if report.degraded else "  ✓ 正常"))
    return dict(report.stats)


def _label_hist(groups: list[dict]) -> dict[str, int]:
    h: dict[str, int] = {}
    for g in groups:
        k = str(g.get("label"))
        h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def main() -> int:
    llm._load_dotenv(str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="篇名 stem（如 2606.18837）")
    ap.add_argument("--scan", action="store_true", help="只列出待修篇（不调 LLM）")
    ap.add_argument("--all", action="store_true", help="修补全部待修篇")
    ap.add_argument("--dry-run", action="store_true", help="不调 LLM，只列出将要处理的篇")
    args = ap.parse_args()

    if args.scan or args.all or (not args.stems and not args.dry_run):
        broken, degraded = scan()
        n = len(list(P.VIEW_DIR.glob("*.summary.json")))
        print(f"[扫描] {n} 篇 ｜ **待修 {len(broken)}** ｜ "
              f"已知降级 {len(degraded)}（确实无核心主张，**非损坏**）")
        for stem, why in broken:
            print(f"  ✗ 待修  {stem:<24} {why}")
        for stem, why in degraded:
            print(f"  · 降级  {stem:<24} {why}")
        if args.scan or (not args.stems and not args.all and not args.dry_run):
            return 0
        if args.dry_run:
            print("\n[dry-run] 未调用 LLM。去掉 --dry-run 或加 --all 执行修补。")
            return 0
    targets = args.stems or [s for s, _ in scan()[0]]

    if not args.stems and not llm.is_configured():
        print("\n✗ LLM 未配置（修补需要重跑打标/概述）。请填 .env 或设环境变量。")
        return 1

    ok = fail = 0
    for stem in targets:
        why = needs_repair(stem)
        print(f"\n{'=' * 78}\n■ {stem}" + (f"  ← {why}" if why else "  （未检出问题，仍按需修补）"))
        try:
            repair(stem)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  ✗ 失败: {type(e).__name__}: {str(e)[:200]}")
    print(f"\n{'=' * 78}\n完成：成功 {ok} ｜ 失败 {fail}")
    print("建议复核：uv run python cli/run_repair_labels.py --scan")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
