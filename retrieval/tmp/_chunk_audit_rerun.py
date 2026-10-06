# -*- coding: utf-8 -*-
"""chunk_audit.py 的补跑器（复用它的 analyze()，不改原脚本）。

为什么需要它：
  · 原脚本 `SAMPLE_N=12` 只管「算 embedding 的篇数」，**没有限制处理篇数的参数**；
    而 `assets/artifacts/out_mineru` 已从当年的 49 篇涨到 226 篇。
  · 原脚本只 print，不落盘 → 2026-09-23 那次跑挂后数字全丢。本脚本同时写 CSV。

用法：
  python retrieval/tmp/_chunk_audit_rerun.py --limit 12 --sample 6
  python retrieval/tmp/_chunk_audit_rerun.py --limit 0            # 0 = 全部
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

# ── 复用原脚本（exec_module 只跑它的顶层正则/路径定义，main() 在 __main__ 下才跑）──
_spec = importlib.util.spec_from_file_location(
    "chunk_audit", str(ROOT / "retrieval/scripts/chunk_audit.py"))
CA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CA)

from paperpilot.tools import analyzer  # noqa: E402
from paperpilot.tools.chunker import chunk_document  # noqa: E402
from paperpilot.tools.pdf_parser import parse_pdf  # noqa: E402


def run(limit: int, sample: int):
    stems = sorted(p.name for p in CA.MINERU_OUT.iterdir() if p.is_dir())
    if limit:
        stems = stems[:limit]
    print(f"MinerU 产物 {len(list(CA.MINERU_OUT.iterdir()))} 篇 ｜ 本次处理 {len(stems)} 篇 "
          f"｜ C 组（要 encode）前 {min(sample, len(stems))} 篇\n", flush=True)

    rows_m, rows_p = [], []
    t0 = time.time()
    for i, stem in enumerate(stems):
        cl = CA._find_content_list(CA.MINERU_OUT / stem)
        if cl is None:
            continue
        emb = i < sample
        # ⚠️ 与生产一致：两条路径都要套 analyzer.extractable
        rows_m.append(CA.analyze(analyzer.extractable(CA.chunks_from_mineru(cl)),
                                 "mineru", embed=emb, cl=cl) | {"stem": stem})
        pdf = CA.PAPERS / f"{stem}.pdf"
        if pdf.exists():
            try:
                chunks_p = chunk_document(parse_pdf(str(pdf))["blocks"])
                rows_p.append(CA.analyze(analyzer.extractable(chunks_p),
                                         "pymupdf", embed=emb) | {"stem": stem})
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] pymupdf {stem}: {type(e).__name__}: {e}", flush=True)
        if emb:
            print(f"  [embed {i + 1}/{min(sample, len(stems))}] {stem}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(stems)}  t={time.time() - t0:.0f}s", flush=True)
    return rows_m, rows_p, stems, t0


def report(rows_m, rows_p) -> str:
    agg = lambda rows, k: (sum(r.get(k, 0) for r in rows) / len([1 for r in rows if k in r])
                           if any(k in r for r in rows) else 0)
    agg_len = lambda rows, k: (sum(r["len"][k] for r in rows if "len" in r)
                               / len([1 for r in rows if "len" in r])
                               if any("len" in r for r in rows) else 0)
    tot_m = sum(r["n"] for r in rows_m)
    tot_p = sum(r["n"] for r in rows_p)
    L = []
    L.append(f"{'=' * 92}\n【A 长度】（每篇块均值）；分母：MinerU {tot_m} 块 / pymupdf {tot_p} 块")
    L.append(f"  {'指标':<26}{'MinerU':>12}{'pymupdf':>12}")
    for k, lab in (("min", "最短块（均值）"), ("p25", "p25"), ("med", "中位"),
                   ("p75", "p75"), ("max", "最长块（均值）")):
        L.append(f"  {lab:<26}{agg_len(rows_m, k):>12.0f}{agg_len(rows_p, k):>12.0f}")
    for k, lab in (("short100", "<100 字（近空）"), ("short200", "<200 字"),
                   ("long4000", ">4000 字（超限）"), ("multipage", "跨页块"),
                   ("with_part", "带 part（子块）")):
        L.append(f"  {lab:<26}{agg(rows_m, k):>12.1f}{agg(rows_p, k):>12.1f}")

    L.append(f"\n【B 自包含性】（占全部块的比例 %）")
    L.append(f"  {'指标':<28}{'MinerU':>12}{'pymupdf':>12}   说明")
    notes = {"lead_ref": "首句悬空指代（this/it/该/其）→ 离开上文读不懂",
             "lead_conj": "首句连接词（however/therefore）→ 依赖上文",
             "lead_cn": "首句中文指代/连接",
             "has_title": "块内含所属标题行 → 自带上下文",
             "eq_only": "孤立公式块（几乎无正文）"}
    for k in ("lead_ref", "lead_conj", "lead_cn", "has_title", "eq_only"):
        m = sum(r.get(k, 0) for r in rows_m)
        p = sum(r.get(k, 0) for r in rows_p)
        L.append(f"  {k:<28}{m / max(tot_m, 1) * 100:>11.1f}%{p / max(tot_p, 1) * 100:>11.1f}%"
                 f"   {notes[k]}")
    tm = sum(r.get("table_blocks", 0) for r in rows_m)
    tp = sum(r.get("table_blocks", 0) for r in rows_p)
    cm = sum(r.get("table_with_cap", 0) for r in rows_m)
    cp = sum(r.get("table_with_cap", 0) for r in rows_p)
    L.append(f"  {'表格块带 caption':<26}{cm / max(tm, 1) * 100:>11.1f}%"
             f"{cp / max(tp, 1) * 100:>11.1f}%   （共 {tm} / {tp} 个表格块）")

    have = [r for r in rows_m if "intra_cos" in r]
    havep = [r for r in rows_p if "intra_cos" in r]
    if have:
        L.append(f"\n【C 主题一致性】（抽样 {len(have)} 篇，cosine 越高越好；边界项越低越好）")
        L.append(f"  {'指标':<26}{'MinerU':>12}{'pymupdf':>12}")
        for k, lab in (("intra_cos", "块内前后半 cos（内部一致）"),
                       ("chunk_title_cos", "块↔节标题 cos（归属正确）"),
                       ("adj_cos", "块↔前邻 cos（低=边界干净）")):
            L.append(f"  {lab:<26}{agg(have, k):>12.3f}{agg(havep, k):>12.3f}")
    return "\n".join(L)


KEYS = ["stem", "n", "min", "p25", "med", "p75", "max", "short100", "short200",
        "long4000", "multipage", "with_part", "lead_ref", "lead_conj", "lead_cn",
        "has_title", "table_blocks", "table_with_cap", "eq_only",
        "intra_cos", "chunk_title_cos", "adj_cos"]


def dump_csv(path: Path, rows_m, rows_p) -> int:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source"] + KEYS)
        n = 0
        for src, rows in (("mineru", rows_m), ("pymupdf", rows_p)):
            for r in rows:
                ln = r.get("len", {})
                row = [src] + [r.get("stem", ""), r.get("n", "")] + \
                      [ln.get(k, "") for k in ("min", "p25", "med", "p75", "max")] + \
                      [r.get(k, "") for k in KEYS[7:]]
                w.writerow(row)
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="处理前 N 篇；0=全部")
    ap.add_argument("--sample", type=int, default=12, help="算 embedding 的篇数")
    ap.add_argument("--csv", default="retrieval/results/CHUNK_AUDIT_20261006.csv")
    ap.add_argument("--report", default="retrieval/results/CHUNK_AUDIT_20261006.md")
    a = ap.parse_args()

    rows_m, rows_p, stems, t0 = run(a.limit, a.sample)
    txt = report(rows_m, rows_p)
    print("\n" + txt, flush=True)

    n = dump_csv(Path(a.csv), rows_m, rows_p)
    print(f"\n  → CSV  {n} 行已写 {a.csv}", flush=True)
    Path(a.report).write_text(
        f"# 切块质量审计（重跑）：MinerU vs 同篇 pymupdf\n\n"
        f"> 重跑时间 2026-10-06 ｜ 脚本 `retrieval/scripts/chunk_audit.py`（复用其 `analyze()`）\n"
        f"> 本次处理 **{len(stems)} 篇**（`out_mineru` 共 {len(list(CA.MINERU_OUT.iterdir()))} 篇）"
        f"｜ C 组抽样 {a.sample} 篇 ｜ 耗时 {time.time() - t0:.0f}s\n"
        f"> 逐篇明细见 `{Path(a.csv).name}`\n\n"
        f"> ★ 补档说明：原产出 `CHUNK_AUDIT_20260923.md` 是一份**跑挂的 Tee 日志**"
        f"（只 embed 到第 3/12 篇就断，无任何指标），本文件是对它的补齐。\n\n"
        f"```text\n{txt}\n```\n", encoding="utf-8")
    print(f"  → 报告已写 {a.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
