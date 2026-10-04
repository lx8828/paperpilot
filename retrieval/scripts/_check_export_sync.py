"""**gold ↔ runner 产物同步检查** —— 防「改了 gold 没重导出」的假测试。

## 为什么必须有

`retrieval/tmp/<group>/*.questions.json` 是**唯一真源**，但 runner 不读它：
`_export_questions.py` 把 gold 映射成 runner 字段（`gold.must_any` → `must_have`
等，见该文件注释）后写到

    qa/questions/<stem>.json      单篇层（A 组）
    qa/multi/<group>.json         组级层（B 组）

**中间没有任何 gate**：改了 gold 的锚点、忘了重跑导出，跑批就用**旧锚点**判分
—— 跑批不报错，只是静默地在测另一份题（**假测试**）。
2026-09-23 排查时发现这属于"改了但测试没走"的一类，故补此检查。

## 检查项

    单篇层  qid 集合一致 / must_all·must_any·must_not·expect 逐项一致 /
            question·hint·route_min 一致 / pdf 字段 == <stem>.pdf
    组级层  同上 + kind 一致 + `must_have` 必须等于 `must_all`（旧 runner 回退口径）

用法：python retrieval/scripts/_check_export_sync.py --group group2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import (EXPORT_DIR, ROOT, group_gold_file,  # noqa: E402
                        has_single_layer, material_dir, papers)

LIST_FIELDS = ("must_all", "must_any", "must_not", "expect")
STR_FIELDS = ("question", "hint", "route_min")


def cmp_q(g: dict, e: dict) -> list[tuple[str, object, object]]:
    """逐字段比对 gold 与 runner 产物（只比 runner 真的会用的字段）。"""
    d: list[tuple[str, object, object]] = []
    for f in LIST_FIELDS:
        if list(g.get(f) or []) != list(e.get(f) or []):
            d.append((f, g.get(f), e.get(f)))
    for f in STR_FIELDS:
        if str(g.get(f) or "") != str(e.get(f) or ""):
            d.append((f, str(g.get(f))[:50], str(e.get(f))[:50]))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group2")
    ap.add_argument("--quiet", action="store_true", help="只打结论行")
    args = ap.parse_args()

    bad = 0
    print(f"=== 单篇层：gold → {EXPORT_DIR.relative_to(ROOT)}/<stem>.json   [group={args.group}]")
    print(f"{'stem':<16}{'gold':>5}{'导出':>5}  状态")
    # ⚠️ M1-only 组（`single_layer: False`）**没有单篇层** → 整段跳过。
    #    不跳的话会稳定报「5 个 stem 缺 gold / 5 处不同步」的**假警报**（2026-09-25 实测）。
    if not has_single_layer(args.group):
        print(f"  （该组声明单篇层不存在，跳过）")
    for stem in ([] if not has_single_layer(args.group) else papers(args.group)):
        gf = material_dir(args.group) / f"{stem}.questions.json"
        ef = EXPORT_DIR / f"{stem}.json"
        if not gf.exists():
            print(f"{stem:<16}{'—':>5}{'—':>5}  ⚠️ 缺 gold")
            bad += 1
            continue
        if not ef.exists():
            print(f"{stem:<16}{'—':>5}{'—':>5}  ❌ 未导出（跑批读不到这 14 题）")
            bad += 1
            continue
        g = json.loads(gf.read_text(encoding="utf-8"))["questions"]
        e = json.loads(ef.read_text(encoding="utf-8"))
        gi, ei = {q["qid"]: q for q in g}, {q["qid"]: q for q in e}
        only_g, only_e = sorted(set(gi) - set(ei)), sorted(set(ei) - set(gi))
        diffs = [(k, *cmp_q(gi[k], ei[k])) for k in sorted(set(gi) & set(ei))
                 if cmp_q(gi[k], ei[k])]
        pdf_bad = [(k, ei[k].get("pdf")) for k in ei
                   if str(ei[k].get("pdf") or "") != f"{stem}.pdf"]
        ok = not (only_g or only_e or diffs or pdf_bad)
        bad += 0 if ok else 1
        print(f"{stem:<16}{len(g):>5}{len(e):>5}  {'✓ 同步' if ok else '❌ 不同步'}")
        if not args.quiet:
            for x in diffs[:6]:
                print(f"        ✗ {x[0]}: gold={x[1]!r} → 导出={x[2]!r}")
            if only_g:
                print(f"        ✗ gold 有、导出没有：{only_g}")
            if only_e:
                print(f"        ✗ 导出有、gold 没有：{only_e}")
            if pdf_bad:
                print(f"        ✗ pdf 字段应为 {stem}.pdf：{pdf_bad}")

    print(f"\n=== 组级层：gold → qa/multi/{args.group}.json")
    gfile = group_gold_file(args.group)
    efile = ROOT / "qa" / "multi" / f"{args.group}.json"
    if not gfile.exists():
        print("  （无组级 gold，跳过）")
    elif not efile.exists():
        print(f"  ❌ 未导出 {efile.relative_to(ROOT)}（B 组跑批读不到）")
        bad += 1
    else:
        gd = json.loads(gfile.read_text(encoding="utf-8"))["questions"]
        ed = json.loads(efile.read_text(encoding="utf-8"))
        A, B = {q["qid"]: q for q in gd}, {q["qid"]: q for q in ed}
        only_g, only_e = sorted(set(A) - set(B)), sorted(set(B) - set(A))
        diffs = [(k, *cmp_q(A[k], B[k])) for k in sorted(set(A) & set(B))
                 if cmp_q(A[k], B[k])]
        kind_bad = [(k, A[k].get("kind"), B[k].get("kind")) for k in set(A) & set(B)
                    if str(A[k].get("kind")) != str(B[k].get("kind"))]
        # 组级导出把 must_all 也写进 must_have（未升级的旧 runner 回退时仍按严格口径判）
        gov_bad = [(k, B[k].get("must_have"), B[k].get("must_all")) for k in B
                   if list(B[k].get("must_have") or []) != list(B[k].get("must_all") or [])]
        ok = not (only_g or only_e or diffs or kind_bad or gov_bad)
        bad += 0 if ok else 1
        from collections import Counter
        print(f"  gold {len(gd)} / 导出 {len(ed)} ｜ kind {dict(Counter(str(q.get('kind')) for q in ed))}"
              f" ｜ {'✓ 同步' if ok else '❌ 不同步'}")
        if not args.quiet:
            for x in diffs[:8]:
                print(f"        ✗ {x[0]}: gold={x[1]!r} → 导出={x[2]!r}")
            if only_g:
                print(f"        ✗ gold 有、导出没有：{only_g}")
            if only_e:
                print(f"        ✗ 导出有、gold 没有：{only_e}")
            if kind_bad:
                print(f"        ✗ kind 不一致：{kind_bad}")
            if gov_bad:
                print(f"        ✗ must_have≠must_all（旧 runner 回退口径）：{gov_bad[:5]}")

    print(f"\n结论：{'全部同步 ✓（导出产物可代表 gold）' if bad == 0 else f'{bad} 处不同步 ✗ —— 跑批前先跑 `_export_questions.py --group {args.group}`'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
