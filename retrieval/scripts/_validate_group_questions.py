"""组级（语料级）题集校验器 —— `retrieval/tmp/<group>/_group.questions.json`。

单篇层由 `_validate_questions.py` 把关；本文件管**一次问 5 篇**的题。除了引文逐字
核对，还有两条**单篇层不存在的**硬门槛（这是组级题集能自证"测到该测的东西"的关键）：

  ① **reach（防越界）**
       M0（多篇·免检索）：must_all 必须逐字出现在 5 篇的 **L0 材料**
                            （overview + core_points + limitations）里 —— 否则它
                            根本不是"不走检索能答"的题，`--l0` 必然判「不够」。
       M1（跨篇·需检索）：must_all 必须逐字出现在 5 篇的 **L3 检索视图**的并集里
                            —— 否则锚点在系统输入里不存在，题本身越界。
       X5（语料级拒答）：无 must_all（判据是拒绝措辞），跳过 reach。

  ② **M1 必须真的跨篇**：evidence 至少来自 **2 篇不同论文** —— 只有一篇的题属于
     单篇层（那 70 题已覆盖），放在这里是**假跨篇**，会虚报"跨篇能力"。

用法：uv run python retrieval/scripts/_validate_group_questions.py [--group group1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _qa_groups import (  # noqa: E402
    GROUP_QUOTA,
    group_expected_total,
    group_gold_file,
    material_dir,
    papers,
)
from _check_reachable import l3_text  # noqa: E402
from _validate_questions import LEVELS, diverge, norm, sources_variants  # noqa: E402

REQ = ("qid", "kind", "role", "intent", "question", "hint", "evidence",
       "expect", "route_min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    gf = group_gold_file(args.group)
    if not gf.exists():
        print(f"⚠️ 还没有组级题集：{gf}")
        return 1
    doc = json.loads(gf.read_text(encoding="utf-8"))
    corpus = list(doc.get("corpus") or [])
    qs = doc.get("questions") or []

    # 语料必须与 `_qa_groups.GROUPS` 一致（否则「篇 N」编号会与跑批时不一致）
    want = papers(args.group)
    bad = 0
    print(f"=== {gf.name} ===")
    print(f"  语料 {len(corpus)} 篇：{corpus}")
    if corpus != want:
        print(f"  ❌ corpus 与 _qa_groups.GROUPS[{args.group}] 不一致：期望 {want}")
        bad += 1
    else:
        print("  ✓ corpus 与 _qa_groups 一致（顺序即「篇 1..篇 N」）")

    # ── 配额 ──
    cnt: dict[str, int] = {}
    for q in qs:
        cnt[q.get("kind", "?")] = cnt.get(q.get("kind", "?"), 0) + 1
    print(f"  题数 {len(qs)}（期望 {group_expected_total(args.group)}）")
    for k, w in GROUP_QUOTA.items():
        got = cnt.get(k, 0)
        print(f"    {'✓' if got == w else '⚠️'} {k:<4} {got} / 期望 {w}")
    for k in cnt:
        if k not in GROUP_QUOTA:
            print(f"  ❌ 未知 kind {k!r}（只允许 {sorted(GROUP_QUOTA)}）")
            bad += 1

    # ── 材料（reach 用；只取所需，逐题打印便于定位）──
    # M0 的 reach 必须查**真实合并总览**（`report_l0` 对 5 篇的实际产出），
    # 不能查"各篇 L0 材料拼接"：后者信息量大得多，会放过答不出的题。
    # 实测 2026-09-23：M0-2 在拼接口径下通过，跑 `--l0` 时 judge 却判「不够」
    # （真实合并总览只有 20 条要点 / 1803 字符，且不含各篇的 limitations）。
    from paperpilot.agents.nodes.report import report_l0

    # ⚠️ 必须传 **`<stem>.pdf` 全名**：`report_l0` 内部用 `Path(pdf).stem` 定位报告，
    # 传 "2608.29179v1" 会被切成 "2608" → FileNotFoundError（2026-09-23 踩到）。
    _st0 = dict(report_l0({"question": "", "pdfs": [f"{s}.pdf" for s in corpus],   # type: ignore[arg-type]
                           "route": [], "debug": {}}))
    l0_all = "\n".join(
        [str(_st0.get("overview") or "")]
        + [str(c if isinstance(c, str) else (c.get("text") or c.get("rep_text") or ""))
           for c in (_st0.get("core_points") or [])]
        + [str(x if isinstance(x, str) else (x.get("text") or x.get("rep_text") or ""))
           for x in (_st0.get("limitations") or [])])
    print(f"  [材料] 合并总览 {len(l0_all)} 字（{len(_st0.get('core_points') or [])} 条要点）",
          end="", flush=True)
    l3_all, st = "", []
    for s in corpus:
        t, why = l3_text(s)
        l3_all += "\n" + t
        st.append(f"{s}:{why}")
    print(f" ｜ L3 并集 {len(l3_all)} 字（{', '.join(st)}）")

    # ── 逐题 ──
    for q in qs:
        qid = q.get("qid", "?")
        kind = q.get("kind", "?")
        miss = [k for k in REQ if k not in q]
        if miss:
            print(f"  ❌ {qid}: 缺字段 {miss}")
            bad += 1
        all_ = [x for x in (q.get("must_all") or []) if x]
        any_ = [x for x in (q.get("must_any") or []) if x]

        # 锚点规则：X5 靠拒绝措辞（must_any），其余必须有必现锚点
        if kind == "X5":
            if all_:
                print(f"  ❌ {qid}: X5 不该有 must_all（{all_}）—— 拒答题判据是措辞")
                bad += 1
            if not any_:
                print(f"  ❌ {qid}: X5 缺 must_any（拒绝措辞）")
                bad += 1
        else:
            if not all_:
                print(f"  ❌ {qid}: 缺 must_all（必现锚点，跨语言用专名/数字）")
                bad += 1
            if not any_ and not all_:
                print(f"  ❌ {qid}: 无判定锚点")
                bad += 1

        for key in ("expect", "route_min"):
            vals = q.get(key) if isinstance(q.get(key), list) else [q.get(key)]
            stale = [v for v in vals if v and v not in LEVELS]
            if stale:
                print(f"  ❌ {qid}: {key}={stale} 不在 {sorted(LEVELS)}")
                bad += 1

        # ── 引文逐字核对到**指定篇** ──
        evs = q.get("evidence") or []
        if not evs:
            print(f"  ❌ {qid}: evidence 为空")
            bad += 1
        src_cache: dict[str, list[str]] = {}
        ev_papers: set[str] = set()
        for ev in evs:
            p = str(ev.get("paper") or "")
            if p not in corpus:
                print(f"  ❌ {qid}: evidence.paper={p!r} 不在 corpus")
                bad += 1
                continue
            ev_papers.add(p)
            if p not in src_cache:
                t = material_dir(args.group) / f"{p}.txt"
                src_cache[p] = (sources_variants(t.read_text(encoding="utf-8"))
                                if t.exists() else [])
            quote = norm(ev.get("quote", ""))
            srcs = src_cache[p]
            if not quote:
                print(f"  ❌ {qid}: 空引文")
                bad += 1
            elif not srcs:
                print(f"  ⚠️ {qid}: 缺原文 {p}.txt → 跳过引文核对")
            elif not any(quote in s for s in srcs):
                print(f"  ❌ {qid} {p} p{ev.get('page')}: 引文不在该篇原文中")
                print(f"       {diverge(quote, srcs[1])}")
                bad += 1

        # ── 假跨篇检查 ──
        if kind == "M1" and len(ev_papers) < 2:
            print(f"  ❌ {qid}: M1 的 evidence 只来自 {len(ev_papers)} 篇（须 ≥2 篇才算跨篇）")
            bad += 1

        # ── reach ──
        if kind == "M0" and all_:
            gone = [a for a in all_ if a not in l0_all]
            flag = "✓" if not gone else "❌"
            print(f"  {flag} {qid} [M0] reach=L0  锚点 {all_}"
                  + (f"  ← 合并总览缺 {gone}（这题不是「不走检索可答」→ 改 M1 或换锚点）"
                     if gone else ""))
            if gone:
                bad += 1
        elif kind == "M1" and all_:
            gone = [a for a in all_ if a not in l3_all]
            flag = "✓" if not gone else "❌"
            print(f"  {flag} {qid} [M1] reach=L3  锚点 {all_}"
                  + (f"  ← 检索视图缺 {gone}（锚点越界）" if gone else ""))
            if gone:
                bad += 1
        elif kind == "X5":
            print(f"  ✓ {qid} [X5] 拒答措辞 {len(any_)} 条（reach 跳过）")

    print(f"\n{'=' * 62}\n组级合计 {len(qs)} 题（期望 {group_expected_total(args.group)}）"
          f"，问题 {bad} 处")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
