"""题集**可答性自检**：每题的判定锚点是否真的落在系统能看到的材料里。

为什么必须有这一步（gold 引文 ≠ 系统答得出）：
    `_validate_questions.py` 只保证"引文在论文里有"，**不保证系统答得出**。现行问答链
    只用**两处材料**（`graph/qa_graph_v3.py`，2026-09-22 核实）：

        L0 直答  overview.json 的 overview  +  report.json 的 core_points / limitations
        L3 检索  document_cache.retrieval_chunks(pdf)（默认 MinerU 路 / 备用 pymupdf 路）

    锚点只出现在论文但**两处都没有** → 这题是**越界题**（系统输入里根本没有），
    跑出来必然 ⚠️，却是题的问题不是系统的问题。必须提前发现。

判据与建议：
    锚点**全部**在 L0 材料 → `reach=L0` → expect 建议 ["L0","L3"]（测 Router 判够）
    仅全部在 L3 材料       → `reach=L3` → expect 建议 ["L3"]（必须下钻）
    有锚点两边都缺         → `reach=MISS`（列出缺的锚点，人工改题或降级锚点）
    `category=negative` 的题跳过：**"论文没给"本身就是答案**，锚点是拒答措辞。

用法：python retrieval/scripts/_check_reachable.py [--group group1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import ROOT, gold_files, material_dir, papers  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))


def l0_text(stem: str) -> str:
    """L0 直答能看到的全部文本：overview + core_points + limitations。"""
    view = ROOT / "assets/artifacts/out_views"
    parts: list[str] = []
    ov = view / f"{stem}.overview.json"
    if ov.exists():
        parts.append(str(json.loads(ov.read_text(encoding="utf-8")).get("overview") or ""))
    rp = view / f"{stem}.report.json"
    if rp.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
        parts.append(str(rep.get("overview") or ""))
        for cp in rep.get("core_points") or []:
            parts.append(str(cp if isinstance(cp, str) else cp.get("rep_text") or cp.get("text") or ""))
        for lim in rep.get("limitations") or []:
            parts.append(str(lim if isinstance(lim, str) else lim.get("rep_text") or lim.get("text") or ""))
    return "\n".join(parts)


def l3_text(stem: str) -> tuple[str, str]:
    """L3 检索视图文本（走真实 document_cache，与线上同一份材料）。"""
    from paperpilot.agents import document_cache
    pdf = f"{stem}.pdf"
    chunks = document_cache.retrieval_chunks(pdf)
    status, why = document_cache.mineru_status(pdf)
    return "\n".join(c.text for c in chunks), f"{status}{'' if status == 'ok' else f'({why[:40]})'}"


def anchors(q: dict) -> tuple[list[str], list[str]]:
    """返回 (必现锚点, 备选措辞)。

    可答性只看**必现锚点**（`must_all`）：它们必须在系统材料里逐字可命中。
    备选措辞（`must_any`，对应 runner 的 `must_have`"任一命中"）只是**容忍表述差异**，
    允许只出现在中文答案里，不参与 `reach` 判定（只做提示）。
    旧 schema 兼容：`must_have` 当年是"必现"语义。
    """
    all_ = list(q.get("must_all") or q.get("must_have") or [])
    any_ = list(q.get("must_any") or [])
    return [x for x in all_ if x], [x for x in any_ if x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="group1")
    args = ap.parse_args()

    miss_total = l0_bad = 0
    print(f"{'qid':<14}{'reach':<6}{'建议 expect':<18}缺失锚点")
    for stem in papers(args.group):
        gf = material_dir(args.group) / f"{stem}.questions.json"
        if not gf.exists():
            print(f"⚠️ 缺 {gf.name}")
            continue
        qs = json.loads(gf.read_text(encoding="utf-8"))["questions"]
        t0 = l0_text(stem)
        t3, st = l3_text(stem)
        print(f"\n=== {stem}（L0 材料 {len(t0)} 字 / L3 视图 {len(t3)} 字 · mineru={st}）===")
        for q in qs:
            qid = q["qid"]
            if q.get("category") == "negative":
                print(f"{qid:<14}{'—':<6}{'(拒答题，跳过)':<18}")
                continue
            primary, tolerant = anchors(q)
            if not primary and not tolerant:
                print(f"{qid:<14}{'MISS':<6}{'必须补锚点':<18}（must_all 与 must_any 皆空）")
                miss_total += 1
                continue
            # ① 可答性硬门槛：**必现锚点**（must_all）必须逐字出现在 L3 检索视图里。
            #    纯 must_any 的题无从自检 → 要求补 must_all（校验器同规则）。
            gone = [a for a in primary if a not in t3]
            if not primary or gone:
                reach = "MISS"
                miss_total += 1
                note = "缺 must_all" if not primary else "L3 视图缺: " + str(gone)
            else:
                # ② 层级：**必现**锚点是否也出现在 L0 材料（中文 overview/core_points）里。
                #    L0 命中 → L0 直答必然能判过；只在 L3 → 得下钻检索（或靠 L0 答案的英文引用）才命中。
                l0_hit = all(a in t0 for a in primary)
                reach = "L0" if l0_hit else "L3"
                note = "" if l0_hit else "必现锚点不在 L0 材料"
            hint = '["L0","L3"]' if reach == "L0" else ('["L3"]' if reach == "L3" else "人工改题")
            flag = " "
            if q.get("category") == "L0" and reach == "L3":
                # L0 类题 = **免检索**：必现锚点必须能在 L0 材料（overview/core_points/
                # limitations）里找到。若锚点只在检索视图里，这题其实是 single 题（mislabel）。
                flag = "❌"
                l0_bad += 1
                note = (note + " ｜ L0 题却必须下钻：改标 single 或换 L0 锚点").strip()
            print(f"{qid:<14}{reach:<6}{hint:<18}{flag} {note}")
    print(f"\n越界/缺锚点题：{miss_total} ｜ L0 类题却需下钻：{l0_bad}")
    return 1 if (miss_total or l0_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
