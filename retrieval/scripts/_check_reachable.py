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

## ⚠️ `--search`：本检查**只看材料，不看检索**（2026-09-23 补）

上面两段判据用的是**材料**（L0 视图 / L3 检索视图=语料全集），所以它只能回答
"锚点在不在系统输入里"，**不能回答"检索器能不能把它捞进 top-k"**。
这正是"检索路径改了、测试没走"能长期隐藏的原因（锚点全在视图里 → 自检全绿 →
跑批却召回不到）。加 `--search` 后：用**生产同一条检索路径**
（`pull_chunk.search_l3` 的 `search_layered` + 同一个 `L3_TOP_K`）真跑一遍，
逐题报"必现锚点是否落在 top-k 文本里"。

用法：python retrieval/scripts/_check_reachable.py [--group group1] [--search]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import ROOT, gold_files, group_gold_file, material_dir, papers  # noqa: E402

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


def search_reach(corpus: list[str], qs: list[dict], title: str) -> tuple[int, int]:
    """用**生产检索路径**跑一遍，检查必现锚点是否落在 top-k 文本里。

    与线上完全同款调用（`agents/nodes/pull_chunk.search_l3`）：
        `MultiChunkIndex(pdfs).search_layered(question, top_k=L3_TOP_K)`
    `L3_TOP_K` 直接 import 生产常量 —— 不另留一份数字，避免"改了生产、检查器还按旧值"。

    返回 `(检查题数, 未命中题数)`。零 LLM 成本（只 encode query + 检索）。
    """
    from collections import Counter

    from paperpilot.agents.embedder import MultiChunkIndex

    try:
        from paperpilot.agents.nodes.pull_chunk import L3_TOP_K
    except Exception:  # noqa: BLE001
        L3_TOP_K = 12

    idx = MultiChunkIndex(corpus)
    print(f"\n=== {title} ===")
    print(f"    生产检索路径 search_layered(mode=quota, top_k={L3_TOP_K})"
          f" ｜ 语料 {len(corpus)} 篇 / 单元 {len(idx._doc_chunks())}")
    print(f"{'qid':<14}{'命中篇':<8}{'锚点命中':<10}缺失锚点")
    n = miss = 0
    for q in qs:
        cat = str(q.get("category") or "")
        kind = str(q.get("kind") or "")
        # **只查"需要下钻检索"的题**：
        #   L0 / M0 = 免检索（Router 判够即直答，锚点应是中文、只该在 L0 材料里，
        #             由本文件上半段的材料级 reach 负责，用检索去查它们是误报）
        #   negative / X5 = 拒答题，判据是措辞、没有必现锚点
        if cat in ("L0", "negative") or kind in ("M0", "X5"):
            continue
        primary, _tolerant = anchors(q)
        if not primary:
            continue
        n += 1
        hits = idx.search_layered(str(q.get("question") or ""), top_k=L3_TOP_K)
        txt = "\n".join(str(h.get("text") or "") for h in hits)
        dist = Counter(Path(str(h.get("pdf"))).stem for h in hits)
        gone = [a for a in primary if a not in txt]
        if gone:
            miss += 1
        ratio = f"{len(primary) - len(gone)}/{len(primary)}"
        row = f"{str(q.get('qid')):<14}{len(dist):<8}{ratio:<10}"
        print(row + ("—" if not gone else f"❌ {gone}"))
    print(f"  → 检索未命中 {miss}/{n}"
          + ("（锚点在材料里但**捞不进 top-k** → 要么调检索/加大 top_k，要么换锚点）"
             if miss else "（全部可检索命中）"))
    return n, miss


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
    ap.add_argument("--search", action="store_true",
                    help="额外用**生产检索路径**真跑一遍，检查锚点能否落进 top-k"
                         "（材料级自检之外的检索级自检；零 LLM 成本）")
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
            # ① 可答性硬门槛：**必现锚点**（must_all）必须逐字出现在"这题真的会看到的那份材料"里。
            #    ⚠️ 2026-09-23 修：**L0 题要对 L0 材料（t0）判，不能对 L3 检索视图判**。
            #    L0/M0 是免检索（Router 判够即直答），其锚点按范式是**中文**（L0 材料是中文
            #    overview/core_points）→ 拿去和英文检索视图比对必然 MISS = **误报**。
            #    原先未分流，group2 上产生 3 处误报（0087-L0-3 / 5389-L0-3 / 2094-L0-3）。
            #    其余题（需下钻）仍对 L3 检索视图判。
            if not primary:
                reach, note = "MISS", "缺 must_all"
                miss_total += 1
            elif q.get("category") == "L0":
                gone0 = [a for a in primary if a not in t0]
                if not gone0:
                    reach, note = "L0", ""
                elif all(a in t3 for a in gone0):
                    # L0 材料里没有、检索视图里才有 → 这题其实是 single 题（mislabel）
                    reach = "L3"
                    l0_bad += 1
                    note = "L0 材料没有、检索视图才有 → 改标 single 或换 L0 锚点"
                else:
                    reach = "MISS"
                    miss_total += 1
                    note = "L0 材料缺: " + str([a for a in gone0 if a not in t3])
            else:
                gone = [a for a in primary if a not in t3]
                if gone:
                    reach = "MISS"
                    miss_total += 1
                    note = "L3 视图缺: " + str(gone)
                else:
                    reach, note = ("L0" if all(a in t0 for a in primary) else "L3"), ""
                    if reach == "L3":
                        note = "必现锚点不在 L0 材料"
            hint = '["L0","L3"]' if reach == "L0" else ('["L3"]' if reach == "L3" else "人工改题")
            flag = "❌" if (q.get("category") == "L0" and reach == "L3") else " "
            print(f"{qid:<14}{reach:<6}{hint:<18}{flag} {note}")
    n_search = miss_search = 0
    if args.search:
        # ⚠️ 生产语料恒为**整组 5 篇**（本系统没有单篇路径）→ 单篇题也在 5 篇语料里检索，
        #    与跑批的 A 组同口径。组级 M1 题一并检查（M0 免检索、X5 无锚点，跳过）。
        corpus = [f"{s}.pdf" for s in papers(args.group)]
        sq: list[dict] = []
        for stem in papers(args.group):
            gf = material_dir(args.group) / f"{stem}.questions.json"
            if gf.exists():
                sq += json.loads(gf.read_text(encoding="utf-8"))["questions"]
        gfile = group_gold_file(args.group)
        if gfile.exists():
            sq += json.loads(gfile.read_text(encoding="utf-8"))["questions"]
        n_search, miss_search = search_reach(
            corpus, sq, f"检索级自检（{args.group}：单篇 single/table + 组级 M1 @ 5 篇语料）")

    print(f"\n越界/缺锚点题：{miss_total} ｜ L0 类题却需下钻：{l0_bad}"
          + (f" ｜ 检索未命中：{miss_search}/{n_search}" if args.search else ""))
    return 1 if (miss_total or l0_bad or miss_search) else 0


if __name__ == "__main__":
    raise SystemExit(main())
