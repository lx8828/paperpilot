"""临时：因果检验——"碎块导致排序差"是真因果还是相关？

方法：按 base 下 gold 所在块的字符数把题分组（碎块 ≤1000 / 中块 / 大块），
再交叉看同一批题在 B2(target1600, gold 块被撑大) 下的表现。
    - 若"碎块→差"是真因果：碎块组题在 B2 下应显著变好（gold 块变大）；
    - 若只是相关（难题恰落碎块）：碎块组题在 B2 下不变甚至更差。
只读：rows 已有 base/b2 rank；每篇 base chunks 算 gold 块大小。
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    rows = {}
    for l in Path("qa/recall/_b2_rows_t1600.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["qid"]] = r
    # qid → base gold chunk 大小
    per_paper: dict[str, list[dict]] = {}
    for it in data["items"]:
        per_paper.setdefault(it["pid"], []).append(it)
    recs = []
    for pid, its in per_paper.items():
        bchunks = ordered_chunks(f"qasper_{pid}.qpdf")
        ntexts = [ree.norm(c.text) for c in bchunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            r = rows.get(it["qid"])
            if not r:
                continue
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _ = ree.locate_gold(ntexts, ev)
            if not cg:
                continue
            gi = next(iter(cg))
            recs.append({
                "qid": it["qid"], "grp": it["group"],
                "base_len": len(bchunks[gi].text), "base_blocks": bchunks[gi].n_blocks,
                "base": r["base"], "b2": r["b2"],
            })
    # 分组统计：每组 base vs b2 的 MRR/NDCG/gold@1 + B2 净位移
    def agg(sub):
        d = len(sub)
        def mrr_key(f):
            return f if f is not None and f < 16 else None
        out = {}
        for col in ("base", "b2"):
            vals = [1.0 / (r[col] + 1) for r in sub
                    if r[col] is not None and r[col] < 16]
            ndc = [1.0 / math.log2(r[col] + 2) for r in sub
                   if r[col] is not None and r[col] < 12]
            top1 = sum(1 for r in sub if r[col] == 0)
            out[col] = {"mrr": sum(vals) / d if d else 0,
                        "ndcg": sum(ndc) / d if d else 0,
                        "top1": top1 / d if d else 0}
        gain = sum(1 for r in sub if r["base"] is not None and r["b2"] is not None
                   and r["b2"] < r["base"])
        lose = sum(1 for r in sub if r["base"] is not None and r["b2"] is not None
                   and r["b2"] > r["base"])
        return out, gain, lose, d

    L = ["# 因果检验：碎块是检索差的因，还是难题的外观特征？", ""]
    L.append("> 同批题（按 base 下 gold 块大小分组）在 base vs B2 的表现；"
             "若碎块是真因 → 碎块组 B2 应变好")
    L.append("")
    L.append("| 分组(base gold块) | n | base MRR | b2 MRR | base NDCG | b2 NDCG | "
             "base gold@1 | b2 gold@1 | B2变好 | B2变差 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    groups = [
        ("≤1000（碎块）", lambda r: r["base_len"] <= 1000),
        ("1001-2500", lambda r: 1000 < r["base_len"] <= 2500),
        ("2501-4000", lambda r: 2500 < r["base_len"] <= 4000),
    ]
    for name, f in groups:
        sub = [r for r in recs if f(r)]
        if not sub:
            continue
        out, gain, lose, d = agg(sub)
        L.append(f"| {name} | {d} | {out['base']['mrr']:.3f} | {out['b2']['mrr']:.3f} | "
                 f"{out['base']['ndcg']:.3f} | {out['b2']['ndcg']:.3f} | "
                 f"{out['base']['top1']:.3f} | {out['b2']['top1']:.3f} | "
                 f"{gain} | {lose} |")
    sub = recs
    out, gain, lose, d = agg(sub)
    L.append(f"| ALL | {d} | {out['base']['mrr']:.3f} | {out['b2']['mrr']:.3f} | "
             f"{out['base']['ndcg']:.3f} | {out['b2']['ndcg']:.3f} | "
             f"{out['base']['top1']:.3f} | {out['b2']['top1']:.3f} | "
             f"{gain} | {lose} |")
    # 按段落数再分
    L.append("")
    L.append("| 分组(base gold段落数) | n | base MRR | b2 MRR | base NDCG | b2 NDCG | B2变好 | B2变差 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, f in [
        ("1段", lambda r: r["base_blocks"] == 1),
        ("2段", lambda r: r["base_blocks"] == 2),
        ("≥3段", lambda r: r["base_blocks"] >= 3),
    ]:
        sub = [r for r in recs if f(r)]
        if not sub:
            continue
        out, gain, lose, d = agg(sub)
        L.append(f"| {name} | {d} | {out['base']['mrr']:.3f} | {out['b2']['mrr']:.3f} | "
                 f"{out['base']['ndcg']:.3f} | {out['b2']['ndcg']:.3f} | "
                 f"{gain} | {lose} |")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/B2_CAUSE_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
