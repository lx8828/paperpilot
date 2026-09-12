"""临时：gold 口径对比——只取 evidence[0]（现行） vs 取全部 evidence 段（QASPER 官方口径）。

动机：QASPER 单题常有多段人工 evidence（38% 的题 ≥2 段），而 gold_answer() 只返回首段。
      若检索命中第 2/3 段（同样正确）会被计为 miss，系统性压低 NDCG/MRR。
本脚本量化该口径缺陷的影响，其余完全不变（同 chunk、同 base 向量缓存、同 RRF60 融合）。
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
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402


def all_evidence(q: dict) -> list[str]:
    """镜像 gold_answer 的答案选取优先级，但返回该答案的全部 evidence 段。"""
    for a in q.get("answers") or []:
        inner = a.get("answer") or {}
        if inner.get("unanswerable"):
            continue
        if inner.get("free_form_answer") or inner.get("yes_no") is not None \
                or inner.get("extractive_spans"):
            return [e for e in (inner.get("evidence") or []) if e]
    return []


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    res = {"single(now)": [], "multi(official)": []}
    n_multi_only = 0     # 单段口径下非#1 但多段口径下 #1
    n_multi_gt = 0       # 多段口径 rank 更好
    n_multi_lt = 0
    n_ev2plus = 0
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
        vecs = ChunkIndex(pdf).vectors()
        bm = BM25Index([c.text for c in chunks])
        ntexts = [ree.norm(c.text) for c in chunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            evs = all_evidence(q)
            if not evs:
                continue
            if len(evs) >= 2:
                n_ev2plus += 1
            cg_s, _ = ree.locate_gold(ntexts, evs[0])
            cg_m: set[int] = set()
            for e in evs:
                got, _ = ree.locate_gold(ntexts, e)
                cg_m |= got
            if not cg_s and not cg_m:
                continue
            if not cg_s:
                cg_s = set(cg_m)
            if not cg_m:
                cg_m = set(cg_s)
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            rrf = np.zeros(n)
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order = list(np.argsort(-rrf))

            def rank(cg):
                return next((p for p in range(len(order)) if order[p] in cg), None)

            rs, rm = rank(cg_s), rank(cg_m)
            res["single(now)"].append(rs)
            res["multi(official)"].append(rm)
            if rs != rm:
                if rm is not None and (rs is None or rm < rs):
                    n_multi_gt += 1
                    if rm == 0 and rs != 0:
                        n_multi_only += 1
                else:
                    n_multi_lt += 1

    L = ["# gold 口径对比：evidence[0] vs 全部 evidence 段", ""]
    L.append(f"> 题 {len(res['single(now)'])} | 有多段 evidence 的题 {n_ev2plus}")
    L.append(f"> 多段口径改善 {n_multi_gt} 题 / 变差 {n_multi_lt} 题 | 其中"
             f"'单段非#1 → 多段#1' {n_multi_only} 题")
    L.append("")
    L.append("| 口径 | gold@1 | gold@3 | R@8 | R@12 | R@16 | MRR@16 | NDCG@12 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in ("single(now)", "multi(official)"):
        rr = res[name]
        d = len(rr) or 1
        t1 = sum(1 for x in rr if x == 0) / d
        t3 = sum(1 for x in rr if x is not None and x <= 2) / d
        h8 = sum(1 for x in rr if x is not None and x < 8) / d
        h12 = sum(1 for x in rr if x is not None and x < 12) / d
        h16 = sum(1 for x in rr if x is not None and x < 16) / d
        mrr = sum(1 / (x + 1) for x in rr if x is not None and x < 16) / d
        ndcg = sum(1 / math.log2(x + 2) for x in rr if x is not None and x < 12) / d
        L.append(f"| {name} | {t1:.3f} | {t3:.3f} | {h8:.3f} | {h12:.3f} | "
                 f"{h16:.3f} | {mrr:.3f} | {ndcg:.3f} |")
    txt = "\n".join(L)
    Path("qa/recall/GOLD_MULT_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
