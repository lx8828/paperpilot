"""临时：由 top-dump 引发的两个补充测量（零新 encode，仅查 encode_query）。

1) 三路 gold rank 对比：hybrid(RRF60) / vec-only / bm-only / oracle(min)
   —— 判断"排不准"是融合的锅还是单路就找不到。
2) RRF k 扫描（k=5/10/20/60/100）：NDCG@12、gold@1、MRR
   —— dump 里 RRF 分数全挤在 0.029-0.033（k=60 太平），看收紧 k 是否有收益。
3) gold 所在节类别 + gold 块长度分布 + 节标题与问题的词重叠 × rank
   —— 验证"gold 标签本身可疑（附录/致谢/极小节）"的比例。
"""
from __future__ import annotations
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

NOISE = ("reference", "bibliograph", "acknowledg", "appendix", "supplement")
CATS = [("intro", ("introduction",)), ("concl", ("conclusion", "conclusions", "discussion")),
        ("results", ("result",)), ("method", ("method", "model", "approach", "architecture")),
        ("data", ("dataset", "data", "corpus", "corpora", "annotation")),
        ("exp", ("experiment", "evaluation", "setup", "ablation", "baseline")),
        ("style", ("section", "credit", "ruler", "instructions", "template", "acknowledg"))]
WORD = re.compile(r"[a-z0-9]+")


def cat_of(tp) -> str:
    l = ((list(tp)[-1] if tp else "") or "").lower()
    for name, kws in CATS:
        if any(k in l for k in kws):
            return name
    return "other"


def tok(s: str) -> set:
    return set(WORD.findall(s.lower()))


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    KS = [5, 10, 20, 60, 100]
    acc = {k: {"ndcg": 0.0, "top1": 0, "mrr": 0.0, "d": 0} for k in KS}
    route = {"hyb": [], "vec": [], "bm": [], "oracle": []}
    cat_c = Counter()
    g_len = []
    overlap = {"hit1": [], "hit2_3": [], "rest": []}
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
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _ = ree.locate_gold(ntexts, ev)
            if not cg:
                try:
                    cg = {int(np.argmax(np.asarray(bm.score(ree.norm(ev)), dtype="float64")))}
                except Exception:
                    continue
            gi = next(iter(cg))
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            ov = list(np.argsort(-vs))
            ob = list(np.argsort(-bs))
            rv = next(p for p in range(n) if ov[p] in cg)
            rb = next(p for p in range(n) if ob[p] in cg)
            route["vec"].append(rv)
            route["bm"].append(rb)
            route["oracle"].append(min(rv, rb))
            for k in KS:
                rrf = np.zeros(n)
                for r, i in enumerate(ov):
                    rrf[int(i)] += 1.0 / (k + r + 1)
                for r, i in enumerate(ob):
                    rrf[int(i)] += 1.0 / (k + r + 1)
                order = list(np.argsort(-rrf))
                rk = next((p for p in range(n) if order[p] in cg), None)
                a = acc[k]
                a["d"] += 1
                if rk == 0:
                    a["top1"] += 1
                if rk is not None and rk < 12:
                    a["ndcg"] += 1.0 / math.log2(rk + 2)
                if rk is not None and rk < 16:
                    a["mrr"] += 1.0 / (rk + 1)
                if k == 60:
                    route["hyb"].append(rk)
            gc = chunks[gi]
            cat_c[cat_of(gc.title_path)] += 1
            g_len.append(len(gc.text))
            qs = tok(it["question"])
            ss = tok(" ".join(list(gc.title_path) or []))
            jac = len(qs & ss) / max(len(qs | ss), 1)
            hr = route["hyb"][-1]
            if hr == 0:
                overlap["hit1"].append(jac)
            elif hr is not None and hr <= 2:
                overlap["hit2_3"].append(jac)
            else:
                overlap["rest"].append(jac)

    L = ["# TOP-AGG：排不准的机制拆解（base 官方口径）", ""]
    L.append("## 1) 三路 gold rank（rank 越小越好）")
    L.append("| 路 | gold@1 | gold@3 | MRR@16 | 均值rank |")
    L.append("|---|---|---|---|---|")
    for name in ("hyb", "vec", "bm", "oracle"):
        rr = [x for x in route[name] if x is not None]
        d = len(rr) or 1
        L.append(f"| {name} | {sum(1 for x in rr if x==0)/d:.3f} | "
                 f"{sum(1 for x in rr if x<=2)/d:.3f} | "
                 f"{sum(1/(x+1) for x in rr if x<16)/d:.3f} | {np.mean(rr):.1f} |")
    L.append("")
    L.append("## 2) RRF k 扫描（k 越小越锐利）")
    L.append("| k | NDCG@12 | gold@1 | MRR@16 |")
    L.append("|---|---|---|---|")
    for k in KS:
        a = acc[k]
        d = a["d"] or 1
        L.append(f"| {k} | {a['ndcg']/d:.4f} | {a['top1']/d:.4f} | {a['mrr']/d:.4f} |")
    L.append("")
    L.append("## 3) gold 所在节类别 / gold 块长度 / 节标题×问题词重叠")
    tot = sum(cat_c.values()) or 1
    L.append("| gold 节类别 | 题数 | 占比 |")
    L.append("|---|---|---|")
    for k, v in cat_c.most_common():
        L.append(f"| {k} | {v} | {v/tot*100:.1f}% |")
    L.append("")
    gl = sorted(g_len)
    L.append(f"- gold 块长度：中位 {np.median(gl):.0f} | p10 {gl[len(gl)//10]} | "
             f"p90 {gl[int(len(gl)*0.9)]} | <300 有 {sum(1 for x in gl if x<300)} 题")
    L.append("- 节标题与问题的词重叠(Jaccard) × rank：")
    for name, lab in [("hit1", "gold@1"), ("hit2_3", "gold@2-3"), ("rest", "gold≥4")]:
        xs = overlap[name]
        if xs:
            L.append(f"  - {lab}（n={len(xs)}）均值 {np.mean(xs):.3f} | "
                     f"=0 占比 {sum(1 for x in xs if x==0)/len(xs)*100:.0f}%")
    txt = "\n".join(L)
    Path("qa/recall/TOP_AGG_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
