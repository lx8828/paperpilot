"""临时：融合策略扫描——oracle(单路最好) 远超 hybrid，说明 RRF 融合在丢分。

候选融合（全部复用缓存向量，仅 encode_query）：
  1) RRF(k)：等权
  2) wRRF：加权 RRF（vec 权重 wv）
  3) topk-RRF：每路只贡献前 TK 名（弱路噪声不进榜）
  4) minmax 凸组合：a*vec_norm + (1-a)*bm_norm
  5) zscore 凸组合
输出 NDCG@12 / gold@1 / gold@3 / MRR@16，并统计"融合比 oracle 差多少"。
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
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def zs(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    return (x - float(x.mean())) / s if s > 1e-9 else np.zeros_like(x)


def rank_of(order, cg, k_lim=None):
    for p, i in enumerate(order):
        if i in cg:
            return p
    return None


def rrf_order(vs, bs, k, wv=1.0, wb=1.0, tk=None):
    n = len(vs)
    ov = list(np.argsort(-vs))
    ob = list(np.argsort(-bs))
    if tk:
        ov, ob = ov[:tk], ob[:tk]
    s = np.zeros(n)
    for r, i in enumerate(ov):
        s[int(i)] += wv / (k + r + 1)
    for r, i in enumerate(ob):
        s[int(i)] += wb / (k + r + 1)
    return list(np.argsort(-s))


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    fusions = {
        "rrf_k10": lambda vs, bs: rrf_order(vs, bs, 10),
        "rrf_k60(now)": lambda vs, bs: rrf_order(vs, bs, 60),
        "wrrf_k10_v.7": lambda vs, bs: rrf_order(vs, bs, 10, 0.7, 0.3),
        "wrrf_k10_v.6": lambda vs, bs: rrf_order(vs, bs, 10, 0.6, 0.4),
        "wrrf_k10_v.8": lambda vs, bs: rrf_order(vs, bs, 10, 0.8, 0.2),
        "top20rrf_k10": lambda vs, bs: rrf_order(vs, bs, 10, 1.0, 1.0, 20),
        "top10rrf_k10": lambda vs, bs: rrf_order(vs, bs, 10, 1.0, 1.0, 10),
        "minmax_a.5": lambda vs, bs: list(np.argsort(-(0.5 * norm01(vs) + 0.5 * norm01(bs)))),
        "minmax_a.6": lambda vs, bs: list(np.argsort(-(0.6 * norm01(vs) + 0.4 * norm01(bs)))),
        "zscore_a.5": lambda vs, bs: list(np.argsort(-(0.5 * zs(vs) + 0.5 * zs(bs)))),
        "zscore_a.6": lambda vs, bs: list(np.argsort(-(0.6 * zs(vs) + 0.4 * zs(bs)))),
    }
    acc = {k: {"ndcg": 0.0, "t1": 0, "t3": 0, "mrr": 0.0, "d": 0} for k in fusions}
    acc["vec_only"] = {"ndcg": 0.0, "t1": 0, "t3": 0, "mrr": 0.0, "d": 0}
    acc["bm_only"] = {"ndcg": 0.0, "t1": 0, "t3": 0, "mrr": 0.0, "d": 0}
    acc["oracle"] = {"ndcg": 0.0, "t1": 0, "t3": 0, "mrr": 0.0, "d": 0}
    fusion_hurt = 0     # hybrid 排位差于 oracle
    fusion_help = 0
    n = 0
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
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
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            n += 1

            def upd(name, order):
                a = acc[name]
                r = rank_of(order, cg)
                a["d"] += 1
                if r == 0:
                    a["t1"] += 1
                if r is not None and r <= 2:
                    a["t3"] += 1
                if r is not None and r < 12:
                    a["ndcg"] += 1.0 / math.log2(r + 2)
                if r is not None and r < 16:
                    a["mrr"] += 1.0 / (r + 1)

            upd("vec_only", list(np.argsort(-vs)))
            upd("bm_only", list(np.argsort(-bs)))
            rv = rank_of(list(np.argsort(-vs)), cg)
            rb = rank_of(list(np.argsort(-bs)), cg)
            if rv is not None and rb is not None:
                # oracle: 取两路各自排序中的较优名次（用假 order 近似）
                best = min(rv, rb)
                a = acc["oracle"]
                a["d"] += 1
                if best == 0:
                    a["t1"] += 1
                if best <= 2:
                    a["t3"] += 1
                a["ndcg"] += 1.0 / math.log2(best + 2) if best < 12 else 0
                a["mrr"] += 1.0 / (best + 1) if best < 16 else 0
            for name, fn in fusions.items():
                upd(name, fn(vs, bs))
            hy = rank_of(rrf_order(vs, bs, 60), cg)
            best = min([x for x in (rv, rb) if x is not None]) if (rv is not None or rb is not None) else None
            if hy is not None and best is not None:
                if hy > best:
                    fusion_hurt += 1
                elif hy < best:
                    fusion_help += 1

    L = ["# 融合策略扫描（base 向量缓存复用，仅 encode_query）", "",
         f"> 题 {n} | 融合优于oracle {fusion_help} / 差于oracle {fusion_hurt}", "",
         "| 策略 | NDCG@12 | gold@1 | gold@3 | MRR@16 |", "|---|---|---|---|---|"]
    order_names = ["oracle", "vec_only", "bm_only"] + list(fusions)
    for name in order_names:
        a = acc[name]
        d = a["d"] or 1
        L.append(f"| {name} | {a['ndcg']/d:.4f} | {a['t1']/d:.4f} | {a['t3']/d:.4f} | {a['mrr']/d:.4f} |")
    txt = "\n".join(L)
    Path("qa/recall/FUSION_PROBE_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
