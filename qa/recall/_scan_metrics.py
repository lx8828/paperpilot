"""临时：多维离线指标 + rank 位移诊断（不只 R@8）。

指标：Recall@8/12、MRR@16、NDCG@12、hit@12（gold 是否进实际上下文 top12）。
对比：baseline vs fullpath cap1 vs fullpath cap1+decay0.7。
位移诊断：full1+d0.7 相对 baseline 每题 gold rank 变化——
  仍在前12(能答) / 掉出12(可能影响) / 提升。量化"R@8 掉的题到底伤没伤实际上下文"。
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

KS = (8, 12, 16)
_OV = ("introduction", "conclusion", "abstract", "related work", "background")
CFGS = ["baseline", "full1/d1.0", "full1/d0.7"]


def is_overview(path) -> bool:
    if not path:
        return False
    return any(k in str(path[0]).lower() for k in _OV)


def build_rrf(vs, bs):
    n = len(vs)
    rrf = np.zeros(n)
    for r, i in enumerate(np.argsort(-vs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    for r, i in enumerate(np.argsort(-bs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    return rrf


def pick(rrf, chunks, top_k, cap, decay):
    if decay < 1.0:
        w = np.array([is_overview(getattr(c, "title_path", None)) for c in chunks])
        rrf = rrf * np.where(w, decay, 1.0)
    order = list(np.argsort(-rrf))
    if cap <= 0:
        return order[:top_k]
    out, used = [], {}
    for idx in order:
        key = tuple(getattr(chunks[int(idx)], "title_path", None) or [])
        if used.get(key, 0) >= cap:
            continue
        out.append(int(idx)); used[key] = used.get(key, 0) + 1
        if len(out) >= top_k:
            break
    return out


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    plan = {"baseline": (0, 1.0), "full1/d1.0": (1, 1.0), "full1/d0.7": (1, 0.7)}

    # 每 cfg 聚合 + 逐题 gold rank（全序）供位移诊断
    agg = {c: {"denom": 0, "R": {k: 0 for k in KS}, "MRR": 0.0, "NDCG12": 0.0,
               "hit12": 0} for c in CFGS}
    ranks = {c: {} for c in CFGS}  # qid -> gold rank(0-based, 全序)
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
                continue
            grp = it["group"]
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            rrf = build_rrf(vs, bs)
            # 全序 gold rank（cap/decay 前的 RRF 全序）
            full_order = list(np.argsort(-rrf))
            gold_first_full = next((p for p in range(n) if full_order[p] in cg), None)
            for cfg in CFGS:
                cap, decay = plan[cfg]
                picked = pick(rrf.copy(), chunks, max(KS), cap, decay)
                pos = {i: r for r, i in enumerate(picked)}
                first = min((pos[i] for i in cg if i in pos), default=None)
                if first is not None:
                    ranks[cfg][it["qid"]] = first  # 0-based 在最终列表
                a = agg[cfg]
                a["denom"] += 1
                if first is None:
                    continue
                for k in KS:
                    if first < k:
                        a["R"][k] += 1
                if first < 16:
                    a["MRR"] += 1.0 / (first + 1)
                if first < 12:
                    a["hit12"] += 1
                    a["NDCG12"] += 1.0 / math.log2(first + 2)  # binary rel, gain=1

    L = ["# 多维指标 + 位移诊断（不只 R@8）", "",
         "| config | R@8 | R@12 | hit@12 | MRR@16 | NDCG@12 |"]
    L.append("|---|---|---|---|---|---|")
    for cfg in CFGS:
        a = agg[cfg]; d = max(a["denom"], 1)
        L.append(f"| {cfg} | {a['R'][8]/d:.3f} | {a['R'][12]/d:.3f} | "
                 f"{a['hit12']/d:.3f} | {a['MRR']/d:.3f} | {a['NDCG12']/d:.3f} |")
    # 位移诊断：d0.7 vs baseline
    common = set(ranks["baseline"]) & set(ranks["full1/d0.7"])
    base12 = {q for q in ranks["baseline"] if ranks["baseline"][q] < 12}
    d07_12 = {q for q in ranks["full1/d0.7"] if ranks["full1/d0.7"][q] < 12}
    dropped = base12 - d07_12          # baseline 在12内、d0.7 掉出12
    gained = d07_12 - base12
    kept = base12 & d07_12
    L.append("")
    L.append(f"位移诊断（full1/d0.7 vs baseline，仅比 gold 能否进实际上下文 top12）:")
    L.append(f"  两边都在12内: {len(kept)} | baseline在12内但d0.7掉出12: {len(dropped)} | "
             f"d0.7新进12: {len(gained)}")
    # R@8 掉的题里有多少其实仍在 12 内（= 假掉）
    base8 = {q for q in ranks["baseline"] if ranks["baseline"][q] < 8}
    d078 = {q for q in ranks["full1/d0.7"] if ranks["full1/d0.7"][q] < 8}
    r8_dropped = base8 - d078
    r8_dropped_still12 = {q for q in r8_dropped if q in d07_12}
    L.append(f"  R@8 掉的题: {len(r8_dropped)} | 其中仍在前12(能答): {len(r8_dropped_still12)} "
             f"({len(r8_dropped_still12)/max(len(r8_dropped),1):.0%}) | 真正掉出12: "
             f"{len(r8_dropped)-len(r8_dropped_still12)}")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/METRICS_SCAN_20260909.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
