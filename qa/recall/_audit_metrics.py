"""临时（唯一口径）：官方分母对齐的 Recall/MRR/NDCG/hit 多维对比。

先复现官方 RECALL_BASELINE 已知值校验分母（应≈R@8 0.86 / R@12 0.96 / MRR 0.465，
官方 20% fallback 兜底乐观 → 本脚本含同 fallback），再读策略列：
  base    = 现状 hybrid top_k（无 cap/decay）
  full1   = 完整 title_path cap=1
  full1d7 = 完整 title_path cap=1 + Overview decay 0.7
  full2   = 完整 title_path cap=2
分母 = locate_gold 成功（含 fallback）的全部题——gold 不在 top16 也算分母（真 Recall）。
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
import run_retrieval_eval as ree  # noqa: E402 (norm / locate_gold 官方口径)

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

KS = (8, 12, 16)
_OV = ("introduction", "conclusion", "abstract", "related work", "background")
STRATS = {"base": (0, 1.0), "full1": (1, 1.0), "full1d7": (1, 0.7), "full2": (2, 1.0)}


def is_overview(path) -> bool:
    return bool(path) and any(k in str(path[0]).lower() for k in _OV)


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    # 聚合（分母=可定位题数，全部策略共享）
    agg = {s: {"R": {k: 0 for k in KS}, "MRR16": 0.0, "hit12": 0, "ndcg12": 0.0,
               "hit": 0} for s in STRATS}
    denom = 0
    map_fail = 0
    rank_of = {s: {} for s in STRATS}  # qid -> gold 首个命中的 0-based rank
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
                map_fail += 1
                continue
            cg, _ = ree.locate_gold(ntexts, ev)
            if not cg:  # 官方 fallback：evidence-BM25 自检索 top1
                try:
                    best = int(np.argmax(np.asarray(bm.score(ree.norm(ev)), dtype="float64")))
                    cg = {best}
                except Exception:
                    map_fail += 1
                    continue
            if not cg:
                map_fail += 1
                continue
            denom += 1
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            n = len(vs)
            rrf = np.zeros(n)
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for s, (cap, decay) in STRATS.items():
                rr = rrf.copy()
                if decay < 1.0:
                    w = np.array([is_overview(getattr(c, "title_path", None)) for c in chunks])
                    rr = rr * np.where(w, decay, 1.0)
                order = list(np.argsort(-rr))
                picked = order[: max(KS)]
                if cap > 0:
                    out, used = [], {}
                    for idx in order:
                        key = tuple(getattr(chunks[idx], "title_path", None) or [])
                        if used.get(key, 0) >= cap:
                            continue
                        out.append(int(idx)); used[key] = used.get(key, 0) + 1
                        if len(out) >= max(KS):
                            break
                    picked = out
                pos = {i: r for r, i in enumerate(picked)}
                first = min((pos[i] for i in cg if i in pos), default=None)
                a = agg[s]
                if first is None:
                    continue  # 分母仍计（已 +1），此处只是无命中
                for k in KS:
                    if first < k:
                        a["R"][k] += 1
                if first < 16:
                    a["MRR16"] += 1.0 / (first + 1)
                if first < 12:
                    a["hit12"] += 1
                    a["ndcg12"] += 1.0 / math.log2(first + 2)
                rank_of[s][it["qid"]] = first

    L = ["# 多维检索指标（官方分母口径，样本 v1）", ""]
    L.append(f"> 分母={denom}（locate 成功，含 fallback）| map_fail={map_fail}")
    L.append("| 策略 | R@8 | R@12 | R@16 | MRR@16 | hit@12 | NDCG@12 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in STRATS:
        a = agg[s]; d = max(denom, 1)
        L.append(f"| {s} | {a['R'][8]/d:.3f} | {a['R'][12]/d:.3f} | {a['R'][16]/d:.3f} | "
                 f"{a['MRR16']/d:.3f} | {a['hit12']/d:.3f} | {a['ndcg12']/d:.3f} |")
    # 对账提示
    L.append("")
    L.append(f"> 对账：官方 RECALL_BASELINE hybrid ≈ R@8 .86 R@12 .96 MRR .465 —— 本 base 行应接近")
    # 位移诊断 full1d7 vs base
    b = set(rank_of["base"]); f = set(rank_of["full1d7"])
    both12 = {q for q in b & f if rank_of["base"][q] < 12 and rank_of["full1d7"][q] < 12}
    dropped12 = {q for q in b & f if rank_of["base"][q] < 12 <= rank_of["full1d7"][q]}
    up12 = {q for q in b & f if rank_of["full1d7"][q] < 12 <= rank_of["base"][q]}
    L.append("")
    L.append("位移诊断（full1d7 vs base，gold 是否进实际上下文 top12）：")
    L.append(f"  两者都在12内 {len(both12)} | base在12内但full1d7掉出 {len(dropped12)} | "
             f"full1d7新进12 {len(up12)}")
    L.append(f"  MRR 提升题数: {sum(1 for q in b & f if rank_of['full1d7'][q] < rank_of['base'][q])} "
             f"| 变差: {sum(1 for q in b & f if rank_of['full1d7'][q] > rank_of['base'][q])}")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/AUDIT_METRICS_20260909.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
