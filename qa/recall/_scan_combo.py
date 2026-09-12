"""临时：联合扫描——先 fullpath cap1 理顺候选，再叠加 M2 概述降权（decay）。

验证用户假设：M2 惩罚之前"无效/伤召回"可能是假的——因为它在未修正候选
（Intro×N 挤占 + 顶层节误伤）上测。正确去重后 M2 净效果可能不同。

CONFIGS = (cap_mode, decay)：
  baseline        = 无 cap 无降权（现状）
  full1/decay1.0  = fullpath cap1 + 不降权
  full1/decay0.8  = fullpath cap1 + Intro/Conclusion 分×0.8
  full1/decay0.7  = fullpath cap1 + ×0.7
  full1/decay0.5  = fullpath cap1 + ×0.5
输出 R@8/12/16 + MRR@16（all/hard）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks, top_section  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

KS = (8, 12, 16)
MRR_K = 16
_OV = ("introduction", "conclusion", "abstract", "related work", "background")
CONFIGS = ["baseline", "full1/d1.0", "full1/d0.8", "full1/d0.7", "full1/d0.5"]


def is_overview(path) -> bool:
    if not path:
        return False
    s = str(path[0]).lower()
    return any(k in s for k in _OV)


def rrf_full(vs, bs):
    n = len(vs)
    rrf = np.zeros(n)
    for r, i in enumerate(np.argsort(-vs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    for r, i in enumerate(np.argsort(-bs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    return rrf


def pick(order_rrf, chunks, top_k, cap, decay):
    if decay < 1.0:
        w = np.array([is_overview(getattr(c, "title_path", None)) for c in chunks])
        order_rrf = order_rrf * np.where(w, decay, 1.0)
    order = list(np.argsort(-order_rrf))
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
    stat = {c: {g: {"denom": 0, "R": {k: 0 for k in KS}, "MRR": 0.0}
                for g in ("normal", "hard", "all")} for c in CONFIGS}
    plan = {"baseline": (0, 1.0), "full1/d1.0": (1, 1.0), "full1/d0.8": (1, 0.8),
            "full1/d0.7": (1, 0.7), "full1/d0.5": (1, 0.5)}
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
            rrf = rrf_full(vs, bs)
            for cfg in CONFIGS:
                cap, decay = plan[cfg]
                picked = pick(rrf.copy(), chunks, max(KS), cap, decay)
                pos = {i: r for r, i in enumerate(picked)}
                first = min((pos[i] for i in cg if i in pos), default=None)
                for g in (grp, "all"):
                    d = stat[cfg][g]
                    if first is None:
                        continue
                    d["denom"] += 1
                    for k in KS:
                        if first < k:
                            d["R"][k] += 1
                    if first < MRR_K:
                        d["MRR"] += 1.0 / (first + 1)
    L = ["# fullpath cap1 × M2 概述降权 联合扫描（先理顺再惩罚）", "",
         "| config | R@8 | R@12 | R@16 | MRR@16 |"]
    L.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        s = stat[cfg]["all"]; d = max(s["denom"], 1)
        L.append(f"| {cfg} | {s['R'][8]/d:.3f} | {s['R'][12]/d:.3f} | "
                 f"{s['R'][16]/d:.3f} | {s['MRR']/d:.3f} |")
    L.append("")
    for cfg in CONFIGS:
        s = stat[cfg]["hard"]; d = max(s["denom"], 1)
        L.append(f"hard {cfg}: R@8={s['R'][8]/d:.3f} MRR={s['MRR']/d:.3f}")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/COMBO_SCAN_20260909.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
