"""临时：cap 按【完整 title_path】去重（而非顶层节 top_section）——修正版扫描。

背景：cap=1 按 top_section 误伤大节子标题块（27 子标题被当 1 节），端到端 Δ-3。
真正该去重的是"二级切分共享完整 title_path"的块（Introduction p1/p2 那种）。
本脚本对比两种粒度：
  baseline   = 无 cap（现状 top12）
  toppath1   = cap=1 按 top_section（旧，误伤版）
  fullpath1  = cap=1 按完整 title_path（修正版）
  fullpath2  = cap=2 按完整 title_path
输出 Recall@8/12/16 + MRR@16。
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
CONFIGS = ["baseline", "toppath1", "fullpath1", "fullpath2"]


def select_full(order: list[int], chunks: list, top_k: int, cap: int) -> list[int]:
    """按完整 title_path（tuple）保序 cap。"""
    out, used = [], {}
    for idx in order:
        key = tuple(getattr(chunks[idx], "title_path", None) or [])
        if used.get(key, 0) >= cap:
            continue
        out.append(idx); used[key] = used.get(key, 0) + 1
        if len(out) >= top_k:
            break
    return out


def select_top(order: list[int], chunks: list, top_k: int, cap: int) -> list[int]:
    key = lambda c: top_section(c)  # noqa: E731
    out, used = [], {}
    for idx in order:
        k = key(chunks[idx])
        if used.get(k, 0) >= cap:
            continue
        out.append(idx); used[k] = used.get(k, 0) + 1
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
            order = list(rrf_order_full(vs, bs))
            picked = {
                "baseline": order[: max(KS)],
                "toppath1": select_top(order, chunks, max(KS), 1),
                "fullpath1": select_full(order, chunks, max(KS), 1),
                "fullpath2": select_full(order, chunks, max(KS), 2),
            }
            for cfg in CONFIGS:
                pos = {i: r for r, i in enumerate(picked[cfg])}
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
    L = ["# cap 粒度修正扫描：完整 title_path vs 顶层节（样本 v1 250 题）", "",
         "| config | R@8 | R@12 | R@16 | MRR@16 |"]
    L.append("|---|---|---|---|---|")
    for cfg in CONFIGS:
        s = stat[cfg]["all"]; d = max(s["denom"], 1)
        L.append(f"| {cfg} | {s['R'][8]/d:.3f} | {s['R'][12]/d:.3f} | "
                 f"{s['R'][16]/d:.3f} | {s['MRR']/d:.3f} |")
    L.append("")
    for cfg in CONFIGS:
        s = stat[cfg]["hard"]; d = max(s["denom"], 1)
        L.append(f"hard {cfg}: R@8={s['R'][8]/d:.3f} R@12={s['R'][12]/d:.3f} "
                 f"MRR={s['MRR']/d:.3f}")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/FULLPATH_SCAN_20260909.md").write_text(txt + "\n", encoding="utf-8")
    return 0


def rrf_order_full(vs, bs):
    n = len(vs)
    rrf = np.zeros(n)
    for r, i in enumerate(np.argsort(-vs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    for r, i in enumerate(np.argsort(-bs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    return np.argsort(-rrf)


if __name__ == "__main__":
    raise SystemExit(main())
