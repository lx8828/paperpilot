"""临时：单篇检索的节级优化扫描（Recall@k/MRR，零 LLM，复用 recall_set）。

两个方法，全部作用在 hybrid 全序之后：
  M1 节级配额（per-section cap）：按 RRF 全序遍历，每 top_section 已取满 cap 就跳过，
     直到集满 top_k 或遍历完全文（保序多样化，避免 Introduction×4 挤占）。
  M2 概述节降权（overview decay）：Introduction/Conclusion/Abstract 类节在 RRF 分上乘
     decay（<1 降权）。降权可让这些概述块排后，给 Methods/Experiments 让位。

输出每 (cap, decay) 的 R@8/12/16 + MRR@16 + 命中块覆盖的不同 section 数（多样度）。
cap=0 或 decay=1.0 表示关闭对应方法（cap 参数仅对 M1；decay 仅对 M2，也可两者组合）。
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
# 概述节判定（小写模糊）：这些节对"细节题"帮助有限、却因泛泛表述总排前
_OVERVIEW_RE = ("introduction", "conclusion", "abstract", "related work", "background")

# (cap, decay)：cap=每节配额 0=关 M1；decay=概述降权 1.0=关 M2
CONFIGS = [(0, 1.0), (2, 1.0), (1, 1.0), (3, 1.0),
           (0, 0.7), (0, 0.8), (0, 0.9),
           (2, 0.8), (3, 0.8), (2, 0.7)]


def is_overview(sec: str) -> bool:
    s = sec.lower()
    return any(k in s for k in _OVERVIEW_RE)


def select(order: np.ndarray, secs: list[str], top_k: int,
           cap: int, decay: float, rrf: np.ndarray | None) -> list[int]:
    """按 RRF 全序保序选 top_k：cap>0 时每节配额；decay<1 时先给概述节降权重排。"""
    if decay < 1.0 and rrf is not None:
        w = np.where(np.array([is_overview(s) for s in secs], dtype=bool), decay, 1.0)
        order = np.argsort(-(rrf * w), kind="stable")
    out: list[int] = []
    used: dict[str, int] = {}
    for idx in order:
        sec = secs[int(idx)]
        if cap > 0 and used.get(sec, 0) >= cap:
            continue
        out.append(int(idx))
        used[sec] = used.get(sec, 0) + 1
        if len(out) >= top_k:
            break
    return out


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    items = data["items"]
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in items:
        by_pid.setdefault(it["pid"], []).append(it)

    # per-config stats
    stat = {}
    for (cap, decay) in CONFIGS:
        key = f"cap={cap},decay={decay}"
        stat[key] = {g: {"denom": 0, "R": {k: 0 for k in KS}, "MRR": 0.0, "nsec": []}
                     for g in ("normal", "hard", "all")}
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
        vecs = ChunkIndex(pdf).vectors()
        bm = BM25Index([c.text for c in chunks])
        ntexts = [ree.norm(c.text) for c in chunks]
        secs = [top_section(c) for c in chunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _diag = ree.locate_gold(ntexts, ev)
            if not cg:
                continue
            grp = it["group"]
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            order_v = np.argsort(-vs, kind="stable")
            order_b = np.argsort(-bs, kind="stable")
            rv = np.empty(n); rb = np.empty(n)
            rv[order_v] = np.arange(n); rb[order_b] = np.arange(n)
            rrf = 0.5 / (60 + rv + 1) + 0.5 / (60 + rb + 1)
            order = np.argsort(-rrf, kind="stable")
            for (cap, decay) in CONFIGS:
                s = stat[f"cap={cap},decay={decay}"]
                picked = select(order, secs, max(KS), cap, decay, rrf)
                pos = {idx: r for r, idx in enumerate(picked)}
                first = min((pos[i] for i in cg if i in pos), default=None)
                for g in (grp, "all"):
                    d = s[g]
                    if first is None:
                        continue
                    d["denom"] += 1
                    for k in KS:
                        if first < k:
                            d["R"][k] += 1
                    if first < MRR_K:
                        d["MRR"] += 1.0 / (first + 1)
                    # 多样度：按节去重后的命中块数（仅 all 记录一次）
                    if g == "all":
                        d["nsec"].append(len({secs[i] for i in picked}))

    # 报告
    L = ["# 节级配额 × 概述降权 扫描（Recall@k / MRR，样本 v1 250 题）", ""]
    L.append("> 基线 cap=0,decay=1.0 = 现状（hybrid RRF 等权，无节处理）")
    L.append("> M1 cap: 每 top_section 最多取 cap 块（保序）；M2 decay: Overview 节 RRF 分乘 decay")
    L.append("")
    for g in ("all", "normal", "hard"):
        L.append(f"### group = {g}")
        L.append(f"| config | R@8 | R@12 | R@16 | MRR@16 | 多样度(平均不同节) |")
        L.append("|---|---|---|---|---|---|")
        for (cap, decay) in CONFIGS:
            key = f"cap={cap},decay={decay}"
            s = stat[key][g]
            d = max(s["denom"], 1)
            tag = " ← 现状" if (cap, decay) == (0, 1.0) else ""
            ns = ""
            if g == "all" and s["nsec"]:
                ns = f"{sum(s['nsec'])/len(s['nsec']):.1f}"
            L.append(f"| {key} | {s['R'][8]/d:.3f} | {s['R'][12]/d:.3f} | "
                     f"{s['R'][16]/d:.3f} | {s['MRR']/d:.3f} | {ns} |{tag}")
        L.append("")
    Path("qa/recall/SECTION_SCAN_20260909.md").write_text("\n".join(L), encoding="utf-8")
    print("已写 qa/recall/SECTION_SCAN_20260909.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
