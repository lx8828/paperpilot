"""临时：K 值扫描 + 尾部价值诊断（新 gold 口径=全部 evidence 段）。

回答三问：
  1) 现在 K 是多少、该不该改 → 逐 k 看 R@k / NDCG@k 曲线何时饱和
  2) 尾巴装满吗 → 每个序号槽 k 的"边际价值"：有多少题的**首个 gold** 恰好落在第 k 位；
     以及 top-k 里平均含几个 gold 块（多证据口径下才有意义）
  3) 降召回会提精准吗 → P@k = |top_k ∩ C_gold| / k 随 k 的变化（真实精度，非定义假象）
另给出：每篇 chunk 数分布（K 是否大到"整篇返回"）。
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
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

KMAX = 20
KS = [4, 6, 8, 10, 12, 14, 16, 20]


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    nchunks: list[int] = []
    margin = [0] * (KMAX + 2)      # 首个 gold 恰在第 rank+1 位的题数（rank 从 0）
    hit = {k: 0 for k in KS}
    ndcg = {k: 0.0 for k in KS}
    prec = {k: 0.0 for k in KS}
    ngold_topk = {k: 0.0 for k in KS}
    nq = 0
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
        nchunks.append(n)
        vecs = ChunkIndex(pdf).vectors()
        bm = BM25Index([c.text for c in chunks])
        ntexts = [ree.norm(c.text) for c in chunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, evs = gold_answer_full(q)
            if not evs:
                continue
            cg: set[int] = set()
            for e in evs:
                got, _ = ree.locate_gold(ntexts, e)
                if not got:
                    try:
                        got = {int(np.argmax(np.asarray(bm.score(ree.norm(e)), dtype="float64")))}
                    except Exception:
                        got = set()
                cg |= got
            if not cg:
                continue
            nq += 1
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            rrf = np.zeros(n)
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order = list(np.argsort(-rrf))
            pos = [p for p in range(n) if order[p] in cg]
            first = pos[0] if pos else None
            if first is not None and first <= KMAX:
                margin[first] += 1
            for k in KS:
                topk = order[:k]
                g_in = sum(1 for i in topk if i in cg)
                ngold_topk[k] += g_in
                prec[k] += g_in / k
                if g_in:
                    hit[k] += 1
                dcg = sum(1.0 / math.log2(p + 2) for p in pos if p < k)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(cg), k)))
                ndcg[k] += (dcg / idcg) if idcg > 0 else 0.0

    ncs = sorted(nchunks)
    L = ["# K 值扫描 + 尾部价值（新口径：全部 evidence 段并集）", ""]
    L.append(f"> 题 {nq} | 论文 {len(nchunks)}")
    L.append(f"> 每篇 chunk 数：中位 {np.median(ncs):.0f} | 均值 {np.mean(ncs):.1f} | "
             f"min {ncs[0]} | max {ncs[-1]}")
    L.append(f"> 每篇块数 ≤8 的论文 {sum(1 for x in ncs if x<=8)} 篇 | ≤12 "
             f"{sum(1 for x in ncs if x<=12)} 篇 | ≤16 {sum(1 for x in ncs if x<=16)} 篇"
             f"（K≥该值时=整篇全返回）")
    L.append("")
    L.append("| K | R@K | NDCG@K | P@K | top-k 平均gold块数 | 首个gold恰在此槽的题数 |")
    L.append("|---|---|---|---|---|---|")
    for k in KS:
        L.append(f"| {k} | {hit[k]/nq:.3f} | {ndcg[k]/nq:.3f} | {prec[k]/nq:.3f} | "
                 f"{ngold_topk[k]/nq:.2f} | {margin[k-1]} |")
    L.append("")
    L.append("### 逐槽边际（rank r 处首个 gold 落点，r=1..20）")
    L.append("| 槽位 | 题数 |")
    L.append("|---|---|")
    for r in range(KMAX):
        L.append(f"| #{r+1} | {margin[r]} |")
    L.append(f"| >#{KMAX} | {nq - sum(margin[:KMAX])} |")
    txt = "\n".join(L)
    Path("qa/recall/K_SCAN_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
