"""临时：列出 NDCG 最低 / gold 排位最差的题原文，人工判断是不是"题目宽泛导致难检索"。"""
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


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    rows = []
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
                    best = int(np.argmax(np.asarray(bm.score(ree.norm(ev)), dtype="float64")))
                    cg = {best}
                except Exception:
                    continue
            if not cg:
                continue
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            rrf = np.zeros(n)
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order = list(np.argsort(-rrf))
            first = next((p for p in range(len(order)) if order[p] in cg), None)
            rows.append({"pid": pid[:8], "qid": it["qid"][:10], "grp": it["group"],
                         "rank": first, "q": it["question"],
                         "n_chunk": n, "gold_sec": chunks[first].title_path if first is not None else None})
    rows.sort(key=lambda r: (r["rank"] is None, -(r["rank"] if r["rank"] is not None else -1)))
    print("=== gold 排位最差 / miss 的 20 题（rank 越大越差；None=miss12）===")
    for r in rows[:20]:
        rk = "MISS" if r["rank"] is None else f"rank#{r['rank']+1}"
        nd = "" if r["rank"] is None else f"NDCG={1/math.log2(r['rank']+2):.2f}"
        print(f"[{r['grp']:6}][{rk:6}] {r['qid']} n_chunk={r['n_chunk']} {nd}")
        print(f"    Q: {r['q'][:110]}")
        if r["gold_sec"] is not None:
            print(f"    gold节: {r['gold_sec'][0] if isinstance(r['gold_sec'], list) else r['gold_sec']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
