"""验证 CE/LLM 精排产物对齐的是哪个池子（`topk.npz` vs `topk1000.npz`）。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "retrieval" / "results"
DER = ROOT / "retrieval" / "data" / "litsearch" / "derived"

ids = np.load(DER / "emb" / "corpusid.npy")
id2row = {int(c): i for i, c in enumerate(ids)}
q = pd.read_parquet(glob.glob(str(ROOT / "retrieval/data/litsearch/query/*.parquet"))[0])
G = [{id2row[int(x)] for x in np.atleast_1d(r) if int(x) in id2row} for r in q["corpusids"]]


def rr(ranked: list[np.ndarray], kk: int) -> float:
    return float(np.mean([len(g & set(r[:kk].tolist())) / len(g) if g else 0
                          for r, g in zip(ranked, G)]))


t = np.load(RES / "topk.npz")
t1 = np.load(RES / "topk1000.npz")
same = bool((t["hyb0.5"] == t1["hyb0.5"][:, :100]).all())
print(f"topk.npz['hyb0.5'] == topk1000['hyb0.5'][:, :100] ? {same}")

cache = json.loads((DER / "llm_rerank_cache.json").read_text(encoding="utf-8"))

# ⚠️ `llm_rerank.py` 的官方口径：TOPK=topk1000.npz, CE_SCORES=rerank1000.npy, POOL_DEPTH=100
combos = (("topk.npz + rerank_scores", t["hyb0.5"], np.load(RES / "rerank_scores.npy")),
          ("topk1000 + rerank_scores", t1["hyb0.5"][:, :100], np.load(RES / "rerank_scores.npy")),
          ("topk.npz + rerank1000", t["hyb0.5"], np.load(RES / "rerank1000.npy")[:, :100]),
          ("**topk1000 + rerank1000**（官方）", t1["hyb0.5"][:, :100],
           np.load(RES / "rerank1000.npy")[:, :100]))
for name, pool, ce in combos:
    ceo = np.take_along_axis(pool, np.argsort(-ce, axis=1), axis=1)
    line = (f"{name:<10} hyb R@1={rr([r for r in pool], 1):.4f} R@5={rr([r for r in pool], 5):.4f}"
            f" | CE R@1={rr([r for r in ceo], 1):.4f} R@5={rr([r for r in ceo], 5):.4f}")
    for K in (20, 50):
        out = []
        for i in range(len(G)):
            rec = cache.get(f"k{K}:full:B:{i}")
            cand = np.asarray(ceo[i, :K])
            if not rec:
                out.append(cand)
                continue
            disp = np.asarray(rec["disp"])
            perm = np.asarray(rec["perm"]) - 1
            out.append(cand[disp[perm]])
        line += f" | LLM{K} R@1={rr(out, 1):.4f} R@5={rr(out, 5):.4f}"
    print(line)

print("目标（LITSEARCH_K_DECISION）：hyb R@1 0.3548 ｜ CE R@1 0.4154 ｜ "
      "LLM K20 R@1 0.6092 / R@5 0.7195 ｜ LLM K50 R@1 0.6092 / R@5 0.7638")
