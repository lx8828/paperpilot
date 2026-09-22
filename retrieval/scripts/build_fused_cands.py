"""生成「引用信号融合」后的 top-20 候选，供 `llm_rerank.py --cands-file` 使用。

融合规则（β=0.2 由 5 折 CV 选出，五折一致）：

    pool         = ce top-20 ∪ (ce top-20 的 outgoing references ∩ 语料)   平均 132.6 篇
    ce_score     = 已缓存的 cross-encoder 分数（`ce_onehop_k20.npz`）
    cit_score(d) = Σ_{s ∈ ce top-20, d ∈ cites(s)} 1 / rank(s)
    fused        = z(ce_score) + 0.2 · z(cit_score)
    → 取 fused 的前 20

产出：`results/fused_cands_k20.npy`，形状 (597, 20) 的行号数组。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
DATA = HERE / "data" / "litsearch"
QFILE = DATA / "query" / "full-00000-of-00001.parquet"
TOPK = RESULTS / "topk1000.npz"
CE_SCORES = RESULTS / "rerank1000.npy"

POOL = 100
SEED_K = 20
BETA = 0.2
TOP = 20


def z(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def main() -> int:
    q = pd.read_parquet(QFILE)
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    n = len(golds)

    cit: dict[int, np.ndarray] = {}
    for p in sorted((DATA / "corpus_clean").glob("*.parquet")):
        d = pd.read_parquet(p, columns=["corpusid", "citations"])
        for cid, cs in zip(d["corpusid"].to_numpy(), d["citations"].to_numpy()):
            cit[int(cid)] = cs if cs is not None else np.array([], dtype=np.int64)

    pool = np.load(TOPK)["hyb0.5"][:, :POOL]
    ce0 = np.load(CE_SCORES)[:, :POOL]
    ce_order = [[int(x) for x in pool[i][np.argsort(-ce0[i])]] for i in range(n)]

    zz = np.load(RESULTS / f"ce_onehop_k{SEED_K}.npz", allow_pickle=True)
    rows_arr, sc_arr = zz["rows"], zz["scores"]

    out = np.zeros((n, TOP), dtype=np.int32)
    for i in range(n):
        docs = [int(x) for x in rows_arr[i]]
        pos = {d: j for j, d in enumerate(docs)}
        cs = np.zeros(len(docs), dtype=np.float64)
        for rank, s in enumerate(ce_order[i][:SEED_K], 1):
            for c in cit.get(int(ids[s]), np.array([], dtype=np.int64)):
                row = id2row.get(int(c))
                if row is not None and row in pos:
                    cs[pos[row]] += 1.0 / rank
        f = z(sc_arr[i].astype(np.float64)) + BETA * z(cs)
        order = [docs[j] for j in np.argsort(-f)][:TOP]
        out[i, :len(order)] = order

    # 自检：与基线对比
    def gold_atk(ranked_lists, kk):
        return float(np.mean([len(golds[i] & set(ranked_lists[i][:kk])) / len(golds[i])
                              if golds[i] else 0.0 for i in range(n)]))

    base = [[int(x) for x in ce_order[i][:TOP]] for i in range(n)]
    fused = [[int(x) for x in out[i]] for i in range(n)]
    print(f"β={BETA} 融合 top-{TOP} 已生成，(n={n})\n")
    print(f"  {'':<18}{'gold@1':>9}{'gold@5':>9}{'gold@10':>9}{'gold@20':>9}")
    for name, rl in (("ce top-20（基线）", base), ("引用信号融合", fused)):
        print(f"  {name:<18}" + "".join(f"{gold_atk(rl, k):>9.4f}" for k in (1, 5, 10, 20)))

    np.save(RESULTS / "fused_cands_k20.npy", out)
    print(f"\n已写出：results/fused_cands_k20.npy  {out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
