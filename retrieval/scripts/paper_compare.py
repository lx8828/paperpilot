"""把本项目的各阶段结果换算成 LitSearch 论文口径（Avg-Broad R@20 / Avg-Specific R@5），与论文 Table 3 对标。

论文口径的由来（见 `LITSEARCH_SPECIFICITY.md`）：Broad 类最多 20 篇相关、Specific 类最多 5 篇，
所以论文按 Broad 报 R@20、Specific 报 R@5 —— 这是**公平性修正**，不是策略差异。

⚠️ 口径提醒：论文是在**全语料重排结果**上算的；我们的 LLM 只在 top-K 候选内排序，
  因此 K 较小时 R@20 会被**候选集上界封顶**（K=20 时 74.3%），我们的数字是**下界**。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
TOPK = RESULTS / "topk1000.npz"
CE_SCORES = RESULTS / "rerank1000.npy"
CACHE = DERIVED / "llm_rerank_cache.json"
POOL_DEPTH = 100

# LitSearch 论文 Table 3 的 Avg. 列（仅用 title+abstract 的检索配置）
PAPER = {
    "BM25": (39.9, 50.0),
    "GTR-T5-large": (43.8, 39.6),
    "E5-large-v2": (55.4, 56.2),
    "Instructor-XL": (56.5, 52.3),
    "GPT-4o rerank(BM25)": (59.9, 68.0),
    "GritLM-7B": (70.8, 74.8),
    "GPT-4o rerank(GritLM)": (75.3, 79.2),
}


def main() -> int:
    q = pd.read_parquet(QFILE)
    truth = q["specificity"].to_numpy()
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    n = len(golds)

    pool = np.load(TOPK)["hyb0.5"][:, :POOL_DEPTH]
    ce = np.load(CE_SCORES)[:, :POOL_DEPTH]
    ce_order = [pool[i][np.argsort(-ce[i])] for i in range(n)]
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    # 从缓存恢复各 K 的 LLM 排序
    llm_orders: dict[int, list[np.ndarray | None]] = {}
    for key in cache:
        if not key.startswith("k") or ":full:B:" not in key:
            continue
        k_str, _, _, i_str = key.split(":")
        k = int(k_str[1:])
        llm_orders.setdefault(k, [None] * n)
        c = cache[key]
        disp = np.array(c["disp"])
        perm = np.array(c["perm"]) - 1
        llm_orders[k][int(i_str)] = ce_order[int(i_str)][:k][disp[perm]]

    def R(rl, k, mask=None):
        idx = range(n) if mask is None else np.where(mask)[0]
        vals = []
        for i in idx:
            if rl[i] is None or not golds[i]:
                vals.append(0.0)
                continue
            vals.append(len(golds[i] & set(rl[i][:k].tolist())) / len(golds[i]))
        return float(np.mean(vals))

    z = np.load(TOPK)          # 各第一阶段的 top-1000
    rows = []
    for k in sorted(llm_orders):
        rows.append({"系统": f"ours: hyb0.5→CE→LLM listwise (K={k})",
                     "Broad R@20": 100 * R(llm_orders[k], 20, truth == 0),
                     "Specific R@5": 100 * R(llm_orders[k], 5, truth == 1)})
    rows.append({"系统": "ours: hyb0.5→CE（pointwise）",
                 "Broad R@20": 100 * R(ce_order, 20, truth == 0),
                 "Specific R@5": 100 * R(ce_order, 5, truth == 1)})
    for name, key in (("hyb0.5", "hyb0.5"), ("dense", "dense"), ("BM25", "bm25")):
        rl = [z[key][i] for i in range(n)]
        rows.append({"系统": f"ours: {name}（第1阶段）",
                     "Broad R@20": 100 * R(rl, 20, truth == 0),
                     "Specific R@5": 100 * R(rl, 5, truth == 1)})
    for name, (b, s) in PAPER.items():
        rows.append({"系统": f"paper: {name}", "Broad R@20": b, "Specific R@5": s})

    t = pd.DataFrame(rows).sort_values("Broad R@20", ascending=False)
    print("=" * 84)
    print("LitSearch 论文口径对标（Avg-Broad R@20 / Avg-Specific R@5，%）")
    print("=" * 84)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    t.to_csv(RESULTS / "LITSEARCH_PAPER_COMPARE.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写出：results/LITSEARCH_PAPER_COMPARE.csv")
    print("\n⚠️ 我们的数字是**下界**：LLM 只在 top-K 候选内排序，R@20 会被候选集封顶。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
