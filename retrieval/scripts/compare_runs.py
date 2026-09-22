"""对两个 `*_multi_corpus_raw.csv`（或任意含 q/R@k 的逐题结果）做**配对**比较。

为什么必须配对：两组跑的是**同一批题**，"谁在哪题赢"是强相关的配对信息，
用双样本检验会严重低估/高估。这里用
  · McNemar 精确检验（二元：该题 R@k 是否 > 0，即"命中了没有"）
  · 配对 bootstrap（连续：逐题 R@k 的差，10k 次重采样给 95% CI）

用法：
    python retrieval/scripts/compare_runs.py --a results/LITSEARCH_CE100-0_multi_corpus_raw.csv \
                                             --b results/LITSEARCH_CE100-100_multi_corpus_raw.csv \
                                             --label-a "CE(旧100)" --label-b "CE(旧100+新100)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from significance import mcnemar  # noqa: E402

RESULTS = HERE / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metrics", default="R@1,R@5,R@10,R@100,MRR@10")
    args = ap.parse_args()

    da = pd.read_csv(RESULTS / args.a).set_index("q")
    db = pd.read_csv(RESULTS / args.b).set_index("q")
    common = da.index.intersection(db.index)
    da, db = da.loc[common], db.loc[common]
    print(f"配对题数 {len(common)}    A={args.label_a}  B={args.label_b}\n")
    print(f"{'指标':<9}{'A':>9}{'B':>9}{'Δpt':>9}{'95%CI':>22}{'B赢':>6}{'A赢':>6}{'McNemar p':>13}  结论")
    rng = np.random.default_rng(0)
    for m in [x.strip() for x in args.metrics.split(",") if x.strip()]:
        if m not in da.columns or m not in db.columns:
            continue
        va, vb = da[m].to_numpy(float), db[m].to_numpy(float)
        d = vb - va
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ha, hb = (va > 0).astype(int), (vb > 0).astype(int)
        o1, o0 = int(((hb == 1) & (ha == 0)).sum()), int(((hb == 0) & (ha == 1)).sum())
        pv = mcnemar(o1, o0)
        ci = f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"
        verdict = ("B 显著更好" if lo > 0 else "A 显著更好" if hi < 0 else "无显著差异")
        print(f"{m:<9}{va.mean():9.4f}{vb.mean():9.4f}{100 * d.mean():+9.2f}{ci:>22}"
              f"{o1:>6}{o0:>6}{pv:>13.3g}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
