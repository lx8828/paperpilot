"""修正表①的「分离度」：改为**逐 facet 标准化**再平均。

⚠️ 原先用 `(gold均分 - 干扰均分) / 全体std` —— 那是**跨 facet 混池**统计，
会被各 facet 的分数尺度差异污染（例如 rerank 的 logit 在不同 facet 上基线不同）。
逐 facet 算 Cohen's d 再取平均才是干净的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = pd.read_csv(HERE / "results" / "R2_NLI_CALIB_DOC.csv")
print("=" * 104)
print("【表① 修正】逐 facet 标准化分离度（Cohen's d）｜ 越大越好")
print(f"  {'验证器':<12}{'AUC':>8}{'逐facet d':>12}{'d 中位':>9}{'负 d 的 facet':>14}"
      f"{'d>0.5 占比':>13}")
rows = []
for a, t in d.groupby("arm"):
    ds = []
    for _, x in t.groupby(["cluster", "facet"]):
        g = x[x.gold == 1]["score"].to_numpy()
        n = x[x.gold == 0]["score"].to_numpy()
        if not len(g) or not len(n):
            continue
        sd = np.sqrt((g.var(ddof=1) * len(g) + n.var(ddof=1) * len(n)) / (len(g) + len(n)))
        ds.append((g.mean() - n.mean()) / sd if sd else 0.0)
    ds = np.array(ds)
    # AUC（逐 facet 再平均）
    aus = []
    for _, x in t.groupby(["cluster", "facet"]):
        y = x.gold.to_numpy().astype(bool)
        s = x.score.to_numpy()
        n1, n0 = int(y.sum()), int((~y).sum())
        if n1 and n0:
            r = pd.Series(s).rank().to_numpy()
            aus.append((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    rows.append((a, float(np.mean(aus)), float(ds.mean()), float(np.median(ds)),
                 int((ds < 0).sum()), float((ds > 0.5).mean()), len(ds)))
    print(f"  {a:<12}{np.mean(aus):>8.3f}{ds.mean():>12.3f}{np.median(ds):>9.3f}"
          f"{int((ds < 0).sum()):>10}/{len(ds):<4}{float((ds > 0.5).mean()):>12.1%}")

print("\n  ★ 若「逐facet d」显著为负 → 该验证器在本语料上是**系统性反向**；")
print("    若仅接近 0 → 是**接近随机**，不是反向（别混为一谈）。")
pd.DataFrame(rows, columns=["arm", "auc", "d_mean", "d_med", "n_neg", "frac_gt05", "n"]
             ).to_csv(HERE / "results" / "R2_NLI_CALIB_T1_FIXED.csv",
                      index=False, encoding="utf-8-sig")
