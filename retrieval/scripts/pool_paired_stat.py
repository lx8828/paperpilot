"""池子深度对比的**配对显著性**（McNemar + 配对 bootstrap）。

为什么能配对：`bge-reranker-v2-m3` 是 pointwise —— 同一批查询、同一批 CE 分数，
宽池的排序限制到窄池子集后**逐位等价**。所以两个池深是**完全配对**的，
用 McNemar 而不是两组独立比较。

用法：
    python retrieval/scripts/pool_paired_stat.py --csv LITSEARCH_POOL200_raw.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from significance import mcnemar  # noqa: E402

RESULTS = HERE / "results"


def ci(v: np.ndarray, rng, B: int = 10_000) -> tuple[float, float]:
    n = len(v)
    m = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LITSEARCH_POOL200_raw.csv")
    args = ap.parse_args()

    d = pd.read_csv(RESULTS / args.csv, encoding="utf-8-sig")
    cols = [c for c in d.columns if re.fullmatch(r"(ce|llm)\d+", c)]
    pools = sorted({int(c[2:]) if c.startswith("ce") else int(c[3:]) for c in cols})
    cbase, cmain = pools[0], pools[-1]
    print(f"读入 {args.csv} | 池子 {pools}（对照 {cbase} → 主臂 {cmain}）| {len(d)} 条\n")

    uni = d[d["batch"] == "uniform"]
    rng = np.random.default_rng(7)
    groups = [("总体·均匀", uni), ("旧语料", d[d["语料"] == "旧"]),
              ("新语料", d[d["语料"] == "新"]),
              ("具体型", d[d["抽象层级"] == "具体"]),
              ("上位型", d[d["抽象层级"] == "上位"])]

    print("=" * 112)
    print(f"配对显著性：C={cbase} → C={cmain}")
    print("=" * 112)
    print(f"{'分层':<12}{'指标':<10}{'@' + str(cbase):>9}{'@' + str(cmain):>9}{'Δ':>9}"
          f"{'95%CI':>20}{'翻转 主胜:对照胜':>18}{'p':>11}")
    print("-" * 112)
    for name, sub in groups:
        if sub.empty:
            continue
        for tag in ("ce", "llm"):
            if f"{tag}{cbase}" not in sub.columns:
                continue
            for k in (1, 5):
                a = ((sub[f"{tag}{cbase}"] > 0) & (sub[f"{tag}{cbase}"] <= k)).to_numpy(float)
                b = ((sub[f"{tag}{cmain}"] > 0) & (sub[f"{tag}{cmain}"] <= k)).to_numpy(float)
                dl = b - a
                w1 = int(((b == 1) & (a == 0)).sum())
                w0 = int(((b == 0) & (a == 1)).sum())
                lo, hi = ci(dl, rng)
                mark = "**" if mcnemar(w1, w0) < 0.05 else "  "
                print(f"{name:<12}{tag.upper() + ' R@' + str(k):<10}"
                      f"{a.mean():9.4f}{b.mean():9.4f}{100 * dl.mean():+8.2f}pt"
                      f"   [{100 * lo:+.2f}, {100 * hi:+.2f}]{f'{w1}:{w0}':>16}"
                      f"{mcnemar(w1, w0):12.4g} {mark}")

    # ── 兑现率：R@1 / 天花板 ────────────────────────────────
    print("\n" + "=" * 112)
    print("兑现率 = 落地 R@1 ÷ 天花板（该系数若对池深不变 → 可按天花板外推）")
    print("=" * 112)
    print(f"{'分层':<12}{'天花板@' + str(cbase):>14}{'天花板@' + str(cmain):>14}"
          f"{'兑现@' + str(cbase):>13}{'兑现@' + str(cmain):>13}")
    for name, sub in groups:
        if sub.empty:
            continue
        ceil_b = (sub["g_rank_retr"] <= cbase).mean()
        ceil_m = (sub["g_rank_retr"] <= cmain).mean()
        r_b = ((sub[f"llm{cbase}"] > 0) & (sub[f"llm{cbase}"] <= 1)).mean()
        r_m = ((sub[f"llm{cmain}"] > 0) & (sub[f"llm{cmain}"] <= 1)).mean()
        print(f"{name:<12}{ceil_b:14.4f}{ceil_m:14.4f}"
              f"{r_b / max(ceil_b, 1e-9):13.1%}{r_m / max(ceil_m, 1e-9):13.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
