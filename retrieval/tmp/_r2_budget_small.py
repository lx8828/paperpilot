r"""**低预算方案**：¥1 / ¥0.10 / ¥0.01 每题的对比（复用 `_r2_budget_model.py` 的模型）。

两个**不依赖模型**的数先摆前面：

1. **可判篇数** `m = floor((X − 2c)/c)`
2. **召回物理上界**（无假设）：`R_hard = mean_facet min(1, k / n_gold_facet)`
   —— 只判 `k` 篇，gold 有 `n_gold` 篇 → 最多只能找到 `k` 篇 → **R 不可能超过 `k/n_gold`**。
   这一条**与排序质量、判官质量都无关**，是最硬的约束。

跑法：`./.venv/Scripts/python.exe -u retrieval/tmp/_r2_budget_small.py`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tmp"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location("_BM", HERE / "tmp" / "_r2_budget_model.py")
BM = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(BM)                         # type: ignore[union-attr]

C = 0.002
XS = [1.0, 0.10, 0.01]
N_GOLD = 13.0                                        # 实测均值（范围 2~33）


def main() -> int:
    F = BM.load_facets()
    P = BM.P_JUDGE
    d = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv")
    d["in_gold"] = d["in_gold"].astype(int)
    d["label"] = d["label"].fillna("").astype(str).str.strip().str.lower()
    ng = d.groupby(["cluster", "facet"])["in_gold"].sum().to_numpy(float)

    # ρ(k/N) 锚点
    fx, fy = [], []
    for kk in (5, 10, 15, 20, 25, 30, 40, 50):
        ce = BM.ceil_sim(F, 50, kk, 1)
        rr = float(np.mean([
            len({p for p in x[(x["rank"] <= kk) & (x["label"] == "yes")]["docid"]}
                & set(x[x.in_gold == 1]["docid"])) / max(len(set(x[x.in_gold == 1]["docid"])), 1)
            for _, x in d.groupby(["cluster", "facet"])]))
        fx.append(kk / 50)
        fy.append(rr / ce)
    rho = lambda f: float(np.interp(f, fx, fy))       # noqa: E731

    def f1(rr: float) -> float:
        h = 2 * P * rr / (P + rr) if (P + rr) else 0.0
        return h - 0.0221

    print("=" * 124)
    print(f"【低预算对比】单价 ¥{C}/次 ｜ P_judge {P} ｜ gold 均 {N_GOLD:.0f} 篇")
    print(f"\n  {'X(元/题)':>10}{'固定开销':>10}{'占比':>8}{'可判 m':>9}"
          f"{'R 物理上界':>12}{'F1 上界':>10}{'判官秒':>8}")
    INFO = {}
    for X in XS:
        m = max(0, int((X - 2 * C) // C))
        ov = 2 * C / X
        Rh = float(np.mean(np.minimum(1.0, m / ng)))      # ★ 无假设上界
        INFO[X] = (m, Rh)
        print(f"  {X:>10.2f}{2 * C:>10.4f}{ov:>8.1%}{m:>9,}"
              f"{Rh:>12.4f}{f1(Rh):>10.4f}{m * 0.8 / 8:>8.1f}")

    print("\n" + "=" * 124)
    print("【① 拐点】语料 N ≤ m 时全判 → 指标 = 该预算下的天花板；N 一超过就开始掉")
    cap = f"天花板 setF1（m≥{int(N_GOLD)} 时 = 实测 0.7173）"
    print(f"  {'X(元)':>8}{'m':>7}{cap:>34}")
    for X in XS:
        m, _ = INFO[X]
        ce = BM.ceil_sim(F, max(m, 1), max(m, 1), 60)
        R = ce * rho(1.0)
        print(f"  {X:>8.2f}{m:>7,}{f1(R):>34.4f}")

    print("\n" + "=" * 124)
    print("【② ★ 衰减曲线（各预算 × 语料规模）】`ceil` 由 k/N 决定，`ρ` 实测插值")
    print(f"  {'X(元)':>7}{'m':>6}{'N':>8}{'k/N':>8}{'ceil@k':>9}{'ρ':>7}{'R':>7}"
          f"{'setF1':>8}{'R 上界':>9}")
    for X in XS:
        m, _ = INFO[X]
        for mult in (1, 2, 5, 10, 50):
            N = max(m, 1) * mult
            k = min(m, N)
            ce = BM.ceil_sim(F, N, k, 120)
            fr = k / N
            R = ce * rho(fr)
            print(f"  {X:>7.2f}{m:>6,}{N:>8,}{fr:>8.1%}{ce:>9.4f}{rho(fr):>7.3f}"
                  f"{R:>7.3f}{f1(R):>8.4f}{float(np.mean(np.minimum(1.0, k / ng))):>9.3f}")
        print()

    print("=" * 124)
    print("【③ ★ 检索的边际价值】`ceil` 每 +0.10 换来的 setF1（= 检索值不值钱）")
    print(f"  {'X(元)':>7}{'m':>6}{'N=2m':>8}{'ceil':>8}{'setF1':>8}{'ceil+0.1 后':>12}"
          f"{'ΔsetF1':>9}")
    for X in XS:
        m, _ = INFO[X]
        N = 2 * m
        k = min(m, N)
        ce = BM.ceil_sim(F, N, k, 120)
        r0 = rho(k / N)
        print(f"  {X:>7.2f}{m:>6,}{N:>8,}{ce:>8.3f}{f1(ce * r0):>8.4f}"
              f"{f1(min(1.0, ce + 0.10) * r0):>12.4f}"
              f"{f1(min(1.0, ce + 0.10) * r0) - f1(ce * r0):>+9.4f}")

    print("\n" + "=" * 124)
    print("【结论】")
    for X in XS:
        m, Rh = INFO[X]
        print(f"  · **¥{X:.2f}/题**：可判 **{m:,} 篇** ｜ 召回物理上界 **R ≤ {Rh:.3f}**"
              f" → **F1 ≤ {f1(Rh):.3f}**"
              + ("　← **不可行**（gold 就 "
                 f"{int(N_GOLD)} 篇，3 篇装不下）" if m < N_GOLD else ""))
    print("\n  ★ 三条硬话：")
    print("    1. **¥0.10/题 ≈ 今天的规模**：可判 48 篇 ≳ 我们现在的 50 篇 → "
          "**今天这套（50 篇）就是「一毛钱方案」**")
    print(f"    2. **¥0.01/题 在当前单价下不可行**：只能判 3 篇，而 gold 平均 "
          f"{int(N_GOLD)} 篇 → R ≤ {INFO[0.01][1]:.2f}，**与排序/判官优劣无关**")
    print(f"    3. 要让 ¥0.01 可行（判 ~50 篇），单价需降到 "
          f"**¥{0.01 / 50:.5f}/次**（今天的 1/{C / (0.01 / 50):.0f}）；"
          f"要判 497 篇则需 **¥{0.01 / 497:.6f}/次**（1/{C / (0.01 / 497):.0f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
