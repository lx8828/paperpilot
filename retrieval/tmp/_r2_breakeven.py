r"""**"检索什么时候才有意义"的拐点**（¥1/题，单价 ¥0.002/次 → 可判 497 篇）。

用户估计：*"¥1/题的成本基础上，要想检索有意义，大概 1500 篇左右，这时 P/R/F1 都不会降太多。"*

本脚本：
1. 细扫 N，给精确的 `ceil / R / setF1` 与**相对 N=50 基线的降幅**
2. 给"降幅 ≤5% / ≤10%"对应的 N 阈值
3. 给一个"检索有多重要"的量化：**未被判官覆盖的语料比例 = 1 − k/N**
4. 列出 `k/N = 100% / 50% / 33% / 25%` 四个位置

复用 `_r2_budget_model.py` 的模型（不自造第二套）。
跑法：`./.venv/Scripts/python.exe -u retrieval/tmp/_r2_breakeven.py`
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tmp"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location("_BM", HERE / "tmp" / "_r2_budget_model.py")
BM = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(BM)                         # type: ignore[union-attr]

C = 0.002
X = 1.0
M = max(0, int((X - 2 * C) // C))                    # 可判篇数
BASE_N = 50


def main() -> int:
    F = BM.load_facets()
    P = BM.P_JUDGE

    # ρ(k/N) 锚点（与模型一致，从 DELIV 实测反推）
    import pandas as pd
    d = pd.read_csv(HERE / "results" / "R2_PROD_FINAL_DELIV.csv")
    d["in_gold"] = d["in_gold"].astype(int)
    d["label"] = d["label"].fillna("").astype(str).str.strip().str.lower()
    fx, fy = [], []
    for kk in (5, 10, 15, 20, 25, 30, 40, 50):
        ce = BM.ceil_sim(F, BASE_N, kk, 1)
        rr = float(np.mean([
            len({p for p in x[(x["rank"] <= kk) & (x["label"] == "yes")]["docid"]}
                & set(x[x.in_gold == 1]["docid"])) / max(len(set(x[x.in_gold == 1]["docid"])), 1)
            for _, x in d.groupby(["cluster", "facet"])]))
        fx.append(kk / BASE_N)
        fy.append(rr / ce)
    rho = lambda f: float(np.interp(f, fx, fy))       # noqa: E731

    base = None
    rows = []
    for N in (50, 200, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 3000):
        k = min(M, N)
        ce = BM.ceil_sim(F, N, k, 200)
        fr = k / N
        R = ce * rho(fr)
        h = 2 * P * R / (P + R) if (P + R) else 0.0
        f1h = h
        f1m = h - 0.0221                            # 与定稿报数同口径（Jensen 差，模型内标定）
        if N == BASE_N:
            base = f1m
        rows.append(dict(N=N, k=k, fr=fr, uncov=1 - fr, ceil=ce, rho=rho(fr), R=R,
                         F1h=f1h, F1=f1m))

    print("=" * 122)
    print(f"【拐点扫描】预算 **¥{X:.2f}/题** ｜ 单价 ¥{C}/次 → **可判 {M} 篇** ｜ "
          f"P_judge 假设 {P}")
    print(f"  基准（N=50 全判）= setF1 {base:.4f}")
    print(f"\n  {'N(语料)':>9}{'k':>6}{'k/N':>8}{'未被判官看过':>13}{'ceil@k':>9}{'ρ':>7}"
          f"{'R':>7}{'setF1':>8}{'降幅':>8}{'判官秒':>8}")
    for r in rows:
        drop = r["F1"] - base
        rel = drop / base
        mark = "" if r["N"] > M else "  ← 全判"
        print(f"  {r['N']:>9,}{r['k']:>6,}{r['fr']:>8.1%}{r['uncov']:>13.1%}"
              f"{r['ceil']:>9.4f}{r['rho']:>7.3f}{r['R']:>7.3f}{r['F1']:>8.4f}"
              f"{rel:>+8.1%}{r['k'] * 0.8 / 8:>8.1f}{mark}")

    print("\n" + "=" * 122)
    print("【★ 三个阈值：什么 N 才算\"降幅可接受\"】")
    for tol, lab in ((0.05, "降幅 ≤5%"), (0.10, "≤10%"), (0.20, "≤20%")):
        hit = [r for r in rows if (base - r["F1"]) / base <= tol]
        if hit:
            r = max(hit, key=lambda z: z["N"])
            print(f"  {lab:<10} → N ≤ **{r['N']:,} 篇**（该点 setF1 {r['F1']:.4f}，"
                  f"k/N {r['fr']:.0%}）")
        else:
            print(f"  {lab:<10} → **无**（N=50 起就已超；本档预算下做不到）")

    print("\n" + "=" * 122)
    print("【★ 四个\"检索权重\"位置】")
    for target, lab in ((1.0, "k/N=100%（刚好全判）"), (0.5, "k/N=50%"),
                        (1 / 3, "k/N=33%（**你的 1500 篇**）"), (0.25, "k/N=25%")):
        N = int(round(M / target))
        k = min(M, N)
        ce = BM.ceil_sim(F, N, k, 200)
        R = ce * rho(k / N)
        f1 = (2 * P * R / (P + R) if (P + R) else 0) - 0.0221
        print(f"  {lab:<28} N = {N:>7,} 篇 ｜ k = {k:,} ｜ 未被判官看过 "
              f"{1 - k / N:>5.1%} ｜ ceil {ce:.3f} ｜ R {R:.3f} ｜ "
              f"setF1 **{f1:.4f}**（相对基准 "
              f"{(f1 - base) / base:+.1%}）")

    print("\n" + "=" * 122)
    print("【★ 检索改进的边际价值】`ceil` 每 **+0.10**（= 排序把 gold 更靠前）换来多少 setF1")
    print(f"  {'N(语料)':>9}{'k/N':>8}{'ceil@k':>9}{'ρ':>7}{'R':>7}{'setF1':>8}"
          f"{'ceil+0.1 后':>12}{'ΔsetF1':>9}{'检索还有空间？':>14}")

    def _f1(rr: float) -> float:
        h = 2 * P * rr / (P + rr) if (P + rr) else 0.0
        return h - 0.0221

    for N in (50, 497, 1500, 10000):
        k = min(M, N)
        ce = BM.ceil_sim(F, N, k, 200)
        r0 = rho(k / N)
        R = ce * r0
        R2 = min(1.0, ce + 0.10) * r0
        room = "**0（已满）**" if ce >= 0.999 else f"有（{1 - ce:.0%} 未覆盖）"
        print(f"  {N:>9,}{k / N:>8.1%}{ce:>9.3f}{r0:>7.3f}{R:>7.3f}{_f1(R):>8.4f}"
              f"{_f1(R2):>12.4f}{_f1(R2) - _f1(R):>+9.4f}{room:>14}")

    print("\n" + "=" * 122)
    print("【结论】")
    print(f"  · **拐点在 N = {M} 篇**（k/N 刚过 100%）：语料 ≤ {M} 时全判，指标恒定；"
          f"超过就开始掉 → \"检索有意义\"**从这里就开始了**，不必等到 1500")
    print(f"  · 你估的 **1500 篇**：未被判官看过 "
          f"{1 - min(M, 1500) / 1500:.0%} 的语料 → 检索确实承担主要筛选；"
          f"但 setF1 已降约 "
          f"{(min(r['F1'] for r in rows if r['N'] == 1500) - base) / base:+.1%}（不是\"不会降太多\"）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
