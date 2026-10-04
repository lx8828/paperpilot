"""从 `R2_RD_*.csv` 打表（秒级，不重跑编码）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D_GRID = [3, 5, 10, 20, 50]
ORDER = ["dense", "4:1", "2:1", "1:1", "1:2", "1:4", "bm25"]
KS = [5, 10, 13, 20]


def main() -> int:
    A = pd.read_csv(RES / "R2_RD_GRID.csv")
    B = pd.read_csv(RES / "R2_RD_AGG.csv")
    C = pd.read_csv(RES / "R2_RD_KCUT.csv")
    print("=" * 118)
    print("【表 A｜d × ρ 网格】k=13 ｜ 篇分=篇内 max（n_keep=1）｜ 变体 nopara")
    a = A[A["k"] == 13]
    piv = a.pivot_table(index="d", columns="rho", values="setF1", aggfunc="mean")
    print(f"  {'d':<5}" + "".join(f"{o:>9}" for o in ORDER if o in piv.columns))
    for d in D_GRID:
        if d in piv.index:
            print(f"  {d:<5}" + "".join(f"{piv.loc[d, o]:>9.3f}"
                                       for o in ORDER if o in piv.columns))
    ag = a.groupby(["d", "rho"]).agg(f1=("setF1", "mean")).reset_index()
    rb = ag.sort_values("f1", ascending=False).iloc[0]
    BRHO = rb["rho"]
    print(f"  ★ 最优：d={int(rb['d'])} ｜ ρ=**{BRHO}** → setF1 **{rb['f1']:.3f}**")

    print("\n【表 A2｜各 k 下 ρ 的方向】setF1")
    print(f"  {'k':<5}" + "".join(f"{o:>9}" for o in ORDER))
    for k in KS:
        t = A[A["k"] == k]
        g = t.groupby("rho").setF1.mean()
        print(f"  {k:<5}" + "".join(f"{g.get(o, float('nan')):>9.3f}" for o in ORDER))

    print("\n" + "=" * 118)
    print(f"【表 B｜段级聚合】d=50 ｜ k=13 ｜ 按 ρ 分组（n_keep × 聚合）")
    b = B[B["k"] == 13]
    for rho in [BRHO, "1:1"]:
        t = b[b["rho"] == rho]
        if not len(t):
            continue
        print(f"  ρ={rho}：" + "".join(f"{x:>11}" for x in ("max", "sum", "mean")))
        for nk in [1, 3, 6]:
            row = ""
            for aggm in ("max", "sum", "mean"):
                s = t[(t["nk"] == nk) & (t["aggm"] == aggm)]
                row += f"{s['setF1'].mean():>11.3f}" if len(s) else f"{'-':>11}"
            print(f"    n_keep={nk} {row}")
        bg = t.groupby(["nk", "aggm"]).setF1.mean()
        print(f"    ★ 该 ρ 下最优：{bg.idxmax()} → {bg.max():.3f}")

    print("\n" + "=" * 118)
    print("【表 C｜统一候选块 top-K 截断】d=50, n_keep=3, 聚合=mean, k=13")
    c = C[C["k"] == 13]
    for rho in [BRHO, "1:1"]:
        t = c[c["rho"] == rho]
        if not len(t):
            continue
        print(f"  ρ={rho}：")
        for kc in [1, 2, 5, 0]:
            s = t[t["kc"] == kc]
            if not len(s):
                continue
            lbl = "不截(全库)" if kc == 0 else f"{kc}×k×nk"
            print(f"    {lbl:<12}块数 {s['K_chunks'].mean():>7.0f} ｜ "
                  f"StRecall {s['StRecall'].mean():.3f} ｜ setP {s['setP'].mean():.3f} ｜ "
                  f"setF1 {s['setF1'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
