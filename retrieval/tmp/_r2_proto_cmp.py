"""判官协议对照：reader 旧三档（`p3`）vs 真值四档（`p4`）vs 真值完整协议（`dual`）。

检验的假设：reader 在 `code_release`/`fine_tuning` 上**系统偏严**（判 yes 远少于 gold），
          原因是**协议不一致**（reader 三档 vs 真值四档）。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

files = {"p3": RES / "R2_r2reader2_3_b6_a0.csv",
         "p4": RES / "R2_r2reader2_4_b6_a0.csv",
         "dual": RES / "R2_r2reader2_dual_b6_a0.csv"}
D = {k: pd.read_csv(v) for k, v in files.items() if v.exists()}

print("=" * 112)
print("【判官协议对照（同 28 题 / 同真值 / 同检索结果）】")
print(f"  {'协议':<8}{'题数':>5}{'集合P':>9}{'集合R':>9}{'集合F1':>10}{'判yes/题':>11}{'gold/题':>9}")
for k, d in D.items():
    print(f"  {k:<8}{len(d):>5}{d.reader_P.mean():>9.3f}{d.reader_R.mean():>9.3f}"
          f"{d.reader_F1.mean():>10.3f}{d.n_yes.mean():>11.2f}{d.n_gold.mean():>9.2f}")

print("\n【逐 facet：三协议 F1 与 判yes−gold】")
keys = sorted(set(D["p3"].apply(lambda r: (int(r.cluster), str(r.facet)), axis=1))) if "p3" in D else []
print(f"  {'簇':>3} {'facet':<17}{'gold':>5}"
      + "".join(f"{k + ' F1':>9}{k + ' yes':>9}" for k in D))
for (c, f) in keys:
    row = f"  {c:>3} {f:<17}"
    g = None
    cells = ""
    for k, d in D.items():
        x = d[(d.cluster == c) & (d.facet == f)]
        if not len(x):
            cells += f"{'—':>9}{'—':>9}"
            continue
        r = x.iloc[0]
        g = int(r.n_gold) if g is None else g
        cells += f"{r.reader_F1:>9.3f}{int(r.n_yes):>9}"
    print(f"{row}{g if g is not None else '':>5}{cells}")

print("\n【假设检验：偏严是否被修掉？】")
for (c, f) in keys:
    a = D.get("p3")
    b = D.get("p4")
    if a is None or b is None:
        continue
    xa = a[(a.cluster == c) & (a.facet == f)]
    xb = b[(b.cluster == c) & (b.facet == f)]
    if len(xa) and len(xb) and int(xa.iloc[0].n_gold) - int(xa.iloc[0].n_yes) >= 3:
        print(f"  偏严 {c}·{f:<17} gold {int(xa.iloc[0].n_gold)} ｜ "
              f"三档 yes {int(xa.iloc[0].n_yes)} (F1 {xa.iloc[0].reader_F1:.3f}) → "
              f"四档 yes {int(xb.iloc[0].n_yes)} (F1 {xb.iloc[0].reader_F1:.3f})")
