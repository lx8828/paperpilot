"""新旧 reader 结果对照（同指标：reader 集合 P/R/F1 + 检索基线 F1）。

⚠️ **不是 apples-to-apples**：新旧**题集与真值都不同**（旧 35 题/较松真值；新 28 题/三判官较严真值），
只用于说明"接线后指标落在哪个量级"，不构成"改善/退化"的证据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

files = sorted(RES.glob("R2_r2reader*.csv"))
print("=" * 104)
print(f"{'文件':<26}{'题数':>5}{'gold/题':>8}{'readerP':>9}{'readerR':>9}"
      f"{'readerF1':>10}{'retF1':>8}{'n_yes':>7}")
for f in files:
    d = pd.read_csv(f)
    if "reader_F1" not in d.columns:
        continue
    tag = "★ 新口径" if "r2reader2" in f.name else "　旧口径"
    print(f"{f.name:<26}{len(d):>5}{d.n_gold.mean():>8.2f}{d.reader_P.mean():>9.3f}"
          f"{d.reader_R.mean():>9.3f}{d.reader_F1.mean():>10.3f}{d.ret_F1.mean():>8.3f}"
          f"{d.n_yes.mean():>7.2f}  {tag}")

print("\n【新口径逐 facet】（按 F1 升序，看短板在哪）")
new = RES / "R2_r2reader2_b6.csv"
if new.exists():
    d = pd.read_csv(new)
    print(f"  {'簇':>3} {'facet':<17}{'gold':>5}{'判yes':>6}{'readerF1':>10}{'retF1':>8}")
    for _, r in d.sort_values("reader_F1").iterrows():
        print(f"  {int(r.cluster):>3} {str(r.facet):<17}{int(r.n_gold):>5}{int(r.n_yes):>6}"
              f"{r.reader_F1:>10.3f}{r.ret_F1:>8.3f}")
    print(f"\n  ➖ 偏严（判 yes 明显少于 gold）：")
    for _, r in d[d.n_yes < d.n_gold - 1].sort_values("reader_F1").iterrows():
        print(f"     簇{int(r.cluster)} {r.facet:<17} gold {int(r.n_gold)} vs 判 yes {int(r.n_yes)}"
              f"  F1 {r.reader_F1:.3f}")
    print(f"  ➕ 偏松（判 yes 明显多于 gold）：")
    for _, r in d[d.n_yes > d.n_gold + 1].sort_values("reader_F1").iterrows():
        print(f"     簇{int(r.cluster)} {r.facet:<17} gold {int(r.n_gold)} vs 判 yes {int(r.n_yes)}"
              f"  F1 {r.reader_F1:.3f}")
