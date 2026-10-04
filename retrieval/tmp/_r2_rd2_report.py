"""读 `R2_RD2_*.csv` 打表（秒级）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D_CURVE = [5, 10, 20, 30, 50, 75, 100, 150, 300, 0]


def main() -> int:
    d = pd.read_csv(RES / "R2_RD2_CURVE.csv")
    print("=" * 112)
    print("【逐类 d 曲线】该类各列表 **top-d 块的并集** 覆盖多少 gold（召回上界，与 k 无关）")
    print("  说明：`不截`= 全库（该类的列表排满 1,094 块）→ 覆盖率必为 1.000，是平凡上界；")
    print("        真正要看的是**早期斜率**与**边际拐点**。")
    print(f"  {'类':<10}{'d':>6}{'覆盖gold':>10}{'边际':>9}{'块数':>8}{'篇数':>7}")
    for cls in ("zh-dense", "sq-dense", "sq-bm25"):
        t = d[d["cls"] == cls]
        for dd in D_CURVE:
            s = t[t["d"] == dd]
            if not len(s):
                continue
            print(f"  {cls if dd == 5 else '':<10}{('不截' if dd == 0 else str(dd)):>6}"
                  f"{s['cov'].mean():>10.3f}{s['marginal'].mean():>+9.3f}"
                  f"{s['n_chunks'].mean():>8.0f}{s['n_docs'].mean():>7.1f}")

    g = pd.read_csv(RES / "R2_RD2_GRID.csv")
    g13 = g[g["k"] == 13]
    print("\n" + "=" * 112)
    print("【固定 ρ=1:4 下的 d_zh 曲线】看 d_zh 最优是否落在网格边界")
    print(f"  {'d_zh':>6}" + "".join(f"{x:>14}" for x in ("d_sd=50,d_sb=50", "d_sd=100,d_sb=50")))
    for dzh in sorted(g13["dzh"].unique()):
        row = ""
        for dsd in (50, 100):
            s = g13[(g13["dzh"] == dzh) & (g13["dsd"] == dsd) & (g13["dsb"] == 50)
                    & (g13["rho"] == "1:4")]
            row += f"{s['setF1'].mean():>14.3f}" if len(s) else f"{'-':>14}"
        print(f"  {dzh:>6}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
