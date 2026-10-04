"""汇总**定稿配置**的各项指标（v2 口径 + v1 对照）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    d1 = pd.read_csv(RES / "R2_C3_W1_SWEEP_v1.csv")
    d2 = pd.read_csv(RES / "R2_C3_W1_SWEEP_v2.csv")
    print("=" * 104)
    print("【定稿配置 @k=30】②C3 窗口判")
    print(f"  {'窗口子查询':<10}{'setP':>8}{'setR':>8}{'setF1':>9}{'交付篇/题':>11}")
    for tag, d in (("v1（旧）", d1), ("**v2（定稿）**", d2)):
        t = d[(d.k == 30) & (d.arm == "②C3窗口判")]
        print(f"  {tag:<10}{t.setP.mean():>8.3f}{t.setR.mean():>8.3f}"
              f"{t.setF1.mean():>9.3f}{t.n_pred.mean():>11.1f}")
    t0 = d2[(d2.k == 30) & (d2.arm == "①交付即正")]
    t2 = d2[(d2.k == 30) & (d2.arm == "②C3窗口判")]
    print(f"\n  {'①不判（对照）':<10}{t0.setP.mean():>8.3f}{t0.setR.mean():>8.3f}"
          f"{t0.setF1.mean():>9.3f}{t0.n_pred.mean():>11.1f}")
    print(f"  ★ **C3 判官增益 ΔF1 = {t2.setF1.mean() - t0.setF1.mean():+.3f}**"
          f"（精确率 {t0.setP.mean():.3f} → {t2.setP.mean():.3f}）")

    print("\n" + "=" * 104)
    print("【分簇 @k=30】")
    print(f"  {'簇':>3}{'gold/题':>8}{'①P':>7}{'①R':>7}{'①F1':>7}"
          f"{'②P':>7}{'②R':>7}{'②F1':>7}{'②交付':>7}")
    for c in (1, 2, 3):
        a = t0[t0.cluster == c]
        b = t2[t2.cluster == c]
        print(f"  {c:>3}{a.n_gold.mean():>8.1f}{a.setP.mean():>7.3f}{a.setR.mean():>7.3f}"
              f"{a.setF1.mean():>7.3f}{b.setP.mean():>7.3f}{b.setR.mean():>7.3f}"
              f"{b.setF1.mean():>7.3f}{b.n_pred.mean():>7.1f}")

    print("\n" + "=" * 104)
    print("【全 k 曲线（v2 定稿）】")
    print(f"  {'k':>4}{'①F1':>8}{'②P':>7}{'②R':>7}{'②F1':>8}{'②交付':>7}{'C3调用/题':>10}")
    for k in sorted(d2.k.unique()):
        a = d2[(d2.k == k) & (d2.arm == "①交付即正")]
        b = d2[(d2.k == k) & (d2.arm == "②C3窗口判")]
        print(f"  {k:>4}{a.setF1.mean():>8.3f}{b.setP.mean():>7.3f}{b.setR.mean():>7.3f}"
              f"{b.setF1.mean():>8.3f}{b.n_pred.mean():>7.1f}{k * 2.1:>10.0f}")

    print("\n" + "=" * 104)
    print("【检索阶段单独看（①=纯检索，不判）】")
    print(f"  {'k':>4}{'setP':>8}{'setR':>8}{'setF1':>9}")
    for k in sorted(d2.k.unique()):
        a = d2[(d2.k == k) & (d2.arm == "①交付即正")]
        print(f"  {k:>4}{a.setP.mean():>8.3f}{a.setR.mean():>8.3f}{a.setF1.mean():>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
