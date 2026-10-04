r"""**C3 k 扫描**（离线，零新调用）——回答"窗口判下 k 该取多少"

## 为什么 k 需要重定
k=20 全量结果显示：**C3 窗口判的 setP = 0.986**（28 facet 里 24 个 P=1.000）。
→ **判官几乎不误杀**，那**扩大 k 就是"免费"提升召回**（P 不降），F1 随 k 单调上升。
→ 极端：**k=50（全交付）时 C3 必然完全复现 gold**（因为 gold 就是这个协议在 50 篇上的结果）
   ⇒ **k 不该由 F1 定，该由成本定。**

## 自检（关键）
`k=50` 时 `②C3` 的 setF1 **应当 = 1.000**。若不足 → 说明窗口构造/判官协议与产出 `gold_final3`
的那一版有差异（子查询版本、facet 正则表等），必须查出来。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_ksweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
KS = [5, 8, 10, 13, 20, 30, 40, 50]
CALLS_PER_PAIR = 2.1          # A + B + 分歧第三轮（实测 160/560 = 28.6% 分歧）


def main() -> int:
    L = pd.read_csv(HERE / "results" / "R2_C3_WINDOW_FULL.csv")
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    L["yes"] = L["final"].astype(str).str.lower().eq("yes")
    print(f"判官记录 {len(L)} ｜ facet {L.groupby(['cluster','facet']).ngroups}"
          f" ｜ A/B 分歧 {int((L.A != L.B).sum())} ｜ ERR "
          f"{int(L.A.astype(str).str.contains('ERR').sum() + L.B.astype(str).str.contains('ERR').sum())}")
    print(f"判官整体判正率 {L['yes'].mean():.1%} ｜ 窗口字符数中位 {L.ev_chars.median():.0f}")

    rows = []
    for (ci, fa), x in L.groupby(["cluster", "facet"]):
        gold = NG.get((int(ci), str(fa)), set())
        x = x.sort_values("rank")
        for k in KS:
            top = x[x["rank"] <= k]
            for tag, s in (("①交付即正", set(top.docid)),
                           ("②C3窗口判", set(top[top.yes].docid))):
                inter = len(s & gold)
                p = inter / max(len(s), 1)
                r = inter / max(len(gold), 1)
                rows.append(dict(cluster=int(ci), facet=fa, k=k, arm=tag,
                                 calls=len(top) * CALLS_PER_PAIR,
                                 n_pred=len(s), n_gold=len(gold), inter=inter,
                                 setP=p, setR=r,
                                 setF1=2 * p * r / (p + r) if (p + r) else 0.0))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / "R2_C3_KSWEEP.csv", index=False, encoding="utf-8-sig")

    # ── 自检 ──
    chk = R[(R.k == 50) & (R.arm == "②C3窗口判")]
    print(f"\n[自检] k=50 时 ②C3 的 setF1 = {chk.setF1.mean():.4f}（应 = 1.000）"
          f" ｜ setP {chk.setP.mean():.4f} ｜ setR {chk.setR.mean():.4f}")
    if chk.setF1.mean() < 0.999:
        bad = chk[chk.setF1 < 0.99][["cluster", "facet", "setP", "setR", "inter", "n_gold"]]
        print(f"  ⚠️ 未复现 gold 的 facet（{len(bad)} 个）：")
        print(bad.head(12).to_string(index=False))
        print("  → 说明窗口/判官口径与 gold 有差（子查询版本? facet 正则表?），需查。")

    # ── 主表 ──
    print("\n" + "=" * 116)
    print("【k 扫描】同批判官记录，离线切不同 k ｜ 调用来数 = k × 2.1 /题")
    print(f"  {'k':>4}{'①P':>7}{'①R':>7}{'①F1':>7}{'②留':>6}{'②P':>7}{'②R':>7}{'②F1':>7}"
          f"{'ΔF1':>8}{'②调用/题':>10}{'28题调用':>10}")
    for k in KS:
        a1 = R[(R.k == k) & (R.arm == "①交付即正")]
        a2 = R[(R.k == k) & (R.arm == "②C3窗口判")]
        print(f"  {k:>4}{a1.setP.mean():>7.3f}{a1.setR.mean():>7.3f}{a1.setF1.mean():>7.3f}"
              f"{a2.n_pred.mean():>6.1f}{a2.setP.mean():>7.3f}{a2.setR.mean():>7.3f}"
              f"{a2.setF1.mean():>7.3f}{a2.setF1.mean() - a1.setF1.mean():>+8.3f}"
              f"{a2.calls.mean():>10.0f}{a2.calls.sum():>10.0f}")

    # ── 逐簇（k=20 vs k=30）──
    print("\n" + "=" * 116)
    print("【逐簇：k=20 vs k=30】②C3窗口判")
    print(f"  {'簇':>3}{'k=20 P':>9}{'k=20 R':>9}{'k=20 F1':>10}{'k=30 P':>9}{'k=30 R':>9}"
          f"{'k=30 F1':>10}{'ΔF1':>8}")
    for c in (1, 2, 3):
        for arm in ("②C3窗口判",):
            t20 = R[(R.cluster == c) & (R.k == 20) & (R.arm == arm)]
            t30 = R[(R.cluster == c) & (R.k == 30) & (R.arm == arm)]
            print(f"  {c:>3}{t20.setP.mean():>9.3f}{t20.setR.mean():>9.3f}"
                  f"{t20.setF1.mean():>10.3f}{t30.setP.mean():>9.3f}{t30.setR.mean():>9.3f}"
                  f"{t30.setF1.mean():>10.3f}{t30.setF1.mean() - t20.setF1.mean():>+8.3f}")

    # ── 判官自身假阴（关键：这是 k 之外的天花板）──
    print("\n" + "=" * 116)
    print("【判官自身的天花板】k=50 下 gold 篇被判官判正的比例（= 判官召回上限）")
    t = R[(R.k == 50) & (R.arm == "②C3窗口判")]
    print(f"  ②R = {t.setR.mean():.4f} → **判官漏掉 {(1 - t.setR.mean()) * 100:.1f}% 的 gold 篇**")
    low = t.nsmallest(8, "setR")[["cluster", "facet", "n_gold", "inter", "setR"]]
    print("  最差的 8 个 facet：")
    print(low.to_string(index=False))
    print(f"\n  → 已写 results/R2_C3_KSWEEP.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
