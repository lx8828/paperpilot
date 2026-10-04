"""**修路 #1｜证据预算 sweep 对照 + 逐 facet 副作用体检**

配置来自 `R2_r2reader2_4_b{b}_a{a}.csv`；prompt token 从各次运行日志记录（CSV 里没存）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# prompt token 总量（来自各次运行的 `llm.usage_stats`，451~452 次 map 调用）
PROMPT = {(6, 0): 1_523_343, (6, 1): 1_552_594, (6, 2): 1_574_650, (6, 3): 1_599_682,
          (12, 0): 2_734_507, (12, 4): 2_764_572}


def load() -> dict[tuple[int, int], pd.DataFrame]:
    out = {}
    for p in sorted(RES.glob("R2_r2reader2_4_b*_a*.csv")):
        m = re.search(r"_b(\d+)_a(\d+)\.csv$", p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = pd.read_csv(p)
    return out


def main() -> int:
    D = load()
    base = D[(6, 0)]
    print("=" * 122)
    print("【修路 #1｜证据预算 sweep】同 28 题 / 真值=三判官多数票 / 生产切块 / 协议 4")
    print(f"  {'b':>3}{'锚点a':>6}{'集合P':>9}{'集合R':>9}{'集合F1':>10}{'ΔF1':>8}{'判yes/题':>10}"
          f" ｜{'code_release F1':>18}{'Δ':>8}{'判yes':>7}{'gold':>6}{'prompt':>12}{'相对':>9}{'F1/1%':>9}")
    rows = []
    for (b, a), d in sorted(D.items()):
        cr = d[d.facet == "code_release"]
        tk = PROMPT.get((b, a), 0)
        rel = (tk / PROMPT[(6, 0)] - 1) if tk else float("nan")
        d_f1 = d.reader_F1.mean() - base.reader_F1.mean()
        rows.append(dict(b=b, a=a, P=d.reader_P.mean(), R=d.reader_R.mean(),
                         F1=d.reader_F1.mean(), dF1=d_f1, n_yes=d.n_yes.mean(),
                         cr=cr.reader_F1.mean(), cr_yes=cr.n_yes.mean(), cr_gold=cr.n_gold.mean(),
                         tok=tk, rel=rel,
                         eff=(d_f1 / (rel * 100) if rel and rel > 0 else float("nan"))))
    for r in rows:
        print(f"  {r['b']:>3}{r['a']:>6}{r['P']:>9.3f}{r['R']:>9.3f}{r['F1']:>10.3f}{r['dF1']:>+8.3f}"
              f"{r['n_yes']:>10.2f} ｜{r['cr']:>18.3f}{r['cr'] - rows[0]['cr']:>+8.3f}"
              f"{r['cr_yes']:>7.2f}{r['cr_gold']:>6.1f}{r['tok']:>12,}{r['rel']:>+9.1%}"
              f"{r['eff']:>9.4f}")

    print(f"\n  {'★ 性价比（ΔF1 / 每 +1% prompt）':<40}")
    for r in sorted([x for x in rows if x["eff"] == x["eff"]], key=lambda x: -x["eff"])[:4]:
        print(f"     b={r['b']} a={r['a']}：ΔF1 {r['dF1']:+.3f} / prompt {r['rel']:+.1%}"
              f"  →  **{r['eff']:.4f}**")

    # ── 逐 facet 副作用体检 ──
    for cfg in [(6, 2), (12, 4)]:
        if cfg not in D:
            continue
        d = D[cfg]
        m = base.merge(d, on=["cluster", "facet"], suffixes=("_0", "_x"))
        m["d"] = m.reader_F1_x - m.reader_F1_0
        m["dyes"] = m.n_yes_x - m.n_yes_0
        print("\n" + "=" * 122)
        print(f"【逐 facet 副作用体检】b={cfg[0]} a={cfg[1]} vs 基线 b=6 a=0"
              f"（ΔF1 升序；**负=被锚点注入带坏**）")
        print(f"  {'簇':>3} {'facet':<17}{'gold':>5}{'F1基线':>9}{'F1现':>8}{'ΔF1':>8}"
              f"{'Δ判yes':>9}   备注")
        for _, r in m.sort_values("d").iterrows():
            note = ("⚠️ 退化" if r.d < -0.02 else ("✅ 改善" if r.d > 0.02 else ""))
            if r.dyes > 2:
                note += " ｜ 明显多判（假阳性风险）"
            if r.dyes < -2:
                note += " ｜ 明显少判"
            print(f"  {int(r.cluster):>3} {str(r.facet):<17}{int(r.n_gold_0):>5}"
                  f"{r.reader_F1_0:>9.3f}{r.reader_F1_x:>8.3f}{r.d:>+8.3f}{r.dyes:>+9.0f}   {note}")
        n_up = int((m.d > 0.02).sum())
        n_dn = int((m.d < -0.02).sum())
        print(f"\n  → 改善 {n_up} 题 ｜ 退化 {n_dn} 题 ｜ 持平 {len(m) - n_up - n_dn} 题"
              f" ｜ 判 yes 总量 {base.n_yes.sum():.0f} → {d.n_yes.sum():.0f}"
              f"（gold 合计 {base.n_gold.sum():.0f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
