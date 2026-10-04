"""从 `R2_L3_NLI.csv` 打印 L3 对照表（避免重跑 4 分钟）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]

d = pd.read_csv(HERE / "results" / "R2_L3_NLI.csv")
BS = [1, 3, 6, 12]

print("【NLI 判定 vs 词面代理】claim = 按篇级分取 top-|gold|（隔离判定误差，只看证据）")
print(f"  {'b':>3}  {'指标':<28}{'词面代理':>10}{'NLI@.5':>9}{'NLI@.3':>9}{'NLI@.2':>9}")
for b in BS:
    t = d[d.b == b]
    for nm, pc, nc in (("cite_recall 论断有支持", "cite_recall_proxy", "cite_recall_nli"),
                       ("cite_precision 引用块支持率", "cite_precision_proxy", "cite_precision_nli")):
        g = t.groupby("thr")[nc].mean()
        print(f"  {b:>3}  {nm:<28}{t[pc].mean():>10.3f}{g.get(0.5, float('nan')):>9.3f}"
              f"{g.get(0.3, float('nan')):>9.3f}{g.get(0.2, float('nan')):>9.3f}")

print("\n【按是否落在真值内】(b=3, NLI@0.3)")
t = d[(d.b == 3) & (d.thr == 0.3)]
print(f"  {'组':<10}{'n':>5}{'cite_recall':>12}{'cite_precision':>15}{'contra':>9}{'entail均值':>11}")
for g, sub in t.groupby("in_gold"):
    tag = "在真值" if g else "不在真值"
    print(f"  {tag:<10}{len(sub):>5}{sub['cite_recall_nli'].mean():>12.3f}"
          f"{sub['cite_precision_nli'].mean():>15.3f}{sub['contra_nli'].mean():>9.3f}"
          f"{sub['ent_concat'].mean():>11.3f}")

print("\n【entail 绝对分分布】(b=1)")
t = d[(d.b == 1) & (d.thr == 0.5)]
print("  分位 10/25/50/75/90 :", {q: round(float(t["ent_concat"].quantile(q)), 3)
                                 for q in (.1, .25, .5, .75, .9)})
print(f"  ≥0.5 比例 {(t['ent_concat'] >= 0.5).mean():.3f} ｜ ≥0.3 {(t['ent_concat'] >= 0.3).mean():.3f}")

print("\n【contradiction（新能力：抓「说反了」）】")
for b in BS:
    t = d[(d.b == b) & (d.thr == 0.5)]
    print(f"  b={b:<3} 至少一块被判矛盾 {t['contra_nli'].mean():.3f}"
          f" ｜ 引用块整体被判矛盾 {t['con_concat'].mean():.3f}")

cal = pd.read_csv(HERE / "data" / "r2dev" / "nli_calibration.csv")
print(f"\n【人工校准集】{len(cal)} 条 → data/r2dev/nli_calibration.csv")
print("  分层：", dict(cal["分层"].value_counts()))
print(cal[["分层", "论断", "NLI_entail", "NLI_contra", "构造标签", "词面命中"]]
      .head(10).to_string(index=False))
