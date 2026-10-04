"""真值重标 v2 的判官行为诊断：规则是否偏严？两判官谁在主导？"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
df = pd.read_csv(HERE / "data" / "r2dev" / "gold_recalib2.csv")
df["anchor_gold"] = df["anchor_gold"].astype(bool)
df["new_gold"] = df["new_gold"].astype(str).str.lower().isin(["yes", "y", "true"])
df["A_label"] = df["A_label"].astype(str).str.lower()
df["B_label"] = df["B_label"].astype(str).str.lower()
n = len(df)

print("=" * 100)
print(f"【1】判官标签分布（n={n}）")
print(f"  {'label':<12}{'A(deepseek-chat)':>18}{'B(glm-4-flash)':>18}")
for lab in ("entail", "partial", "neutral", "contradict", "err"):
    a = (df.A_label == lab).mean()
    b = (df.B_label == lab).mean()
    if a or b:
        print(f"  {lab:<12}{a:>18.1%}{b:>18.1%}")

print(f"\n【2】采信路径")
both_e = (df.A_label == "entail") & (df.B_label == "entail")
none_e = ~df.A_label.eq("entail") & ~df.B_label.eq("entail") & (df.A_label != "err") & (df.B_label != "err")
third = df.need_third.astype(bool)
print(f"  双方 entail（→yes）        {both_e.mean():>7.1%}  ({int(both_e.sum())})")
print(f"  双方非 entail（→no）       {none_e.mean():>7.1%}  ({int(none_e.sum())})")
print(f"  分歧 → 第三轮              {third.mean():>7.1%}  ({int(third.sum())})")

print(f"\n【3】第三轮（严格对抗）结果分布")
t = df[third]
print(f"  第三轮 YES → 采信为正例    {(t.third.astype(str).str.upper().str.startswith('Y')).mean():>7.1%}"
      f"  ({int(t.third.astype(str).str.upper().str.startswith('Y').sum())}/{len(t)})")
print(f"  → 分歧里有 {(t.third.astype(str).str.upper().str.startswith('Y')).mean():.0%} 被判正例")

print(f"\n【4】双方非 entail 里，**有一方是 entail** 的比例（= 被分歧规则挡掉的）")
part = df[~both_e & (df.A_label.eq("entail") | df.B_label.eq("entail"))]
print(f"  数量 {len(part)}（占 {len(part) / n:.1%}）→ 全部走第三轮")
if len(part):
    y = part.third.astype(str).str.upper().str.startswith("Y").mean()
    print(f"  其中第三轮判 YES 的 {y:.1%} → 说明**单方 entail 常是真阳性**，只靠 B 会漏")

print(f"\n【5】关键：B 判 neutral 而 A 判 entail 的子集")
sub = df[(df.A_label == "entail") & (df.B_label == "neutral")]
if len(sub):
    y = sub.third.astype(str).str.upper().str.startswith("Y").mean()
    print(f"  数量 {len(sub)} ｜ 第三轮 YES {y:.1%}")
    print(f"  → 若这个比例高，说明 **B(glm-4-flash) 偏保守**，规则整体偏严")

print(f"\n【6】把「双方 entail」放宽成「任一方 entail + 第三轮确认」后的真值规模")
alt = both_e | (third & df.third.astype(str).str.upper().str.startswith("Y"))
print(f"  现规则正例 {df.new_gold.mean():>7.1%}（{int(df.new_gold.sum())}）")
print(f"  放宽后正例 {alt.mean():>7.1%}（{int(alt.sum())}）"
      f"  ← 差 {int(alt.sum()) - int(df.new_gold.sum())} 条")
print(f"  锚点       {df.anchor_gold.mean():>7.1%}（{int(df.anchor_gold.sum())}）")

print(f"\n【7】每簇新真值篇数/题")
s = df.groupby(["cluster", "facet"]).agg(n_new=("new_gold", "sum"),
                                        n_anchor=("anchor_gold", "sum")).reset_index()
print(f"  {s.groupby('cluster').n_new.mean().round(2).to_dict()}")
print(f"  极端：新真值 = 0 的题 → {s[s.n_new == 0][['cluster', 'facet']].to_dict('records')}")
