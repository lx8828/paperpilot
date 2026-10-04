"""从 `calib2.csv` 出校准报表（不重跑 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]
df = pd.read_csv(HERE / "data" / "r2dev" / "calib2.csv")
ok = df[df["LLM判定"] != "ERROR"].copy()

print(f"样本 {len(df)} 条 ｜ 证据长度 中位 {int(df['块长'].median())} 字（b=1，不截断）")
print("LLM 判定分布：" + " ｜ ".join(f"{k} {v}" for k, v in ok["LLM判定"].value_counts().items()))
print("NLI 判定分布：" + " ｜ ".join(f"{k} {v}" for k, v in ok["NLI_判定"].value_counts().items()))
agree = float((ok["NLI_判定"].str.startswith("entail") == (ok["LLM判定"] == "entail")).mean())
print(f"两者三分类完全一致率：{float((ok['NLI_判定'].map(lambda s: 'entail' if 'entail' in s else ('contradict' if 'contra' in s else 'neutral')) == ok['LLM判定']).mean()):.3f}")

y = (ok["LLM判定"] == "entail").astype(int).values
print(f"\n【① NLI 阈值校准】正类 = LLM 判 entail（{int(y.sum())}/{len(y)}）")
print(f"  {'阈值':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'F1':>7}")
best = None
for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    pred = (ok["NLI_entail"] >= thr).astype(int).values
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    mk = ""
    if best is None or f1 > best[1]:
        best, mk = (thr, f1), "  ← 最优"
    print(f"  {thr:>6.2f}{tp:>5}{fp:>5}{fn:>5}{p:>11.3f}{r:>8.3f}{f1:>7.3f}{mk}")

print("\n【② 锚点（词面真值）精度 vs LLM 判定】")
print(pd.crosstab(ok["构造标签"], ok["LLM判定"]).to_string())
sup = ok[ok["构造标签"] == "支持"]
unn = ok[ok["构造标签"] == "不支持"]
print(f"  构造「支持」{len(sup)} 条 → LLM 判 entail {float((sup['LLM判定'] == 'entail').mean()):.3f}"
      f"（partial {float((sup['LLM判定'] == 'partial').mean()):.3f}）")
print(f"  构造「不支持」{len(unn)} 条 → LLM 判 entail（假阳）{float((unn['LLM判定'] == 'entail').mean()):.3f}")
tp = int(((ok["构造标签"] == "支持") & (ok["LLM判定"] == "entail")).sum())
fp = int(((ok["构造标签"] == "不支持") & (ok["LLM判定"] == "entail")).sum())
fn = int(((ok["构造标签"] == "支持") & (ok["LLM判定"] != "entail")).sum())
print(f"  → 词面锚点 vs LLM（entail 为正类）：TP {tp} / FP {fp} / FN {fn} → "
      f"precision {tp / max(tp + fp, 1):.3f} ｜ recall {tp / max(tp + fn, 1):.3f}")

print("\n【③ contradiction 是否成立】")
con = ok[ok["NLI_contra"] >= 0.5]
print(f"  NLI 报矛盾 {len(con)} 条 → LLM 分布：" +
      (" ｜ ".join(f"{k} {v}" for k, v in con["LLM判定"].value_counts().items()) or "无"))
if len(con):
    print(f"  → LLM 认同「说反了」比例：{float((con['LLM判定'] == 'contradict').mean()):.3f}")
print(f"  LLM 判 contradict 的条数：{int((ok['LLM判定'] == 'contradict').sum())}")

print("\n【④ 谁更可信：抽样看分歧】")
dis = ok[((ok["NLI_entail"] >= 0.7) & (ok["LLM判定"] != "entail"))
         | ((ok["NLI_entail"] < 0.3) & (ok["LLM判定"] == "entail"))]
print(f"  NLI 高但 LLM 非 entail，或 NLI 低但 LLM entail：{len(dis)}/{len(ok)} 条")
for _, r in dis.head(10).iterrows():
    print("-" * 88)
    print(f"[{r['构造标签']}] {r['论断']} ｜ NLI ent={r['NLI_entail']} con={r['NLI_contra']}"
          f" ｜ LLM={r['LLM判定']}（{str(r['LLM理由'])[:46]}）")
    print(f"  证据：{str(r['证据'])[:220]}…")
