"""**单判官筛选 vs 三判官多数票** —— 决定"扩库时能否用 1 次 LLM 便宜地筛干扰项"

问题：加干扰项时若不想做"完整重标"（A+B 双判官 + 第三轮，2~3 次调用/对），
      能不能只用**单判官 1 次调用**判出"非正例"？
→ 关键是**假阴性率**：单判官判负的篇里，有多少其实是正例（多数票正例）。
   假阴性高 → 干扰项里混入真阳性 → 召回指标被污染（正确召回的反被算成假阳性）。

数据：`gold_recalib3.csv`（A/B 标签）+ `gold_adjudicate_gold_recalib3.csv`（第三判官 C）
      + `gold_final2.csv`（多数票最终真值）。**零 LLM 成本**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    a = pd.read_csv(DEV / "gold_recalib3.csv")
    c = pd.read_csv(DEV / "gold_adjudicate_gold_recalib3.csv")
    f = pd.read_csv(DEV / "gold_final2.csv")
    a["key"] = a.apply(lambda r: f"{int(r.cluster)}|{r.facet}|{r.docid}", axis=1)
    c["key"] = c["key"].astype(str)
    f["key"] = f.apply(lambda r: f"{int(r.cluster)}|{r.facet}|{r.docid}", axis=1)
    m = (a.merge(c[["key", "third_label"]], on="key", how="left")
          .merge(f[["key", "gold", "votes"]], on="key", how="left"))
    m = m.dropna(subset=["gold"])
    m["gold"] = m["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    for x, col in (("A", "A_label"), ("B", "B_label"), ("C", "third_label")):
        m[f"{x}_yes"] = m[col].astype(str).str.strip().str.lower().eq("entail")
    m["A_yes"] = m["A_yes"].fillna(False)

    print("=" * 110)
    print(f"【样本】{len(m):,} 对（{m.groupby(['cluster', 'facet']).ngroups} 个组合）"
          f" ｜ 多数票正例 {int(m.gold.sum())}")

    print("\n【1｜单判官 vs 多数票：逐篇混淆（以多数票为基准）】")
    print(f"  {'判官':<8}{'判正':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>7}"
          f"{'准':>7}{'召':>7}{'F1':>7}")
    for x in ("A", "B", "C"):
        p = m[f"{x}_yes"]
        g = m.gold
        tp, fp, fn, tn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        print(f"  {x:<8}{int(p.sum()):>7}{tp:>6}{fp:>6}{fn:>6}{tn:>7}"
              f"{pr:>7.3f}{rc:>7.3f}{(2 * pr * rc / (pr + rc) if pr + rc else 0):>7.3f}")

    print("\n【2｜★ 关键：单判官判「负」的篇里，有多少其实是**正例**（假阴性）】")
    print("  （扩库筛干扰项时，这些会变成「藏在库里的真阳性」→ 污染评测）")
    print(f"  {'判官':<8}{'判负篇数':>10}{'其中正例(FN)':>14}{'**假阴性率**':>13}")
    for x in ("A", "B", "C"):
        neg = ~m[f"{x}_yes"]
        fn = int((neg & m.gold).sum())
        print(f"  {x:<8}{int(neg.sum()):>10}{fn:>14}{fn / max(int(neg.sum()), 1):>13.2%}")

    print("\n【3｜若用「单判官判负」当干扰项，对**召回指标**的影响】")
    print("  ⚠️ 召回指标只看「gold 是否被召回到」→ **与干扰项真值无关**；")
    print("     受影响的是 **P/F1**（正确召回被算成假阳性）。量化一下：")
    for x in ("A",):
        fn = m[~m[f"{x}_yes"] & m.gold]
        print(f"  {x} 判负但实为正例：{len(fn)} 篇 / 全部正例 {int(m.gold.sum())} 篇"
              f" = **{len(fn) / max(int(m.gold.sum()), 1):.1%}** 的正例会「从真值里消失」")
        print(f"     分布（簇, facet）：")
        for (cl, fa), g2 in fn.groupby(["cluster", "facet"]):
            print(f"       {int(cl)}·{fa:<18}{len(g2):>3} 篇  {sorted(g2.docid)[:8]}")

    print("\n【4｜降级成本：用单判官代替多数票，逐题篇级集合指标差多少】")
    rows = []
    for (cl, fa), g2 in m.groupby(["cluster", "facet"]):
        gold = set(g2[g2.gold].docid)
        docs = list(g2.docid)
        for x in ("A", "B", "C"):
            yes = set(g2[g2[f"{x}_yes"]].docid)
            tp = len(yes & gold)
            p = tp / max(len(yes), 1)
            r = tp / max(len(gold), 1)
            rows.append(dict(cluster=cl, facet=fa, judge=x, P=p, R=r,
                             F1=(2 * p * r / (p + r) if p + r else 0.0), n_gold=len(gold)))
    d = pd.DataFrame(rows)
    print(f"  {'判官':<8}{'集合P':>9}{'集合R':>9}{'集合F1':>10}{'ΔF1 vs 多数票(=1.0)':>22}")
    for x in ("A", "B", "C"):
        t = d[d.judge == x]
        print(f"  {x:<8}{t.P.mean():>9.3f}{t.R.mean():>9.3f}{t.F1.mean():>10.3f}"
              f"{'（多数票=1.000，此为下界）':>22}")
    print("\n  读法：这一栏不是「A 有多差」，而是「**真值自己有多少不确定性**」。")
    print("       单判官与多数票的差距 ≈ 真值噪声的量级；用它筛干扰项会**继承同样的噪声**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
