"""**真值扩展**：原 47 篇（多数票）+ 新增 103 篇干扰项（重标）→ `gold_final3.csv`

## 为什么要判新增篇
新加的 103 篇是"同领域干扰项"，**但同领域 ≠ 一定不是正例**。
不判就把它们当负例 → 若其中真有不少正例，指标会被系统性压低，且"干扰项"这个说法站不住。
所以必须过一遍判官，并报告 **纯度**（= 新篇里真为正例的比例）。

## 合并口径
· 原 451 对：直接取 `gold_final2.csv`（三判官多数票，已封版）
· 新增 949 对：取 `gold_ext.csv`（`_r2_gold_recalib2.py` 双判官 + 分歧第三轮）
  → `gold = (new_gold == yes)`；`A_yes/B_yes/C_yes` 由标签还原；`votes` = 三者之和

产物：`data/r2dev/gold_final3.csv` ｜ 控制台打**干扰项纯度表**

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_extend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COLS = ["cluster", "facet", "docid", "gold", "votes", "A_yes", "B_yes", "C_yes",
        "anchor_gold"]


def main() -> int:
    base = pd.read_csv(DEV / "gold_final2.csv")
    ext = pd.read_csv(DEV / "gold_ext.csv")
    print("=" * 112)
    print(f"【真值扩展】原 {len(base)} 对（{base.docid.nunique()} 篇）"
          f" + 新 {len(ext)} 对（{ext.docid.nunique()} 篇）")

    def yes(x: object) -> bool:
        return str(x).strip().lower() in ("yes", "y", "true", "entail", "1")

    a = ext["A_label"].map(yes)
    b = ext["B_label"].map(yes)
    c = ext["third"].map(yes)
    new = pd.DataFrame({
        "cluster": ext["cluster"].astype(int),
        "facet": ext["facet"].astype(str),
        "docid": ext["docid"].astype(str),
        "gold": ext["new_gold"].map(yes),
        "votes": (a.astype(int) + b.astype(int) + c.astype(int)),
        "A_yes": a, "B_yes": b, "C_yes": c,
        "anchor_gold": ext["anchor_gold"].astype(bool),
    })[COLS]

    # 原始篇 + 新增篇（去重保护：新表不应含已判过的篇）
    overlap = set(zip(new.cluster, new.facet, new.docid)) & set(
        zip(base.cluster, base.facet, base.docid))
    if overlap:
        print(f"  ⚠️ 重叠 {len(overlap)} 对（新表里含已判过的篇）→ 以**新表为准**去重")
        new = new[~pd.Series(list(zip(new.cluster, new.facet, new.docid))).isin(overlap)
                  .to_numpy()]
    out = pd.concat([base[COLS], new], ignore_index=True)
    out.to_csv(DEV / "gold_final3.csv", index=False, encoding="utf-8-sig")
    print(f"  → `gold_final3.csv`：{len(out)} 对 ｜ {out.groupby(['cluster', 'facet']).ngroups} 组合"
          f" ｜ {out.docid.nunique()} 篇 ｜ 正例 {int(out.gold.sum())}")

    # ── 干扰项纯度：新篇里真为正例的比例 ──
    print("\n【★ 干扰项纯度】新篇被判为正例的比例（越低 = 越「纯干扰」）")
    print(f"  {'簇':>3}{'facet':<18}{'新篇数':>7}{'判为正例':>9}{'纯度(负例率)':>13}")
    rows = []
    for (cl, fa), g in new.groupby(["cluster", "facet"]):
        n, pos = len(g), int(g["gold"].sum())
        rows.append(dict(cluster=cl, facet=fa, n_new=n, n_pos=pos,
                         neg_rate=1 - pos / n if n else float("nan")))
        print(f"  {cl:>3}{fa:<18}{n:>7}{pos:>9}{1 - pos / n if n else 0:>13.1%}")
    s = pd.DataFrame(rows)
    print(f"\n  合计：新篇 **{len(new)}** 对 ｜ 判为正例 **{int(new.gold.sum())}**"
          f"（{new.gold.mean():.1%}）｜ 负例率 **{1 - new.gold.mean():.1%}**")
    print("  ⚠️ 若某 facet 纯度很低（正例占比高），说明那些「干扰项」其实是正例 → "
          "它改变的是**任务难度**，不是「干扰」，解读指标时要区分。")

    # ── 语料层面的正例率变化（20 篇 → 50 篇）──
    print(f"\n【语料正例率变化】（每组合：原 47 篇 → 扩到 50 篇口径）")
    print(f"  {'簇':>3}{'facet':<18}{'原正例/篇':>12}{'新正例/篇':>12}{'正例率变化':>12}")
    for (cl, fa) in sorted(set(zip(base.cluster, base.facet))):
        gb = base[(base.cluster == cl) & (base.facet == fa)]
        gn = new[(new.cluster == cl) & (new.facet == fa)]
        if not len(gn):
            continue
        r0 = gb.gold.mean() * 100
        r1 = pd.concat([gb.gold, gn.gold]).mean() * 100
        print(f"  {cl:>3}{fa:<18}{f'{int(gb.gold.sum())}/{len(gb)}':>12}"
              f"{f'{int(gn.gold.sum())}/{len(gn)}':>12}{r0:>7.1f}%→{r1:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
