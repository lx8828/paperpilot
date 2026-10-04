"""**新检索指标汇总**：从 `R2_RETR_{new,anchor}_3arms.csv` 出表（主表 / 逐 facet / 新旧对照）。

⚠️ 两口径**不是单变量对照**：新口径 =「三判官多数票真值 + 28 组合题集 + v2 操作化题面」整体；
   旧口径 =「锚点正则真值 + 35 组合 + v1 简述题面」整体。**题集/真值/题面三者同时变**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEW = RES / "R2_RETR_new_3arms.csv"
ANC = RES / "R2_RETR_anchor_3arms.csv"
SEL = json.loads((DEV / "facets_v2_selected.json").read_text(encoding="utf-8"))
CORE = set(SEL["core_facets"])
COLS = [("MRecall@5", "MRecall@5"), ("MRecall@10", "MRecall@10"),
        ("StRecall@5", "StRecall@5"), ("StRecall@10", "StRecall@10"),
        ("StRecall@20", "StRecall@20"),
        ("setP@10", "集合P@10"), ("setF1@10", "集合F1@10"),
        ("aNDCG@10", "α-nDCG@10"), ("ev_recall_c12k", "证据召回@12k")]


def table(d: pd.DataFrame, arms: list[str], title: str) -> None:
    m = d[d.sorter == "B mq_max"]
    print("\n" + "=" * 122)
    print(f"{title}（排序臂 `B mq_max`，多查询取 max）")
    print(f"  {'臂':<13}{'题数':>5}{'块数':>7}" + "".join(f"{lab:>12}" for _, lab in COLS))
    for arm in arms:
        s = m[m.arm == arm]
        if not len(s):
            continue
        print(f"  {arm:<13}{s.groupby(['cluster', 'facet']).ngroups:>5}{s['n_chunks'].mean():>7.0f}"
              + "".join(f"{s[c].mean():>12.3f}" for c, _ in COLS))


def main() -> int:
    dn = pd.read_csv(NEW)
    da = pd.read_csv(ANC)
    arms = ["A_ls_fix", "B_pdf_fix", "C_pdf_prod"]

    table(dn, arms, "【★ 新口径｜新检索指标】三判官多数票真值 + 28 组合 + v2 题面")
    table(da, arms, "【旧口径｜对照】锚点正则真值 + 35 组合 + v1 题面")

    print("\n" + "=" * 122)
    print("【★ 结论：生产切块（C_pdf_prod）vs 旧基线（A_ls_fix）—— 两口径下符号相反】")
    print(f"  {'口径':<22}{'StRecall@10':>13}{'集合F1@10':>12}{'证据召回@12k':>14}")
    for nm, d in (("新（多数票·28 题）", dn), ("旧（锚点·35 题）", da)):
        m = d[d.sorter == "B mq_max"]
        c, a = m[m.arm == "C_pdf_prod"], m[m.arm == "A_ls_fix"]
        print(f"  {nm:<22}{c['StRecall@10'].mean() - a['StRecall@10'].mean():>+13.3f}"
              f"{c['setF1@10'].mean() - a['setF1@10'].mean():>+12.3f}"
              f"{c['ev_recall_c12k'].mean() - a['ev_recall_c12k'].mean():>+14.3f}")
    print("  ⚠️ 两口径**同时**换了真值、题集、题面 → 符号反转**不能**归因到单一项（需隔离实验）")

    m = dn[(dn.sorter == "B mq_max") & (dn.arm == "C_pdf_prod")]
    print("\n" + "=" * 122)
    print("【★ 新口径逐 facet（生产切块）】按 集合F1@10 升序")
    print(f"  {'簇':>3} {'facet':<17}{'档':>5}{'gold':>5}{'MRec@10':>9}{'StRec@10':>10}"
          f"{'集合P@10':>10}{'集合F1@10':>11}{'αNDCG@10':>10}")
    for _, r in m.sort_values("setF1@10").iterrows():
        tier = "核心" if r.facet in CORE else "扩展"
        print(f"  {int(r.cluster):>3} {str(r.facet):<17}{tier:>5}{int(r.n_gold):>5}"
              f"{r['MRecall@10']:>9.3f}{r['StRecall@10']:>10.3f}"
              f"{r['setP@10']:>10.3f}{r['setF1@10']:>11.3f}{r['aNDCG@10']:>10.3f}")

    print("\n  【按档聚合】")
    for tier, sub in (("核心集（≥2 簇）", m[m.facet.isin(CORE)]),
                      ("扩展集（1 簇）", m[~m.facet.isin(CORE)]),
                      ("全部 28 题", m)):
        print(f"    {tier:<16}{len(sub):>3} 题 ｜ StRecall@10 {sub['StRecall@10'].mean():.3f}"
              f" ｜ 集合F1@10 {sub['setF1@10'].mean():.3f}"
              f" ｜ MRecall@10 {sub['MRecall@10'].mean():.3f}"
              f" ｜ 均 gold {sub.n_gold.mean():.2f}")

    print("\n  【单查询 `A base` vs 多查询 `B mq_max`（生产切块·新口径）】")
    mn = dn[dn.arm == "C_pdf_prod"]
    for s_, lab in ((mn[mn.sorter == "A base"], "A base"), (mn[mn.sorter == "B mq_max"], "B mq_max")):
        print(f"    {lab:<10}StRecall@10 {s_['StRecall@10'].mean():.3f}"
              f" ｜ 集合F1@10 {s_['setF1@10'].mean():.3f}"
              f" ｜ 证据召回@12k {s_['ev_recall_c12k'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
