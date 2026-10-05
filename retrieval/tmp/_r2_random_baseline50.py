"""50 篇语料下的**随机 / oracle 参照**（纯排序计算，零 LLM、零编码）。

用途：给 50 篇口径的检索臂一个相对水位（vs 随机、vs 上界），
结果写进 `docs/RESULTS_SUMMARY.md` §2.2。

数据来源（全部已落盘，**不重跑检索、不调 LLM**）
----------------------------------------------
- 真值：`retrieval/data/r2dev/gold_final3.csv`（1,400 对 ｜ 28 组合 ｜ 50 篇/簇）
- 本系统名次：`retrieval/results/R2_RETR_n50_gnew_cnew_2arms.csv`
  （arm `C_pdf_prod` = 生产切块，sorter `B mq_max` = 多查询；与 `R2_CORPUS_50` §4.1 同口径）
- 指标函数：逐字复用 `retrieval/tmp/_r2_std_metrics.py`（mrecall / strecall / setpf / alpha_ndcg）

口径说明
--------
- 随机 = 对**同一簇的 50 篇**做 200 次等概率排列，取指标均值（seed=0，与旧脚本一致）。
- oracle = 全部正例置顶（上界）。
- ★ `MRecall` 在 50 篇下退化（gold 均 13 > k=10），按 `R2_CORPUS_50` §6.7 的口径**不报**。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "retrieval" / "data" / "r2dev"
RES = ROOT / "retrieval" / "results"

# ── 指标函数：逐字复用，保证与 3×20 时代的数字同口径 ──
_spec = importlib.util.spec_from_file_location(
    "stdm", ROOT / "retrieval" / "tmp" / "_r2_std_metrics.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
strecall, setpf, alpha_ndcg = _m.strecall, _m.setpf, _m.alpha_ndcg

KEYS = {5: ("StRecall@5", "setP@5", "setF1@5", "aNDCG@5"),
        10: ("StRecall@10", "setP@10", "setF1@10", "aNDCG@10"),
        20: ("StRecall@20", "setP@20", "setF1@20", "aNDCG@20")}

gold_df = pd.read_csv(DEV / "gold_final3.csv")
gold_df["gold"] = gold_df["gold"].astype(bool)

retr = pd.read_csv(RES / "R2_RETR_n50_gnew_cnew_2arms.csv")
ours = retr[(retr.arm == "C_pdf_prod") & (retr.sorter == "B mq_max")].set_index(
    ["cluster", "facet"])

print("  语料 = 3 簇 x 50 篇 ｜ 组合 = 28 ｜ 真值 = gold_final3.csv")
print(f"  本系统臂 = C_pdf_prod / B mq_max（{len(ours)} 个组合）")
print()

rng = np.random.default_rng(0)
rand_rows, orac_rows = [], []
for (c, f), sub in gold_df.groupby(["cluster", "facet"]):
    docs = sorted(sub["docid"].unique().tolist())
    gold = set(sub[sub["gold"]]["docid"])
    if not gold or len(gold) == len(docs):
        continue
    perms = [rng.permutation(docs).tolist() for _ in range(200)]
    orac = sorted(docs, key=lambda d: (d not in gold, d))
    for k in KEYS:
        pv = [setpf(x, gold, k) for x in perms]
        rand_rows.append(dict(
            cluster=c, facet=f, k=k, n_gold=len(gold),
            st=float(np.mean([strecall(x, gold, k) for x in perms])),
            p=float(np.mean([x[0] for x in pv])),
            f1=float(np.mean([x[2] for x in pv])),
            an=float(np.mean([alpha_ndcg(x, gold, k) for x in perms]))))
        po, _, fo = setpf(orac, gold, k)
        orac_rows.append(dict(cluster=c, facet=f, k=k, n_gold=len(gold),
                              st=strecall(orac, gold, k), p=po, f1=fo,
                              an=alpha_ndcg(orac, gold, k)))

rand = pd.DataFrame(rand_rows).groupby("k")[["st", "p", "f1", "an"]].mean()
orac = pd.DataFrame(orac_rows).groupby("k")[["st", "p", "f1", "an"]].mean()

print(f"  {'k':>3}  {'臂':<14}{'StRecall':>9}{'集合P':>8}{'集合F1':>8}{'α-nDCG':>8}"
      f"{'  相对随机(StRecall)':>22}")
print("  " + "-" * 78)
for k, cols in KEYS.items():
    sc, pc, fc, ac = cols
    o = ours[[sc, pc, fc, ac]].mean()
    r = rand.loc[k]
    rows = (("本系统（生产切块）", o[sc], o[pc], o[fc], o[ac]),
            ("随机（200 次排列）", r["st"], r["p"], r["f1"], r["an"]),
            ("oracle（上界）", orac.loc[k, "st"], orac.loc[k, "p"],
             orac.loc[k, "f1"], orac.loc[k, "an"]))
    for i, (nm, a, b, cc, dd) in enumerate(rows):
        ratio = f"{a / r['st']:.2f}x" if (nm.startswith("本系统") and r["st"]) else ""
        print(f"  {k:>3}  {nm:<14}{a:>9.3f}{b:>8.3f}{cc:>8.3f}{dd:>8.3f}"
              f"{ratio:>22}")
    print("  " + "-" * 78)

print()
print("  逐组合明细（k=10，本系统 vs 随机，按随机基线差值排序）")
base = pd.DataFrame(rand_rows)
b10 = base[base.k == 10].set_index(["cluster", "facet"])
ours_d = pd.DataFrame(index=b10.index)
ours_d["ours_st"] = ours["StRecall@10"]
ours_d["rand_st"] = b10["st"]
ours_d["n_gold"] = b10["n_gold"]
ours_d = ours_d.sort_values("rand_st", ascending=False)
for (c, f), x in ours_d.iterrows():
    print(f"    簇{int(c)} {f:<16} gold={int(x['n_gold']):>2}  "
          f"随机={x['rand_st']:.3f}  本系统={x['ours_st']:.3f}")
