"""**汇总所有审计后的"干净数字"**（只取经审计确认无泄露的口径）"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES = Path(__file__).resolve().parents[1] / "results"

print("=" * 108)
print("【1】R2 QAMPARI（LoFT 官方，100 题）｜ 运行 = exh_k40_k40（官方 CLI 逐位复现）")
o = pd.read_csv(RES / "R2_QAMPARI_OFFICIAL_ALLRUNS.csv")
r = o[o["运行"] == "exh_k40_k40"].iloc[0]
p = pd.read_csv(RES / "R2_QAMPARI_PRECISION_ALLRUNS.csv")
pr = p[p["运行"] == "exh_k40_k40"].iloc[0]
f = pd.read_csv(RES / "R2_QAMPARI_AUDIT_FINAL.csv")
fa = f[f["运行"] == "exh_k40_k40"].iloc[0]
print(f"  {'指标':<34}{'值':>10}   口径")
for k, v, note in [
    ("subspan_em（LoFT 主指标）", r["官方subspan_em"], "官方 evaluation/rag.py"),
    ("em（集合完全一致）", r["官方em"], "官方"),
    ("coverage（官方报道·非空行）", r["官方coverage"], "官方；分母剔掉空预测"),
    ("coverage（全行口径·更保守）", fa["coverage_全行"], "空预测记 0 → 跨运行可比"),
    ("precision（ALCE 官方算术）", r["ALCE_P"], "HELMET/ALCE eval_alce.py 逐字移植"),
    ("recall（ALCE 官方算术）", r["ALCE_R"], "同上"),
    ("F1（ALCE 官方算术）", r["ALCE_F1"], "同上"),
    ("recall@5（ALCE 官方）", pr["ALCE_R5"], "同上"),
    ("平均预测答案数 num_preds", pr["预测均"], "官方 num_preds 同口径"),
    ("gold 平均答案数", pr["gold均"], "数据集事实（每题 5~6）"),
    ("预测/gold 比值", pr["比值"], ">1.2 才算「列多了」"),
]:
    print(f"  {k:<34}{v:>10.4f}   {note}")
print(f"  空预测题数 {int(fa['空预测题'])}/100 ｜ 无 recall 灌水（P {r['ALCE_P']:.3f} ≥ R {r['ALCE_R']:.3f}）")

print("\n" + "=" * 108)
print("【2】R2 集合型检索（用户自带 20 篇语料，35 个真值）｜ **干净打分 = dense_zh**")
L = pd.read_csv(RES / "R2_STD_L1_paper.csv")
print(f"  列：{list(L.columns)[:12]}")
dz = L[L["scorer"] == "dense_zh"]
for k in (5, 10, 20):
    t = dz[dz["k"] == k]
    if len(t):
        print(f"  k={k:<3} MRecall {t['MRecall'].mean():.3f} ｜ StRecall {t['StRecall'].mean():.3f}"
              f" ｜ 集合P {t['setP'].mean():.3f} ｜ 集合F1 {t['setF1'].mean():.3f}"
              f" ｜ α-nDCG {t['aNDCG'].mean():.3f}")
RND = L[L["scorer"] == "random"]
ORC = L[L["scorer"] == "oracle"]
for nm, D in (("随机基线", RND), ("oracle 上界", ORC)):
    t10 = D[D["k"] == 10]
    print(f"  对照 {nm:<10} k=10 MRecall {t10['MRecall'].mean():.3f} StRecall {t10['StRecall'].mean():.3f}")
AUC = pd.read_csv(RES / "R2_LEAK_AUC.csv")
auc = AUC.groupby("variant")["auc"].mean()
print(f"\n  篇级 AUC（区分度）：zh 干净 {auc['zh  (干净)']:.3f} ｜ "
      f"随机 0.500 ｜ 污染版 zh+anchor {auc['zh+anchor (现行·污染)']:.3f}（**不可引用**）")

print("\n" + "=" * 108)
print("【3】第 2 层：等块预算（**干净打分 zh**）")
L2 = pd.read_csv(RES / "R2_LEAK_L2.csv")
z = L2[L2["scorer"] == "zh 干净"]
print(f"  {'预算B':>6}{'策略':<14}{'涉及篇':>7}{'覆盖率':>8}{'证据召回':>9}")
for b in sorted(z["budget"].unique()):
    for st in ("global_topB", "round_robin"):
        t = z[(z["budget"] == b) & (z["strategy"] == st)]
        print(f"  {b:>6}{st:<14}{t['n_used'].mean():>7.1f}{t['cover'].mean():>8.3f}"
              f"{t['ev_recall'].mean():>9.3f}")

print("\n" + "=" * 108)
print("【4】生产 QA 链路（`ok_strict` = 仅答案判分 = 干净口径）")
pg = pd.read_csv(RES / "R2_PROD_JUDGE_GAP.csv")
for g, gr in pg.groupby("group"):
    print(f"  {g:<12} 运行 {len(gr):>3} 个 ｜ 题数 {gr['n'].mean():>5.1f}/run ｜ "
          f"**ok_strict（干净）{gr['ok_strict'].mean():.3f}** ｜ "
          f"ok（含自引证据）{gr['ok'].mean():.3f} ｜ 放水 {gr['gap'].mean():+.3f}")

print("\n" + "=" * 108)
print("【5】泄露审计结论一览（哪些数字能用 / 哪些不能用）")
print(f"  {'数字':<40}{'状态':<10}说明")
for nm, st, note in [
    ("dense_zh_en / dense_en / bm25_anchor 的任何分数", "❌ 弃用", "63% 锚点词 = gold 正则词；+0.074 AUC 全来自泄露"),
    ("『加英文锚点词 +4pt / XLING 值 +4pt』", "❌ 撤回", "纯泄露产物"),
    ("oracle 阈值 / oracle 排序", "⚠️ 仅上界", "用了 gold，不可部署"),
    ("dense_zh / bm25_para", "✅ 可用", "与 gold 正则零/极低重合"),
    ("LoFT 官方 em/coverage/subspan_em", "✅ 可用", "官方 CLI 逐位复现；但 coverage 需带空预测数一起报"),
    ("ALCE precision / num_preds", "✅ 可用", "官方算术；空预测记 0、分母恒定"),
    ("生产 ok", "⚠️ 偏高", "含模型自引证据通道"),
    ("生产 ok_strict", "✅ 可用", "仅答案判分"),
    ("人工出题的题面", "✅ 大体干净", "gold 引文与题面 token 重合 0.000；8.3% 题面含专名锚点（多是指代）"),
]:
    print(f"  {nm:<40}{st:<10}{note}")
