"""**打分融合 vs 排序融合**：聚焦头对头（50 篇口径，零 LLM）

## 要回答的问题
· 分数级融合（`normsum`：归一化后加权和）与名次级融合（`rrf`：`Σ w/(60+rank)`）谁更好？
· 若接近，**谁更稳**？（← 这才是决策依据：更少假设、更少超参 = 更该选）

## 三个比较维度
1. **同权对照**：同一 ρ、同一层级、同一 k 下直接比 `setF1`
2. **配对自举**：对 28 题重采样，报 95% CI 与 p
3. ★ **稳健性（关键）**：
   · 对每个方法族算**跨权重极差**（ρ 取遍后 max−min）→ 该族对超参多敏感
   · 对分数融合再算**跨归一化极差**（z-score vs min-max）→ 该族对"归一化选择"多敏感
   → 若"归一化选择"造成的摆动 ≥ 两族之间的差异，则 RRF 是更稳的选择（**更少假设**）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_score_vs_rank.py --corpus 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RNG = np.random.default_rng(20260930)
NBOOT = 4000
RANK, SCORE_Z, SCORE_MM = "rrf", "normsum_z", "normsum_mm"
FAM = {RANK: "排序融合", SCORE_Z: "打分融合(z)", SCORE_MM: "打分融合(mm)"}


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str = "setF1"):
    keys = ["cluster", "facet", "qmode", "k"]
    x = a.set_index(keys)[col].to_frame("a").join(
        b.set_index(keys)[col].to_frame("b"), how="inner")
    if not len(x):
        return None
    d = (x["a"] - x["b"]).to_numpy()
    idx = RNG.integers(0, len(d), size=(NBOOT, len(d)))
    m = d[idx].mean(axis=1)
    return dict(mu=float(d.mean()), lo=float(np.percentile(m, 2.5)),
                hi=float(np.percentile(m, 97.5)), p=float((m <= 0).mean()), n=len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--k-main", type=int, default=13)
    args = ap.parse_args()
    dall = pd.read_csv(RES / f"R2_FUSION_n{args.corpus}.csv")
    orc = dall[dall.method == "oracle2"]          # ⚠️ 上界行要单独留（下面会过滤掉）
    d = dall[dall.method != "oracle2"]
    q = d[(d.qmode == "Q_prod") & (d.k == args.k_main)]
    print("=" * 118)
    print(f"【打分融合 vs 排序融合】{args.k_main} 篇交付 ｜ 库 {int(d.n_docs.max())} 篇/簇"
          f" ｜ 均 gold {d.n_gold.mean():.2f} ｜ 口径 Q_prod ｜ 题数 "
          f"{q.groupby(['level','method','cfg']).ngroups and len(q)//42}")
    print(f"  方法：`rrf` = Σ w/(60+rank)（名次）｜ `normsum_z` / `normsum_mm` = 归一化后加权和（分数）")

    # ── 表 1：同权对照（层级 × 权重）──
    print(f"\n【表 1｜同权同层级直接对照 @ k={args.k_main}】setF1（每格同一 ρ）")
    cfgs = [c for c in ["dense", "4:1", "2:1", "1:1", "1:2", "1:4", "bm25"]]
    print(f"  {'层级':<7}{'ρ':<7}｜{FAM[RANK]:>12}{FAM[SCORE_Z]:>14}{FAM[SCORE_MM]:>14}"
          f"｜{'排序−打分(z)':>13}{'排序−打分(mm)':>14}")
    for lvl in ("doc", "chunk"):
        for cfg in cfgs:
            s = q[(q.level == lvl) & (q.cfg == cfg)]
            v = {m: s[s.method == m].setF1.mean() for m in FAM}
            if any(pd.isna(x) for x in v.values()):
                continue
            print(f"  {lvl:<7}{cfg:<7}｜{v[RANK]:>12.3f}{v[SCORE_Z]:>14.3f}"
                  f"{v[SCORE_MM]:>14.3f}｜{v[RANK] - v[SCORE_Z]:>+13.3f}"
                  f"{v[RANK] - v[SCORE_MM]:>+14.3f}")

    # ── 表 2：各族最优 + 配对自举 ──
    print(f"\n【表 2｜各族最优配置 + 配对自举（{NBOOT} 次）】")
    best: dict[str, tuple[str, str]] = {}
    for m in FAM:
        s = q[q.method == m].groupby(["level", "cfg"]).setF1.mean().sort_values(
            ascending=False)
        if len(s):
            best[m] = s.index[0]
            print(f"  {FAM[m]:<14}最优 = {s.index[0][0]} / ρ={s.index[0][1]:<6}"
                  f"setF1 **{s.iloc[0]:.3f}**")
    # 取"同等条件是同一个 level"的最优，做干净配对
    for lvl in ("doc", "chunk"):
        for m in (SCORE_Z, SCORE_MM):
            lm = q[(q.level == lvl) & (q.method == m)].groupby("cfg").setF1.mean()
            if not len(lm):
                continue
            ca = q[(q.level == lvl) & (q.method == RANK) & (q.cfg == lm.index[0])]
            cb = q[(q.level == lvl) & (q.method == m) & (q.cfg == lm.index[0])]
            r = paired(ca, cb)
            if r:
                print(f"  [{lvl}/{lm.index[0]}] {FAM[RANK]} − {FAM[m]}："
                      f"ΔsetF1 {r['mu']:+.3f}  95%CI [{r['lo']:+.3f}, {r['hi']:+.3f}]"
                      f"  p(≤0)={r['p']:.3f}  n={r['n']}")

    # ★ 各族**最优**之间的配对（含选择偏差，故只作参考；同权配对才是干净对照）
    print(f"\n  【各族最优之间】⚠️ 含选择偏差（跨 2 层级 × 7 权重挑最大）")
    for m in (SCORE_Z, SCORE_MM):
        lv, cf = best[m]
        A = d[(d.qmode == "Q_prod") & (d.k == args.k_main) & (d.method == RANK)
              & (d.level == best[RANK][0]) & (d.cfg == best[RANK][1])]
        B = d[(d.qmode == "Q_prod") & (d.k == args.k_main) & (d.method == m)
              & (d.level == lv) & (d.cfg == cf)]
        r = paired(A, B)
        if r:
            print(f"  排序最优({best[RANK][0]}/{best[RANK][1]}) − {FAM[m]}最优({lv}/{cf})："
                  f"Δ {r['mu']:+.3f}  95%CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                  f"p(≤0)={r['p']:.3f}  n={r['n']}")

    # ★★ 重要性对比：**权重** vs **方法**
    # ⚠️ 必须用**配置级均值**（groupby mean 再取 max），不能用逐题 `.max()`
    #    （后者对三族都返回 1.0 → 极差恒为 0）
    fam_best = {m: q[q.method == m].groupby(["level", "cfg"]).setF1.mean().max()
                for m in FAM}
    spread_method = max(fam_best.values()) - min(fam_best.values())
    spread_w = {}
    for m in FAM:
        for lvl in ("doc", "chunk"):
            by = q[(q.method == m) & (q.level == lvl)].groupby("cfg").setF1.mean()
            if len(by):
                # 排除退化的纯单通道端点（dense / bm25），只看"权重比例"的影响
                mid = by[[c for c in by.index if c not in ("dense", "bm25")]]
                if len(mid) > 1:
                    spread_w[f"{m}/{lvl}"] = float(mid.max() - mid.min())
    print(f"\n  ★★ **重要性对比**：方法族之间最优差 **{spread_method:.3f}**"
          f" ｜ 仅调权重比例（排除纯单通道端点）造成的差 **"
          f"{min(spread_w.values()):.3f}~{max(spread_w.values()):.3f}**")
    print(f"     → 权重的影响是融合方式的 **约 "
          f"{np.mean(list(spread_w.values())) / max(spread_method, 1e-6):.0f} 倍**")

    # ── 表 3：★ 稳健性（决策依据）──
    print(f"\n【表 3｜★ 稳健性】对「超参选择」的敏感度 @ k={args.k_main}（越小越稳）")
    print(f"  {'层级':<7}{'方法族':<16}{'跨权重极差':>12}{'跨归一化极差':>14}"
          f"{'最优 setF1':>12}{'最差 setF1':>12}")
    for lvl in ("doc", "chunk"):
        for m in (RANK, SCORE_Z, SCORE_MM):
            s = q[(q.level == lvl) & (q.method == m)]
            if not len(s):
                continue
            by = s.groupby("cfg").setF1.mean()
            print(f"  {lvl:<7}{FAM[m]:<16}{by.max() - by.min():>12.3f}"
                  f"{'—':>14}{by.max():>12.3f}{by.min():>12.3f}")
        # 归一化极差（同 ρ 下 z vs mm）
        a = q[(q.level == lvl) & (q.method == SCORE_Z)].groupby("cfg").setF1.mean()
        b = q[(q.level == lvl) & (q.method == SCORE_MM)].groupby("cfg").setF1.mean()
        if len(a) and len(b):
            diff = (a - b).abs()
            print(f"  {lvl:<7}{'  └ 归一化之差':<16}{'—':>12}"
                  f"{diff.max():>14.3f}   中位 {diff.median():.3f}")

    # ── 表 4：k 曲线（各族最优 vs 并集上界）──
    print(f"\n【表 4｜k 曲线】各族最优（跨 ρ 取最优）vs 两通道并集上界")
    print(f"  {'k':>3}{'k/库':>7}{'排序setF1':>11}{'打分(z)':>10}{'打分(mm)':>10}"
          f"{'排序StRecall':>13}{'并集上界':>10}{'留白':>8}")
    for k in sorted(d.k.unique()):
        sub = d[(d.qmode == "Q_prod") & (d.k == k)]
        _o = orc[(orc.qmode == "Q_prod") & (orc.k == k)]
        o = _o.setF1.mean() if len(_o) else float("nan")
        row, kr = {}, None
        for m in (RANK, SCORE_Z, SCORE_MM):
            s = sub[sub.method == m].groupby(["level", "cfg"]).setF1.mean()
            row[m] = s.max() if len(s) else float("nan")
        sr = sub[sub.method == RANK].groupby(["level", "cfg"]).StRecall.mean()
        kr = sr.max() if len(sr) else float("nan")
        print(f"  {k:>3}{k / d.n_docs.max():>7.2f}{row[RANK]:>11.3f}"
              f"{row[SCORE_Z]:>10.3f}{row[SCORE_MM]:>10.3f}{kr:>13.3f}"
              f"{o:>10.3f}{o - row[RANK]:>+8.3f}")

    print(f"\n【读法】表 3 的「跨归一化极差」= 分数融合**只因换归一化方式**产生的摆动；"
          f"把它与「两族最优之差」比较 → 若摆动相当或更大，"
          f"则分数融合的高分是**选出来的**，而 RRF 的高分是**稳定得到的**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
