"""路由值不值得做？先算**上界** —— 这是"有没有必要"的唯一客观判据。

一个残酷的事实：**路由的收益不可能超过 per-query oracle**。
oracle = 假设有一个完美路由器，对**每一个查询**都选到当下最优的配置。
如果连 oracle 相对"全局最优固定配置"的领先都很小（headroom 小），
那么**任何**路由器（不管分类多准）都不值得做 —— 可以直接收工。

本脚本算四件事（读 `results/per_query.parquet`，不改任何检索逻辑）：

  ① 全局最优固定配置 与 per-query oracle → **headroom（路由的理论上界）**
  ② 按 `specificity`（Broad/Specific）分组的 oracle → 按标签路由的上界
  ③ 5 折交叉验证的"真实路由"收益（组内选最优会过拟合，必须 CV）
  ④ 一致性：全局最优配置在多少查询上已是最优或并列最优

**设计要点：系统列表从数据里读**，所以将来加了 L3/L5/L4 的新配置后，
直接重跑本脚本即可回答"现在路由值不值得做"。

用法：
    python retrieval/scripts/oracle_routing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
PQ = RESULTS / "per_query.parquet"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
LLM_CACHE = DERIVED / "specificity_llm.json"

METRICS = ["recall@1", "recall@10", "mrr@10"]
SEED = 20260918


def cv_routed(df: pd.DataFrame, systems: list[str], labels: np.ndarray,
              metric: str, n_folds: int = 5) -> float:
    """诚实的路由收益：在训练折上为每组挑最优系统，在测试折上评。

    注意：即使这样仍是乐观的 —— 它假设**标签是免费且完全准确的**。
    真实部署里标签来自分类器（本数据集上 Broad F1 仅 0.526），还要再乘一次折扣。
    """
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, n_folds)
    wide = df.pivot(index="qidx", columns="system", values=metric)
    wide = wide.reindex(columns=[s for s in systems if s in wide.columns])
    vals = wide.to_numpy(dtype=float)
    pred = np.zeros(len(labels), dtype=float)
    for f in folds:
        te = f
        tr = np.setdiff1d(np.arange(len(labels)), te)
        for g in np.unique(labels):
            m_tr = tr[labels[tr] == g]
            if len(m_tr) == 0:
                continue
            per_sys = np.nanmean(vals[m_tr], axis=0)
            best = int(np.nanargmax(per_sys))
            m_te = te[labels[te] == g]
            pred[m_te] = vals[m_te, best]
    return float(np.nanmean(pred))


def main() -> int:
    for p in (PQ, QFILE):
        if not p.exists():
            print(f"缺 {p}（先跑 eval_retrieval.py --dump-topk）")
            return 2

    df = pd.read_parquet(PQ)
    systems = list(df["system"].unique())
    n = df["qidx"].nunique()
    q = pd.read_parquet(QFILE).head(n).reset_index(drop=True)
    truth = q["specificity"].to_numpy()
    pred = None
    if LLM_CACHE.exists():
        c = json.loads(LLM_CACHE.read_text(encoding="utf-8"))
        pred = np.array([c.get(str(s), -1) for s in q["query"]])
    print(f"题目 {n} 条 | 系统 {len(systems)} 个：{', '.join(systems)}\n")

    wide = {m: df.pivot(index="qidx", columns="system", values=m)
            .reindex(columns=systems).to_numpy(dtype=float) for m in METRICS}

    print("=" * 96)
    print("① 全局最优固定配置  vs  per-query oracle（路由的理论上界）")
    print("=" * 96)
    print(f"  {'指标':<12}{'全局最优固定':>16}{'该配置值':>10}{'oracle':>10}"
          f"{'headroom':>11}{'相对':>9}")
    headrooms = {}
    for m in METRICS:
        v = wide[m]
        per_sys = np.nanmean(v, axis=0)
        bi = int(np.nanargmax(per_sys))
        fixed = per_sys[bi]
        oracle = float(np.nanmean(np.nanmax(v, axis=1)))
        h = oracle - fixed
        headrooms[m] = h
        print(f"  {m:<12}{systems[bi]:>16}{fixed:>10.4f}{oracle:>10.4f}"
              f"{h:>+11.4f}{100 * h / max(fixed, 1e-9):>8.1f}%")

    print("\n" + "=" * 96)
    print("② 按标签分组的 oracle（= 完美分类器能达到上界）")
    print("=" * 96)
    labelsets = [("真值 specificity", truth)]
    if pred is not None:
        labelsets.append(("LLM 预测", pred))
        labelsets.append(("预测(仅 Broad 判对)", np.where(pred >= 0, pred, -1)))
    for name, lab in labelsets:
        for m in METRICS:
            v = wide[m]
            per_sys = np.nanmean(v, axis=0)
            glob = per_sys[int(np.nanargmax(per_sys))]
            tot, cnt = 0.0, 0
            picks = {}
            for g in np.unique(lab):
                sel = lab == g
                if sel.sum() == 0 or g < 0:
                    continue
                ps = np.nanmean(v[sel], axis=0)
                bi = int(np.nanargmax(ps))
                picks[int(g)] = (systems[bi], ps[bi], int(sel.sum()))
                tot += ps[bi] * sel.sum()
                cnt += sel.sum()
            if cnt == 0:
                continue
            routed = tot / cnt
            gp = "  ".join(f"{k}→{s}({vv:.4f},n={nn})" for k, (s, vv, nn) in sorted(picks.items()))
            print(f"  [{name}] {m:<11} 全局 {glob:.4f} → 分组 {routed:.4f} "
                  f"({routed - glob:+.4f})   {gp}")

    print("\n" + "=" * 96)
    print("③ 5 折 CV 的诚实路由收益（组内选最优会过拟合；且仍假设标签免费且完全准确）")
    print("=" * 96)
    for name, lab in labelsets:
        if lab.min() < 0:
            print(f"  [{name}] 含无效标签，跳过改为仅用有效部分")
            lab = np.where(lab >= 0, lab, -1)
        for m in METRICS:
            v = wide[m]
            per_sys = np.nanmean(v, axis=0)
            glob = per_sys[int(np.nanargmax(per_sys))]
            r = cv_routed(df, systems, lab, m)
            print(f"  [{name}] {m:<11} 全局 {glob:.4f} → CV路由 {r:.4f} ({r - glob:+.4f})")

    print("\n" + "=" * 96)
    print("④ 一致性：全局最优配置在多少查询上已是最优或并列最优（= 路由无处发力的比例）")
    print("=" * 96)
    for m in METRICS:
        v = wide[m]
        per_sys = np.nanmean(v, axis=0)
        bi = int(np.nanargmax(per_sys))
        best_val = np.nanmax(v, axis=1)
        tie = np.isclose(v[:, bi], best_val)
        print(f"  {m:<12} 全局最优={systems[bi]:<8} 已最优或并列的查询占比 "
              f"{100 * tie.mean():5.1f}%   （路由最多只能在剩下的 "
              f"{100 * (1 - tie.mean()):.1f}% 上发挥作用）")

    print("\n" + "=" * 96)
    print("⑤ headroom 是'可用的结构'还是'同族配置间的噪声'？（决定 ① 的 9pt 能不能用）")
    print("=" * 96)
    print("  核心问题：oracle 之所以高，是因为存在**质性不同**的策略，还是只是")
    print("  同一个管道里 α 抖动（hyb0.3/0.5/0.7/0.9 本质是同一件事）？")
    v1 = wide["recall@1"]
    subsets = {
        "全部 6 个": systems,
        "hyb 家族 4 个（同一管道不同 α）": [s for s in systems if s.startswith("hyb")],
        "仅 {bm25, dense}（质性不同）": [s for s in ("bm25", "dense") if s in systems],
        "仅 hyb0.7 一个（固定）": ["hyb0.7"] if "hyb0.7" in systems else [systems[-1]],
    }
    base = systems[int(np.nanargmax(np.nanmean(v1, axis=0)))]
    base_v = np.nanmean(v1[:, systems.index(base)])
    for name, sub in subsets.items():
        idx = [systems.index(s) for s in sub if s in systems]
        if not idx:
            continue
        o = float(np.nanmean(np.nanmax(v1[:, idx], axis=1)))
        print(f"  {name:<34} oracle={o:.4f}  (相对固定 {base} {o - base_v:+.4f})")

    fam = [s for s in systems if s.startswith("hyb")]
    if len(fam) > 1:
        fi = [systems.index(s) for s in fam]
        sub = v1[:, fi]
        agree = float(np.mean(np.nanstd(sub, axis=1) == 0))
        # 只在"存在严格赢家"的查询上统计谁赢（全错/全对的全并列查询要剔除，
        # 否则 argmax 恒返回第 0 列，统计会完全失真）
        mx = np.nanmax(sub, axis=1)
        n_ties = np.isclose(sub, mx[:, None]).sum(axis=1)
        uniq = n_ties == 1
        win = np.nanargmax(sub, axis=1)
        dist = {fam[k]: int(((win == k) & uniq).sum()) for k in range(len(fam))}
        print(f"\n  hyb 家族内部：{len(fam)} 个配置结论完全一致的查询占比 {100 * agree:.1f}%")
        print(f"  存在**唯一**胜者的查询数 {int(uniq.sum())}（其余为全并列），"
              f"其中胜者分布：{dist}")
        print("  → 若这些'唯一胜者'在各 α 间大致均匀轮换，说明差异是**采样噪声**，")
        print("    不是'某类查询需要某个 α'的可学结构 → 路由无从发力。")

    print("\n[ 判读 ]")
    print("  · headroom 小（比如 R@1 < 2pt）→ **任何路由器都不值得做**，收工；")
    print("  · headroom 大（> 5pt）→ 有空间，但还要看 ③：CV 路由是否真拿到收益；")
    print("  · ③ 仍假设标签免费且准确 —— 若 ③ 只拿到 headroom 的一小部分，说明")
    print("    收益被'组内最优本身的噪声'吃掉了，真实部署（标签也不准）只会更差；")
    print("  · **⑤ 是关键闸门**：若 headroom 主要来自'同一管道的参数抖动'，")
    print("    那么即使把分类器做到 100% 准，也拿不到那 9pt —— 因为抖动**不可预测**。")
    print("    路由只有在候选策略之间存在**质性差异**时才有意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
