"""现状体检：我们差在哪 + L5（查询分解）的 gate 该看什么信号。

回答两个问题：

  Q1 **各项指标是多少、哪些差** ——
     ① 与 LitSearch 论文**可比口径**对标（论文用 Avg-Broad **R@20** / Avg-Specific **R@5**，
        因为 Broad 类最多 20 篇相关、Specific 类最多 5 篇 —— 是公平性修正，不是策略差异）
     ② 把失败**分桶**：命中的是谁、漏在召回还是漏在排序

  Q2 **L5 什么时候启用** ——
     L5（查询分解）解决的是"**一个查询要同时满足多个条件，单条查询向量被稀释**"，
     所以它**只可能改善召回**（R@20/R@100），不可能改善排序（那是 rerank 的事）。
     因此 L5 的目标人群 = **"完全没召回"那一桶**。
     本脚本把 597 条按 best_rank 分桶，再看**零成本的候选 gate 信号**在桶间是否富集 ——
     富集才有资格当 gate；不富集就说明"该不该分解"无法事先预测，
     只能**全跑一遍、从增益分布反推**（成本约 $0.03，比设计 gate 的思考成本还低）。

用法：
    python retrieval/scripts/diagnose_buckets.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
EMB = DERIVED / "emb"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
CORPUS = HERE / "data" / "litsearch" / "corpus_clean" / "corpus.parquet"

# LitSearch 论文 Table 3 的 Avg. 列（仅用 title+abstract）
PAPER = {
    "BM25":                (39.9, 50.0),
    "GTR-T5-large":        (43.8, 39.6),
    "Instructor-XL":       (56.5, 52.3),
    "E5-large-v2":         (55.4, 56.2),
    "GritLM-7B":           (70.8, 74.8),
    "GPT-4o rerank(BM25)": (59.9, 68.0),
    "GPT-4o rerank(GritLM)": (75.3, 79.2),
}


def gold_rows_of(q: pd.DataFrame) -> list[set[int]]:
    ids = np.load(EMB / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    return [{id2row[int(g)] for g in row if int(g) in id2row} for row in q["corpusids"]]


def ranks_of(ranked: np.ndarray, gset: set[int], k: int) -> tuple[int, float]:
    """返回 (best_rank 1-based 或 0，top-k 内的召回率)"""
    pos = [int(np.where(ranked == g)[0][0]) + 1 for g in gset if (ranked == g).any()]
    best = min(pos, default=0)
    hit = sum(1 for x in pos if x <= k)
    return (best if best <= k else 0), (hit / len(gset) if gset else 0.0)


_SIGNALS = {
    "多条件词(and/both/as well as)": re.compile(r"\b(and|both|as well as|along with|combined with)\b", re.I),
    "含 and": re.compile(r"\band\b", re.I),
    "含 or": re.compile(r"\bor\b", re.I),
    "含逗号": re.compile(r","),
    "含缩写/专名": re.compile(r"\b[A-Z]{2,}\b"),
    "含数字": re.compile(r"\d"),
    "含 such as/like": re.compile(r"\b(such as|like|e\.g\.|including)\b", re.I),
    "含 compare/差异类": re.compile(r"\b(compare|comparison|difference|versus|vs\.?|better|trade-?off)\b", re.I),
}


def main() -> int:
    q = pd.read_parquet(QFILE)
    golds = gold_rows_of(q)
    n = len(q)
    truth = q["specificity"].to_numpy()
    print(f"查询 {n} 条 | Broad {(truth == 0).sum()} / Specific {(truth == 1).sum()}")

    z = np.load(RESULTS / "topk.npz")
    systems = [k for k in z.keys()]

    # ── ① 可比口径对标（Avg-Broad R@20 / Avg-Specific R@5）─────────
    def group_metrics(ranked_lists: list[np.ndarray]) -> tuple[float, float]:
        br20, sp5 = [], []
        for i in range(n):
            b, _ = ranks_of(ranked_lists[i], golds[i], 20)
            s, _ = ranks_of(ranked_lists[i], golds[i], 5)
            (br20 if truth[i] == 0 else sp5).append(1.0 if (b if truth[i] == 0 else s) else 0.0)
        return float(np.mean(br20)), float(np.mean(sp5))

    rows = []
    for s in systems:
        rl = [z[s][i] for i in range(n)]
        b20, s5 = group_metrics(rl)
        rows.append({"系统": f"ours: {s}", "Broad R@20": 100 * b20, "Specific R@5": 100 * s5})
    for name, (b, s) in PAPER.items():
        rows.append({"系统": f"paper: {name}", "Broad R@20": b, "Specific R@5": s})

    print("\n" + "=" * 84)
    print("① 与论文可比口径对标（Avg-Broad R@20 / Avg-Specific R@5，仅 title+abstract）")
    print("=" * 84)
    tb = pd.DataFrame(rows).sort_values("Broad R@20", ascending=False)
    print(tb.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # ── rerank 后的排名（在 hyb0.5 top-100 上重排）──────────────
    rr = np.load(RESULTS / "rerank_scores.npy")
    base = z["hyb0.5"] if "hyb0.5" in z else z[systems[0]]
    reranked = [base[i][np.argsort(-rr[i])] for i in range(n)]
    b20, s5 = group_metrics(reranked)
    print(f"\n  ours: hyb0.5 + rerank   Broad R@20 = {100 * b20:.1f}   "
          f"Specific R@5 = {100 * s5:.1f}   ← 当前最强")

    # ── ② 失败分桶（以当前最强的 rerank 结果为准）─────────────────
    print("\n" + "=" * 84)
    print("② 失败分桶：好文档到底卡在哪一环（hyb0.5 + rerank）")
    print("=" * 84)
    pool = []
    for i in range(n):
        pos = min([int(np.where(reranked[i] == g)[0][0]) + 1
                   for g in golds[i] if (reranked[i] == g).any()], default=0)
        pool.append(pos)
    pool = np.array(pool)
    buckets = [("① rank 1（直接对）", pool == 1),
               ("② rank 2~5（排得低）", (pool >= 2) & (pool <= 5)),
               ("③ rank 6~20", (pool >= 6) & (pool <= 20)),
               ("④ rank 21~100（深埋）", (pool >= 21) & (pool <= 100)),
               ("⑤ 完全没召回（rerank 上限内也没有）", pool == 0)]
    print(f"  {'桶':<36}{'n':>6}{'占比':>9}{'L5 可能帮到?':>16}")
    for name, m in buckets:
        help_flag = "★ 是（召回问题）" if name.startswith("⑤") else (
            "否（排序问题→rerank）" if name[0] in "②③④" else "—")
        print(f"  {name:<36}{int(m.sum()):>6}{100 * m.mean():>8.1f}%{help_flag:>16}")

    # ── ③ gate 候选信号在桶间的富集 ─────────────────────────────
    print("\n" + "=" * 84)
    print("③ L5 的 gate 候选信号：在『完全没召回』桶里是否富集？")
    print("    判据：某信号在该桶的占比明显高于全库基线 → 才有资格当 gate")
    print("=" * 84)
    miss = pool == 0
    sig_rows = []
    for name, pat in _SIGNALS.items():
        s = q["query"].str.contains(pat, regex=True).to_numpy()
        sig_rows.append({
            "信号": name,
            "全库占比": 100 * s.mean(),
            "命中桶占比": 100 * s[~miss].mean() if (~miss).sum() else 0,
            "漏检桶占比": 100 * s[miss].mean() if miss.sum() else 0,
            "富集倍数": (s[miss].mean() / s.mean()) if s.mean() > 0 else 0,
        })
    tb2 = pd.DataFrame(sig_rows).sort_values("富集倍数", ascending=False)
    print(tb2.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n  长度对比（词数）：")
    wl = q["query"].str.split().str.len().to_numpy()
    print(f"    全库中位 {np.median(wl):.0f} | 命中桶中位 {np.median(wl[~miss]):.0f} | "
          f"漏检桶中位 {np.median(wl[miss]):.0f}")

    print("\n[ 判读 ]")
    print("  · 若没有任何信号的富集倍数明显 > 1.2 → **gate 无法事先预测**，")
    print("    正确做法是：先在**全部查询**上跑 L5，再从'增益分布'反推 gate（而不是先猜标签）；")
    print("  · 若有信号富集明显 → 它可以当 gate 的候选，但仍需验证：")
    print("    『按该信号切开后，L5 的增益在两组间是否显著不同』；")
    print("  · 注意 L5 只可能改善**召回**（第 ⑤ 桶 18%），排序（②③④ 合计 45%）归 rerank。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
