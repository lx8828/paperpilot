"""路由的**前置条件验证**：Broad / Specific 两组的"最优检索策略"是否真的不同？

L1 路由的价值 = P(分类对) × (扩展策略收益 − 不扩展收益)。
但如果**两组的最优策略本来就一样**，那分流就没有必要 —— 分类再准也白搭。

本脚本不改任何检索逻辑，只把已有的逐题结果（`results/per_query.parquet`）
按两种分组切开重新汇总：

    分组依据 1（真值）  : LitSearch 的 `specificity`（Broad=0 / Specific=1）
    分组依据 2（预测）  : LLM 零样本预测（`derived/specificity_llm.json` 缓存）

观察三件事：
  ① 各系统在两组上的指标水平（Broad 组是否整体更难？）
  ② **各组内部的"最优配置"是否不同**（这才是路由的正当理由）
  ③ 各组内 hybrid 的增益是否有系统差异（扩展该给谁用）

用法：
    python retrieval/scripts/stratify_queries.py
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
LLM_CACHE = DERIVED / "specificity_llm.json"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"

SYSTEMS = ["bm25", "dense", "hyb0.3", "hyb0.5", "hyb0.7", "hyb0.9"]
METRICS = ["recall@1", "recall@10", "mrr@10"]


def main() -> int:
    for p in (PQ, QFILE):
        if not p.exists():
            print(f"缺 {p}（先跑 eval_retrieval.py --dump-topk）")
            return 2

    df = pd.read_parquet(PQ)
    q = pd.read_parquet(QFILE)
    n = df["qidx"].nunique()
    q = q.head(n).reset_index(drop=True)

    if not LLM_CACHE.exists():
        print(f"缺 {LLM_CACHE}（先跑 classify_specificity.py 生成 LLM 预测）")
        return 2
    cache = json.loads(LLM_CACHE.read_text(encoding="utf-8"))
    pred = np.array([cache.get(str(s), -1) for s in q["query"]])
    truth = q["specificity"].to_numpy()
    ok = pred >= 0
    print(f"题目 {n} 条 | 真值 Broad {(truth == 0).sum()} / Specific {(truth == 1).sum()}"
          f" | LLM 预测有效 {ok.sum()}")

    wide = {(m, s): df[df["system"] == s].set_index("qidx")[m].sort_index().to_numpy()
            for m in METRICS for s in SYSTEMS if s in set(df["system"])}

    groups: list[tuple[str, np.ndarray]] = [
        ("全部", np.ones(n, dtype=bool)),
        ("真值·Broad", truth == 0),
        ("真值·Specific", truth == 1),
        ("预测·Broad", ok & (pred == 0)),
        ("预测·Specific", ok & (pred == 1)),
    ]

    print("\n" + "=" * 100)
    print("① recall@1 / mrr@10 按组 × 系统")
    print("=" * 100)
    for metric in ("recall@1", "recall@10", "mrr@10"):
        print(f"\n[{metric}]")
        head = f"  {'组':<14}{'n':>5}  " + "".join(f"{s:>9}" for s in SYSTEMS)
        print(head)
        for name, mask in groups:
            if mask.sum() == 0:
                continue
            vals = [wide[(metric, s)][mask].mean() for s in SYSTEMS if (metric, s) in wide]
            print(f"  {name:<14}{int(mask.sum()):>5}  " + "".join(f"{v:9.4f}" for v in vals))

    print("\n" + "=" * 100)
    print("② 各组内部的最优配置（路由的正当理由，就是这里出现差异）")
    print("=" * 100)
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        best, best_v = None, -1
        for s in SYSTEMS:
            if ("recall@1", s) not in wide:
                continue
            v = wide[("recall@1", s)][mask].mean()
            if v > best_v:
                best, best_v = s, v
        mrr_best, mrr_v = None, -1
        for s in SYSTEMS:
            if ("mrr@10", s) not in wide:
                continue
            v = wide[("mrr@10", s)][mask].mean()
            if v > mrr_v:
                mrr_best, mrr_v = s, v
        print(f"  {name:<14} best(R@1)={best:<8} {best_v:.4f}   "
              f"best(MRR)={mrr_best:<8} {mrr_v:.4f}")

    print("\n" + "=" * 100)
    print("③ 混合检索的增益（hyb0.7 − dense），看扩展该给哪一组用")
    print("=" * 100)
    print(f"  {'组':<14}{'n':>5}{'R@1 dense':>12}{'R@1 hyb0.7':>13}{'Δ':>10}{'相对':>9}")
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        d = wide[("recall@1", "dense")][mask].mean()
        h = wide[("recall@1", "hyb0.7")][mask].mean()
        print(f"  {name:<14}{int(mask.sum()):>5}{d:>12.4f}{h:>13.4f}{h - d:>+10.4f}"
              f"{100 * (h - d) / max(d, 1e-9):>8.1f}%")

    print("\n[ 判读 ]")
    print("  · 若两组的 best 配置不同、且混合增益差异大 → L1 分流有正当理由，值得做；")
    print("  · 若两组 best 相同、增益也接近 → **分流没有必要**，L1 的收益要靠别的机制（如成本控制）；")
    print("  · 注意：预测组里混入了分类错误的样本，会**稀释**差异（这是真值组与预测组对比的意义）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
