"""LitSearch 数据体检：把 qrels 结构与分布打出来（决定指标怎么选）。

这一步是"造尺子"——在写任何检索代码之前，必须先知道：
  · 一条查询平均对应几个 gold（1 个 → 答案是唯一的；多个 → 才需要 Recall@k）
  · 有没有 0 gold 的查询（有 → 评测时要排除，否则那是不可答的题）
  · 查询按来源（query_set）与难度（specificity）怎么分布（→ 分层抽样的依据）

用法：
    python retrieval/scripts/inspect_litsearch.py
    python retrieval/scripts/inspect_litsearch.py --samples 6
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"


def as_list(v) -> list:
    """把 `corpusids` 归一成 Python list。

    ⚠️ 实测（2026-09-18）：pandas 读出来的是 **numpy 数组**（不是字符串），
    所以不能只判断 str/list —— 第一版漏了这条，导致"597 条全 0 gold"的假结论。
    这里按"字符串 → 解析 / 其它可迭代 → 直接展开 / 单值 → 包一层"三段处理。
    """
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for parse in (json.loads, ast.literal_eval):      # JSON 优先，退回更宽松的
            try:
                out = parse(s)
                return list(out) if isinstance(out, (list, tuple)) else [out]
            except Exception:  # noqa: BLE001
                continue
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    try:                                                   # numpy 数组 / pyarrow list
        return [x for x in v]
    except TypeError:
        return [v]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4, help="打印几条查询样例")
    args = ap.parse_args()

    if not QFILE.exists():
        print(f"缺少查询文件：{QFILE}\n请先运行：python retrieval/scripts/fetch_litsearch.py query")
        return 2

    df = pd.read_parquet(QFILE)
    print("=" * 72)
    print("LitSearch · query 体检")
    print("=" * 72)
    print(f"文件      : {QFILE.relative_to(HERE)}")
    print(f"查询数    : {len(df)}")
    print(f"列        : {list(df.columns)}")
    # 类型诊断：qrels 字段到底长什么样（第一版就栽在这里）
    if "corpusids" in df.columns:
        first = df["corpusids"].iloc[0]
        print(f"corpusids : dtype={df['corpusids'].dtype}  首个值类型={type(first).__name__}")

    # ── 标注字段分布 ─────────────────────────────────────────────
    print("\n[ 查询来源 query_set ]")
    for k, v in df["query_set"].value_counts().items():
        print(f"  {k:<18} {v:4d}  ({100 * v / len(df):.1f}%)")

    for col in ("specificity", "quality"):
        if col in df.columns:
            vc = df[col].value_counts().sort_index()
            print(f"\n[ {col} ]")
            for k, v in vc.items():
                print(f"  {k:<18} {v:4d}  ({100 * v / len(df):.1f}%)")

    # ── gold（qrels）分布：最关键 ─────────────────────────────────
    golds = df["corpusids"].map(as_list)
    counts = golds.map(len)
    print("\n[ gold 数 / 查询 ]  ← qrels 结构")
    for k, v in sorted(collections.Counter(counts).items()):
        print(f"  {k:>2d} 个 gold : {v:4d} 条查询  ({100 * v / len(df):.1f}%)")
    print(f"  合计 gold 判断   : {counts.sum()}")
    print(f"  均值 / 中位 / 最大: {counts.mean():.2f} / {counts.median():.0f} / {counts.max()}")

    n0 = int((counts == 0).sum())
    n1 = int((counts == 1).sum())
    nm = int((counts >= 2).sum())
    print("\n[ 判读 ]")
    print(f"  0 gold（不可答，评测要排除）: {n0} 条")
    print(f"  恰好 1 gold               : {n1} 条 ({100 * n1 / len(df):.1f}%)")
    print(f"  ≥2 gold（需要 Recall@k）  : {nm} 条 ({100 * nm / len(df):.1f}%)")
    if n1 == len(df) - n0:
        print("  → 单正例任务：答案唯一，主指标用 Recall@1/5/10 + MRR 即可（注意 @k 的 k 不能超过候选规模）")
    else:
        print("  → 多正例任务：主指标用 nDCG@10 / Recall@k（要召回全部相关文档）")

    # ── 样例 ─────────────────────────────────────────────────────
    print(f"\n[ 样例（前 {args.samples} 条）]")
    for i in range(min(args.samples, len(df))):
        r = df.iloc[i]
        g = as_list(r["corpusids"])
        print(f"\n  #{i + 1}  set={r['query_set']}  spec={r['specificity']}  qual={r['quality']}  gold={len(g)}")
        print(f"      Q: {str(r['query'])[:190]}")
        print(f"  gold: {g[:6]}{' ...' if len(g) > 6 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
