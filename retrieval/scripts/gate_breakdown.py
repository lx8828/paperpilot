"""门控的逐题分解：**被放行 vs 被关掉**两组上，门控相对"无门控"各损失了什么。

用途：g4 显示门控在 hard 查询集上把 R@5 从 0.8417 压到 0.6917（−15pt）——
      这个损失是不是集中发生在"被关掉"的那 23% 上？如果是，说明门控把增益的**长尾**砍掉了。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def main() -> int:
    pairs = [
        ("ARXIV_GATE100-100_th-0.01_HARD_multi_corpus_raw.csv",
         "ARXIV_CE100-100_HARD_multi_corpus_raw.csv", "arXiv-hard"),
        ("ARXIV_GATE100-100_th-0.01_multi_corpus_raw.csv",
         "ARXIV_CE100-100_multi_corpus_raw.csv", "arXiv-normal"),
    ]
    for gate_f, ce_f, tag in pairs:
        gp, cp = RESULTS / gate_f, RESULTS / ce_f
        if not gp.exists() or not cp.exists():
            print(f"[{tag}] 缺文件，跳过")
            continue
        g = pd.read_parquet if False else pd.read_csv(gp).set_index("q")
        c = pd.read_csv(cp).set_index("q")
        common = g.index.intersection(c.index)
        g, c = g.loc[common], c.loc[common]
        op = g["门控放开"] == 1
        print("=" * 88)
        print(f"[{tag}]  共 {len(g)} 题，门控放行 {int(op.sum())}（{100 * op.mean():.1f}%）")
        print("=" * 88)
        print(f"{'组':<12}{'n':>5}{'门控 R@5':>12}{'无门控 R@5':>13}{'Δpt':>10}"
              f"{'门控 R@1':>11}{'无门控 R@1':>13}")
        for name, m in (("放行", op), ("关掉", ~op), ("全部", op | ~op)):
            if m.sum() == 0:
                continue
            a = g.loc[m, "R@5"].mean()
            b = c.loc[m, "R@5"].mean()
            a1 = g.loc[m, "R@1"].mean()
            b1 = c.loc[m, "R@1"].mean()
            print(f"{name:<12}{int(m.sum()):>5}{a:>12.4f}{b:>13.4f}{100 * (a - b):>+10.2f}"
                  f"{a1:>11.4f}{b1:>13.4f}")
        # 关掉组里"无门控本来能命中、门控后丢了"的题数
        lost = int(((c.loc[~op, "R@5"] > 0) & (g.loc[~op, "R@5"] == 0)).sum())
        print(f"\n  「被关掉」组里，无门控本来 R@5 命中、门控后丢失的题数：{lost}"
              f"（占全部 {100 * lost / len(g):.1f}%）")
        gain = int(((g.loc[op, "R@5"] == 0) & (c.loc[op, "R@5"] > 0)).sum())
        print(f"  「被放行」组里，门控后反而丢失的题数：{gain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
