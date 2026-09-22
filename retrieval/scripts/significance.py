"""第 3b 步：检索指标差异的显著性检验。

**为什么两种检验都要**：
- **McNemar 精确检验**（`qa/compare/_signif.py` 同款实现）：只适用**二元**结果 ——
  这里用 "hit@k"（gold 是否进 top-k）。同题配对，只看翻转对，n 小也不做正态近似。
- **配对 bootstrap**：适用于**连续**指标（Recall@k / MRR@10）—— IR 领域的标准做法。
  对"逐题差异"重采样，看均值的 95% 置信区间是否跨 0。

⚠️ 为什么不能用普通双样本 t 检验：所有系统跑的是**同一批题**，
"谁在哪些题上赢"是强相关的配对信息，独立样本检验会低估显著性。

用法：
    python retrieval/scripts/significance.py
    python retrieval/scripts/significance.py --boot 20000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
PQ = RESULTS / "per_query.parquet"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"


def mcnemar(b: int, c: int) -> float:
    """双尾精确 p（H0: 翻转对称）。与 paperpilot 的 `qa/compare/_signif.py` 同一实现。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                     seed: int = 20260918) -> tuple[float, float, float, float]:
    """配对 bootstrap：返回 (mean_diff, ci_lo, ci_hi, p_two_sided)。"""
    d = x - y
    if len(d) == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # 双尾 p：均值分布里跨 0 的比例（取两侧较小者 ×2）
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    if not PQ.exists():
        print(f"缺 {PQ}\n请先运行：python retrieval/scripts/eval_retrieval.py --alpha 0.3,0.5,0.7,0.9")
        return 2

    df = pd.read_parquet(PQ)
    systems = list(dict.fromkeys(df["system"]))
    n = df["qidx"].nunique()
    print(f"逐题结果：{len(df)} 行 = {n} 题 × {len(systems)} 系统")
    print(f"系统：{systems}")

    # 宽表：行=题，列=系统
    def wide(col: str) -> pd.DataFrame:
        return df.pivot(index="qidx", columns="system", values=col).sort_index()

    # 分层：查询是否含精确 token（与 eval 脚本口径一致）
    q = pd.read_parquet(QFILE).head(n).reset_index(drop=True)
    mask_tok = (q["query"].str.contains(r"\b[A-Z]{2,}\b", regex=True)
                | q["query"].str.contains(r"\d", regex=True)).to_numpy()

    PAIRS = [("dense", "bm25"), ("hyb0.5", "dense"), ("hyb0.7", "dense"),
             ("hyb0.7", "hyb0.5"), ("hyb0.5", "bm25")]

    print("\n" + "=" * 96)
    print("① McNemar 精确检验（二元 hit@1 / hit@10）")
    print("=" * 96)
    for metric in ("hit@1", "hit@10"):
        w = wide(metric)
        print(f"\n[{metric}]")
        for a, b in PAIRS:
            if a not in w.columns or b not in w.columns:
                continue
            x, y = w[a].to_numpy(), w[b].to_numpy()
            only_a = int(((x == 1) & (y == 0)).sum())
            only_b = int(((x == 0) & (y == 1)).sum())
            pv = mcnemar(only_a, only_b)
            mark = "**显著**" if pv < 0.05 else "不显著"
            print(f"  {a:>7} vs {b:<6}: 仅{a}命中 {only_a:3d} | 仅{b}命中 {only_b:3d} | "
                  f"净 Δ={only_a - only_b:+4d} | p={pv:.4g}  {mark}")

    print("\n" + "=" * 96)
    print(f"② 配对 bootstrap（连续指标，{args.boot} 次重采样，95% CI）")
    print("=" * 96)
    for metric in ("recall@1", "recall@10", "mrr@10"):
        w = wide(metric)
        print(f"\n[{metric}]  总体均值：" +
              " | ".join(f"{s}={w[s].mean():.4f}" for s in systems if s in w.columns))
        for a, b in PAIRS:
            if a not in w.columns or b not in w.columns:
                continue
            d, lo, hi, pv = paired_bootstrap(w[a].to_numpy(), w[b].to_numpy(), args.boot)
            sig = "**显著**" if (lo > 0 or hi < 0) else "不显著（CI 跨 0）"
            print(f"  {a:>7} vs {b:<6}: Δ={d:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  p≈{pv:.4g}  {sig}")

    print("\n" + "=" * 96)
    print("③ 分层：含精确 token（大写缩写 / 数字）")
    print("=" * 96)
    for metric in ("recall@1", "mrr@10"):
        w = wide(metric)
        print(f"\n[{metric}]")
        for label, mk in (("含token", mask_tok), ("纯自然语言", ~mask_tok)):
            sub = w[mk]
            print(f"  {label}（n={int(mk.sum())}）：" +
                  " | ".join(f"{s}={sub[s].mean():.4f}" for s in systems if s in sub.columns))
            for a, b in PAIRS:
                if a not in sub.columns or b not in sub.columns:
                    continue
                d, lo, hi, pv = paired_bootstrap(sub[a].to_numpy(), sub[b].to_numpy(), args.boot)
                sig = "**显著**" if (lo > 0 or hi < 0) else "不显著"
                print(f"    {a:>7} vs {b:<6}: Δ={d:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  p≈{pv:.4g}  {sig}")
    return 0


if __name__ == "__main__":
    # 把全部输出同时落盘（结论要可追溯，不能只留在终端里）
    import contextlib
    import io
    import sys

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
            return len(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    _buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, _buf)):
        _rc = main()
    RESULTS.mkdir(parents=True, exist_ok=True)
    _out = RESULTS / "LITSEARCH_SIGNIFICANCE.md"
    _out.write_text("# LitSearch 检索指标 · 显著性检验\n\n"
                    "McNemar 精确检验（二元 hit@k）+ 配对 bootstrap（连续 Recall/MRR，95% CI）。\n\n"
                    "```text\n" + _buf.getvalue() + "```\n", encoding="utf-8")
    print(f"\n已写出：{_out.relative_to(HERE)}")
    raise SystemExit(_rc)
