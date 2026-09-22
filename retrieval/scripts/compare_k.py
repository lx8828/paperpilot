"""回答「K 该选多少」的成本-收益问题，只针对**交付口径**（R@1 / R@5，顶多 R@10）。

动机：下游是单篇精读系统 —— 用户看到的是 top-5，不可能消费 50 篇。
      所以池子指标（R@20/R@100）不该主导决策，**R@1/R@5 才是**。

从现成缓存（`llm_rerank_cache.json` 里 k20/k30/k50/k100 × full × 臂B）恢复逐题排序，
做三件事：

  ① 交付口径指标表（R@1 / R@5 / R@10 / MRR@10 + 论文口径 Broad R@20 / Specific R@5）
  ② **K=20 vs K=50 的配对显著性**（配对 bootstrap on recall + McNemar on hit）
     —— 直接回答「+4.4pt 是真的还是噪声」
  ③ 成本表（token / $/千次），重点是**倍数**

⚠️ 不调用 API（全部命中缓存），纯本地计算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from significance import mcnemar  # noqa: E402

RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
TOPK = RESULTS / "topk1000.npz"
CE_SCORES = RESULTS / "rerank1000.npy"
CACHE = DERIVED / "llm_rerank_cache.json"
POOL_DEPTH = 100

# 实测 token/查询（来自 llm_rerank.py 的 tok_by_cfg；见 results/LITSEARCH_LLM_RERANK.md）
TOK = {20: 4082, 30: 6149, 50: 10049, 100: 19304}
USD_PER_MTOK = 0.14          # 与 README 的 $0.6 / $1.4 对齐（**倍数**与单价无关）


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """极简 markdown 表格（不依赖 tabulate）。"""
    cols = list(df.columns)
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, (int, float, np.floating))
                     else str(v))  # noqa: E731
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    q = pd.read_parquet(QFILE)
    truth = q["specificity"].to_numpy()
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    n = len(golds)

    pool = np.load(TOPK)["hyb0.5"][:, :POOL_DEPTH]
    ce = np.load(CE_SCORES)[:, :POOL_DEPTH]
    ce_order = [pool[i][np.argsort(-ce[i])] for i in range(n)]
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    orders: dict[int, list[np.ndarray | None]] = {}
    for key, c in cache.items():
        if not key.startswith("k") or ":full:B:" not in key:
            continue
        k_str, _, _, i_str = key.split(":")
        k = int(k_str[1:])
        orders.setdefault(k, [None] * n)
        disp, perm = np.array(c["disp"]), np.array(c["perm"]) - 1
        orders[k][int(i_str)] = ce_order[int(i_str)][:k][disp[perm]]

    ks = sorted(orders)
    print(f"从缓存恢复的 K：{ks}；每题可用数："
          f"{ {k: sum(x is not None for x in orders[k]) for k in ks} }\n")

    def cov(rl, k, i) -> float:
        """第 i 题的 recall@k = |gold ∩ top-k| / |gold|"""
        r = rl[i]
        if r is None or not golds[i]:
            return 0.0
        return len(golds[i] & set(r[:k].tolist())) / len(golds[i])

    def R(rl, k, mask=None) -> float:
        idx = list(range(n)) if mask is None else list(np.where(mask)[0])
        return float(np.mean([cov(rl, k, i) for i in idx]))

    def H(rl, k, i) -> int:
        r = rl[i]
        return 1 if (r is not None and golds[i] & set(r[:k].tolist())) else 0

    # ── ① 交付口径指标 ────────────────────────────────────────
    print("=" * 100)
    print("① 交付口径指标（下游是单篇精读 → 只关心 R@1 / R@5，顶多 R@10）")
    print("=" * 100)
    rows = []
    for k in ks:
        rl = orders[k]
        mrr = []
        for i in range(n):
            r = rl[i]
            mrr.append(next((1 / (j + 1) for j, d in enumerate(r[:10].tolist())
                             if d in golds[i]), 0.0) if r is not None else 0.0)
        rows.append({"K": k, "R@1": R(rl, 1), "R@5": R(rl, 5), "R@10": R(rl, 10),
                     "MRR@10": float(np.mean(mrr)),
                     "SpecificR@5": 100 * R(rl, 5, truth == 1),
                     "BroadR@20": 100 * R(rl, 20, truth == 0)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  SpecificR@5 / BroadR@20 为论文口径（%）—— 论文：GritLM-7B 74.8 / 70.8，"
          "GPT-4o rerank(GritLM) 79.2 / 75.3")

    # ── ② K=20 vs K=50 配对显著性 ────────────────────────────
    print("\n" + "=" * 100)
    print("② K=20 vs K=50：同一 LLM、同一批题、只改喂进去的候选数 → **配对**比较")
    print("=" * 100)
    a, b = orders[20], orders[50]
    ok = [i for i in range(n) if a[i] is not None and b[i] is not None]
    rng = np.random.default_rng(0)
    sig_rows = []
    for kk in (1, 5, 10):
        va = np.array([cov(a, kk, i) for i in ok])
        vb = np.array([cov(b, kk, i) for i in ok])
        d = vb - va
        boots = np.array([d[rng.integers(0, len(ok), len(ok))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ha = np.array([H(a, kk, i) for i in ok])
        hb = np.array([H(b, kk, i) for i in ok])
        o1, o0 = int(((hb == 1) & (ha == 0)).sum()), int(((hb == 0) & (ha == 1)).sum())
        pv = mcnemar(o1, o0)
        sig_rows.append({"指标": f"R@{kk}", "K=20": va.mean(), "K=50": vb.mean(),
                         "Δ": d.mean(), "Δpt": 100 * d.mean(),
                         "CI下": lo, "CI上": hi,
                         "显著": "是" if (lo > 0 or hi < 0) else "否",
                         "hit翻转(仅50/仅20)": f"{o1}/{o0}", "McNemar_p": pv})
        print(f"\n  [R@{kk}]  K=20={va.mean():.4f}  K=50={vb.mean():.4f}  "
              f"Δ={d.mean():+.4f} ({100 * d.mean():+.2f}pt)")
        print(f"        配对 bootstrap 95%CI = [{lo:+.4f}, {hi:+.4f}] → "
              f"{'**显著**' if lo > 0 or hi < 0 else '不显著（CI 跨 0）'}")
        print(f"        hit@{kk} 翻转：仅K=50命中 {o1} | 仅K=20命中 {o0} | "
              f"净 {o1 - o0:+d} | McNemar p={pv:.4g} "
              f"{'**显著**' if pv < 0.05 else '不显著'}")
    ts = pd.DataFrame(sig_rows)

    # ── ③ 成本 ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("③ 成本（$/千次查询；重点是**倍数** —— 与单价无关）")
    print("=" * 100)
    base = TOK[20]
    cr = []
    for k in ks:
        usd = TOK[k] * USD_PER_MTOK / 1000
        cr.append({"K": k, "tok/查询": TOK[k], "$/千次": round(usd, 3),
                   "$/百万次": round(usd * 1000, 1), "相对K=20": f"{TOK[k] / base:.2f}x",
                   "R@1": R(orders[k], 1), "R@5": R(orders[k], 5)})
    tc = pd.DataFrame(cr)
    print(tc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    d5 = 100 * (R(orders[50], 5) - R(orders[20], 5))
    d1 = 100 * (R(orders[50], 1) - R(orders[20], 1))
    extra = (TOK[50] - base) * USD_PER_MTOK / 1000
    print(f"\n  K=20 → K=50：prompt token ×{TOK[50] / base:.2f}，钱 ×{TOK[50] / base:.2f}"
          f"（+${extra:.2f}/千次）")
    print(f"  换来：R@1 {d1:+.2f}pt（≈0）、R@5 {d5:+.2f}pt")
    if d5 > 0:
        print(f"  → 每个 R@5 百分点 = +${extra / d5:.2f}/千次")
    print("\n  ⚠️ 单价由 README 的 $0.6/$1.4 反推，仅用于给出绝对量级；**倍数**才是可比的。")

    out = RESULTS / "LITSEARCH_K_DECISION.md"
    out.write_text(
        "# LitSearch · K 选择决策（交付口径：R@1 / R@5）\n\n"
        "下游为**单篇精读系统** → 用户看到的是 top-5，池子指标（R@20/R@100）不主导决策。\n\n"
        "## ① 交付口径指标\n\n" + md_table(t) +
        "\n\n> `SpecificR@5` / `BroadR@20` 为论文口径（%）。\n"
        "\n## ② K=20 vs K=50（配对显著性）\n\n"
        + md_table(ts) +
        "\n\n## ③ 成本\n\n" + md_table(tc) +
        f"\n\n- prompt token ×{TOK[50] / base:.2f}，钱 ×{TOK[50] / base:.2f}"
        f"（+${extra:.2f}/千次）\n"
        f"- 换来 R@1 {d1:+.2f}pt、R@5 {d5:+.2f}pt\n"
        f"- **每个 R@5 百分点 = +${extra / d5:.2f}/千次**\n"
        "\n> 单价由 README 的 `$0.6/$1.4` 反推；**倍数**与单价无关。\n", encoding="utf-8")
    print(f"\n已写出：{out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
