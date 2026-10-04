"""**覆盖导向重排**（MMR / 去冗余）——验证"现有精排器在优化错的指标"这个判断。

## 为什么做
`_k_deliver_litsearch.py` 发现：交付 N 篇时，
- **`hit@N`**（那 1 篇 gold 是否进前 N）——**现有所有精排器优化的就是它**；
- **`coverage@N`**（N 篇里覆盖**全部** gold）——**跨篇聚合交付的真判据，没人优化**。
实测 CE（pointwise）在 交付20 的覆盖率上**反而比融合差**（21% vs 27%）——
因为 pointwise 永远把"语义最像的"排前面，**系统性丢多样性**。

## 做法
在 CE 序之上做 **MMR**（Maximal Marginal Relevance）：
    score(d) = λ·rel(d) − (1−λ)·max_{s∈已选} cos(d, s)
- `rel` = CE 分数（按题 min-max 归一）或 **rank 倒数**（CE 分数区间很窄，做对照）
- `cos` = `bge-m3` 论文向量余弦（`derived/emb/`，与检索同源）

## 指标与守卫
- **主指标**：34 题多 gold 子集的 `coverage@N`（N 篇覆盖全部 gold）
- **守卫**：全部 597 题的 `hit@1 / hit@5`（**不许退化**——交付多篇不能牺牲"第一命中"）
- 参考：CE 基线、LLM listwise（cache `k50:full:B`）

## ⚠️ 口径
LitSearch 真值是**单答案型**（563/597 单 gold）→ `coverage@N` 只在 **34 题**上有意义，
样本小，**只能看方向与单调性**，不能当点估计。

用法：uv run python retrieval/tmp/_cover_rerank.py [--lam 1.0,0.9,...]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "retrieval" / "results"
DER = ROOT / "retrieval" / "data" / "litsearch" / "derived"
DEPTH = 100
NS = (1, 3, 5, 10, 20, 50)


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    q = pd.read_parquet(glob.glob(str(ROOT / "retrieval/data/litsearch/query/*.parquet"))[0])
    ids = np.load(DER / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[int(x)] for x in np.atleast_1d(r) if int(x) in id2row}
             for r in q["corpusids"]]
    q = q.assign(gold=golds, n_gold=[len(g) for g in golds])
    pool = np.load(RES / "topk1000.npz")["hyb0.5"][:, :DEPTH]
    ce = np.load(RES / "rerank1000.npy")[:, :DEPTH]
    ceord = np.take_along_axis(pool, np.argsort(-ce, axis=1), axis=1)
    cesc = np.take_along_axis(ce, np.argsort(-ce, axis=1), axis=1)
    cache = json.loads((DER / "llm_rerank_cache.json").read_text(encoding="utf-8"))
    return q, ceord, cesc, pool, cache


def load_vecs(rows: np.ndarray) -> np.ndarray:
    """只取需要的行（`rows` 是行号数组）→ (n_docs, 1024) float32 归一化向量。"""
    n = int(rows.max()) + 1
    out = np.zeros((n, 1024), dtype=np.float32)
    parts = sorted((DER / "emb").glob("part_*.npy"))
    per = 5000
    for pi, p in enumerate(parts):
        lo, hi = pi * per, (pi + 1) * per
        sel = rows[(rows >= lo) & (rows < hi)]
        if not len(sel):
            continue
        a = np.load(p, mmap_mode="r")
        out[sel] = np.asarray(a[sel - lo], dtype=np.float32)
    nz = np.linalg.norm(out[rows], axis=1, keepdims=True) + 1e-9
    out[rows] = out[rows] / nz
    return out


def mmr(order: np.ndarray, rel: np.ndarray, vec: np.ndarray, lam: float, top: int) -> np.ndarray:
    """MMR 重排（`order` = 候选行号按参考序；`rel` 与之对齐）。返回重排后的行号。"""
    cand = order[:top]
    r = rel[:top]
    S = vec[cand] @ vec[cand].T                      # (top, top) 余弦
    np.fill_diagonal(S, -1.0)
    picked = [0]
    rest = list(range(1, top))
    while rest and len(picked) < top:
        sims = S[np.ix_(rest, picked)].max(axis=1)
        sc = lam * r[rest] - (1 - lam) * sims
        j = int(np.argmax(sc))
        picked.append(rest.pop(j))
    return cand[picked]


def select_with_S(cand: np.ndarray, rel: np.ndarray, S: np.ndarray,
                  lam: float, n_out: int) -> np.ndarray:
    """从 `cand`（深池）里"选"前 `n_out` 篇：score = λ·rel − (1−λ)·max 已选相似度。

    ⚠️ 与 `mmr()` 的区别：`mmr()` 只在**前 top 名内重排**（池集合不变 → 覆盖率提不动，
    已实测为负结果）；本函数是**从深池里挑交付**，才能真正把深位的 gold 提上来。
    `S` = `cand` 之间的余弦矩阵（按题只算一次，供多组 λ 复用）。
    """
    picked = [int(np.argmax(rel))]
    sims = S[:, picked[0]].copy()
    for _ in range(n_out - 1):
        sc = lam * rel - (1 - lam) * sims
        sc[picked] = -np.inf
        j = int(np.argmax(sc))
        picked.append(j)
        sims = np.maximum(sims, S[:, j])
    return cand[picked]


def hit_cov(ranked: np.ndarray, golds: list[set[int]]) -> tuple[list[float], list[float]]:
    """→ (hit@N 列表, 覆盖全部 gold@N 列表)，N 取自 `NS`。"""
    hs, cs = [], []
    for N in NS:
        h = c = 0.0
        for row, g in zip(ranked[:, :N], golds):
            got = len(g & set(row.tolist()))
            h += 1.0 if got else 0.0
            c += 1.0 if got == len(g) else 0.0
        n = max(len(golds), 1)
        hs.append(h / n)
        cs.append(c / n)
    return hs, cs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="1.0,0.9,0.8,0.7,0.6,0.5")
    ap.add_argument("--top", type=int, default=50, help="MMR 作用范围（只重排这么多名额）")
    args = ap.parse_args()
    lams = [float(x) for x in args.lam.split(",") if x.strip()]

    q, ceord, cesc, pool, cache = load()
    graw = np.unique(np.concatenate([ceord.ravel(), pool.ravel()]))
    V = load_vecs(graw)
    golds = list(q["gold"])
    mu = (q["n_gold"] >= 2).to_numpy()
    gm = [g for g, m in zip(golds, mu) if m]
    print(f"\n{'=' * 116}\n【覆盖导向重排】CE top-{DEPTH} → MMR（作用范围 {args.top}）"
          f" ｜ 597 题 ｜ 多gold(≥2) {int(mu.sum())} 题 ｜ 向量 {len(graw)} 篇\n{'=' * 116}")

    for name, order in (("CE 基线", ceord), ("融合 hyb0.5", pool)):
        hs, _ = hit_cov(order, golds)
        _, mcs = hit_cov(order[mu], gm)               # ⚠️ 覆盖率必须只看多 gold 的那 34 题
        print(f"\n  {name}")
        print("    交付N   " + "".join(f"{N:>9}" for N in NS))
        print("    hit@N   " + "".join(f"{v:>9.1%}" for v in hs))
        print("    **多gold覆盖** " + "".join(f"{v:>9.1%}" for v in mcs))

    rel_ce = np.zeros_like(cesc)
    for i in range(len(cesc)):
        c = cesc[i]
        rel_ce[i] = (c - c.min()) / (c.max() - c.min() + 1e-9)
    rel_rk = (1.0 / np.arange(1, DEPTH + 1)).astype(np.float32)[None, :].repeat(len(cesc), 0)

    print(f"\n{'#' * 116}\nMMR 扫描（作用范围 top-{args.top}）")
    print(f"  {'λ':>6}{'rel':>6}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@20':>8}"
          f"{'cov@5':>8}{'cov@10':>8}{'cov@20':>8}{'cov@50':>8}")
    best: list[tuple[float, str, list[float], list[float]]] = []
    for rel_name, rel in (("ce", rel_ce), ("rank", rel_rk)):
        for lam in lams:
            out = np.array(ceord, copy=True)          # 前 top 位重排，其余保持 CE 序
            for i in range(len(ceord)):
                out[i, :args.top] = mmr(ceord[i], rel[i], V, lam, args.top)
            hs, _ = hit_cov(out, golds)
            _, mcs = hit_cov(out[mu], gm)
            best.append((lam, rel_name, hs, mcs))
            print(f"  {lam:>6.2f}{rel_name:>6}{hs[0]:>8.1%}{hs[1]:>8.1%}{hs[2]:>8.1%}"
                  f"{hs[4]:>8.1%}{mcs[2]:>8.1%}{mcs[3]:>8.1%}{mcs[4]:>8.1%}{mcs[5]:>8.1%}")

    base_h1, base_c20 = None, None
    for lam, rn, hs, mcs in best:
        if lam == 1.0 and rn == "ce":
            base_h1, base_c20 = hs[0], mcs[4]
    if base_h1 is not None:
        print(f"\n########## 与 CE 基线（λ=1）对比：hit@1 {base_h1:.1%} ／ 多gold覆盖@20 {base_c20:.1%}")
        for lam, rn, hs, mcs in sorted(best, key=lambda t: -t[3][4]):
            if lam == 1.0 and rn == "ce":
                continue
            print(f"  λ={lam:<4} rel={rn:<5} hit@1 {hs[0]:>6.1%}（{hs[0] - base_h1:+.1%}）"
                  f" ｜ cov@20 {mcs[4]:>6.1%}（{mcs[4] - base_c20:+.1%}）"
                  f" ｜ cov@50 {mcs[5]:>6.1%}")

    # ── ③ **深池 + 覆盖导向"选交付"**（MMR 的正确用法：选，不是重排） ──
    deep = np.load(RES / "topk1000.npz")["hyb0.5"]
    print(f"\n{'#' * 116}\n③ **深池 → 覆盖导向选交付**（MMR-select；候选来自融合 top-1000，无 LLM 成本）")
    print(f"  {'池深':>5}{'λ':>6}{'hit@20':>9}{'hit@50':>9}{'多gold覆盖@20':>14}{'@50':>8}"
          f"{'池内天顶':>10}")
    lams_c = (1.0, 0.9, 0.8, 0.6)
    for pool_n in (100, 200, 500, 1000):
        need = np.unique(np.concatenate([ceord.ravel(), deep[:, :pool_n].ravel()]))
        Vd = load_vecs(need)
        rel_d = (1.0 / (1.0 + np.arange(pool_n))).astype(np.float32)
        outs = {lam: np.zeros((len(deep), 50), dtype=np.int64) for lam in lams_c}
        for i in range(len(deep)):
            cand = deep[i, :pool_n]
            Vc = Vd[cand]
            S = Vc @ Vc.T
            for lam in lams_c:
                outs[lam][i] = select_with_S(cand, rel_d, S, lam, 50)
        ceil = sum(1 for row, g in zip(deep[mu][:, :pool_n], gm)
                   if g <= set(row.tolist())) / len(gm)
        for lam in lams_c:
            o = outs[lam]
            hs, _ = hit_cov(o, golds)
            _, mcs = hit_cov(o[mu], gm)
            print(f"  {pool_n:>5}{lam:>6.1f}{hs[4]:>9.1%}{hs[5]:>9.1%}"
                  f"{mcs[4]:>14.1%}{mcs[5]:>8.1%}{ceil:>10.1%}")

    # ── 池覆盖天花板：任何**重排**都无法超过"全部 gold 都在池内"的比例 ──
    print(f"\n{'#' * 116}\n池覆盖天花板（34 题多 gold）：『**全部 gold 都在池内**』的比例 = 重排能到的上界")
    z = np.load(RES / "topk1000.npz")["hyb0.5"]                   # 融合全序（top-1000）
    print(f"  {'池深K':>6}{'融合池':>10}{'CE池':>10}   说明")
    for K in (20, 50, 100, 200, 500, 1000):
        fk = sum(1 for row, g in zip(z[mu][:, :K], gm) if g <= set(row.tolist()))
        ck = (sum(1 for row, g in zip(ceord[mu][:, :K], gm) if g <= set(row.tolist()))
              if K <= DEPTH else None)
        print(f"  {K:>6}{fk / len(gm):>10.1%}" + (f"{ck / len(gm):>10.1%}" if ck is not None
                                                  else f"{'—':>10}")
              + ("   ← CE 池（现用）" if K == DEPTH else
                 ("   ← 融合 top-1000（R@1000 93.8%）" if K == 1000 else "")))

    print("\n读法：**优先看 `cov@N`（主指标），再看 `hit@1` 是否退化（守卫）**。"
          "\n      若某 λ 能『cov@20 上升 且 hit@1 不降』→ 覆盖导向重排成立，可进生产候选。"
          "\n      ⚠️ 34 题样本小，结论只能当方向；要定 λ 必须扩到多答案真值集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
