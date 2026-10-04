"""**论文检索该交付几篇（N）**——完全复用已有产物重算，零 LLM、不重跑检索。

问题（产品视角）：原设计是"用户问一个研究方向 → 检索返回 **5 篇** → 下游直读/问答"。
若下游语料要放大到 20 篇（做多篇聚合），检索侧的交付数 N 要不要调？调了效果如何？

## 两个必须分开的量（容易混）
- **池深 K**：精排/LLM 看多少候选（决定最终排序质量）
- **交付 N**：用户实际看到几篇（`hit@N` = 那 1 篇标注 gold 是否落进前 N）

## 系统（全部取自已有产物；口径已与 `LITSEARCH_K_DECISION.md` 逐位校验）
1. `hyb0.5`  —— 融合（`topk1000.npz`，池 top-100，与精排同池 → 可比）
2. `CE`      —— `bge-reranker-v2-m3`（`rerank1000.npy` 对同一批 top-100 打分）
3. `CE→LLM listwise` —— **生产口径**：CE top-K 候选 → LLM 一次排列
   （`llm_rerank_cache.json` 的 `k{20,30,50,100}:full:B`；与 `llm_rerank.py` 同还原口径）

## ⚠️ 口径天花板（结论边界）
LitSearch 的 qrels **是单答案型**：597 题里 **563 题只有 1 篇 gold**，34 题 ≥2（最多 5）；
**连 Broad 查询（= 论文定义为"语料 6~20 篇可满足"）的 gold 中位也是 1**。
→ 它**测不出"交付 20 篇覆盖了多少相关论文"**，只能测"那 1 篇标注 gold 是否被捞到"。
   唯一能算覆盖率的是 34 题多 gold 子集（样本小，只能看方向）。

用法：uv run python retrieval/tmp/_k_deliver_litsearch.py
"""
from __future__ import annotations

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
DEPTH = 100                 # 与 `llm_rerank.py` 的 POOL_DEPTH 一致
NS = (1, 3, 5, 10, 20, 30, 50, 100)


def load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    q = pd.read_parquet(glob.glob(str(ROOT / "retrieval/data/litsearch/query/*.parquet"))[0])
    ids = np.load(DER / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[int(x)] for x in np.atleast_1d(r) if int(x) in id2row}
             for r in q["corpusids"]]
    q = q.assign(gold=golds, n_gold=[len(g) for g in golds])
    # ⚠️ 官方口径（`llm_rerank.py`: TOPK=topk1000.npz / CE_SCORES=rerank1000.npy / DEPTH=100）
    pool = np.load(RES / "topk1000.npz")["hyb0.5"][:, :DEPTH]
    ce = np.load(RES / "rerank1000.npy")[:, :DEPTH]
    ceord = np.take_along_axis(pool, np.argsort(-ce, axis=1), axis=1)
    cache = json.loads((DER / "llm_rerank_cache.json").read_text(encoding="utf-8"))
    return q, pool, ceord, cache


def llm_order(cache: dict, ceord: np.ndarray, K: int) -> np.ndarray:
    """LLM listwise 排出的行号序列。还原口径**照抄** `llm_rerank.py:405-411`：
    `order = cand[disp[perm-1]]`（`disp` = 呈现顺序；`perm` = 第 t 名的展示位下标，1-based）。
    """
    out = np.zeros((len(ceord), K), dtype=np.int64)
    for i in range(len(ceord)):
        rec = cache.get(f"k{K}:full:B:{i}")
        cand = np.asarray(ceord[i, :K])
        if not rec:
            out[i] = cand
            continue
        out[i] = cand[np.asarray(rec["disp"])[np.asarray(rec["perm"]) - 1]]
    return out


def stat(idxs: np.ndarray, golds: list[set[int]], N: int) -> tuple[float, float]:
    """→ (hit@N, 覆盖全部 gold@N)"""
    hit = cov = 0.0
    for row, g in zip(idxs[:, :N], golds):
        got = len(g & set(row.tolist()))
        hit += 1.0 if got else 0.0
        cov += 1.0 if got == len(g) else 0.0
    n = max(len(golds), 1)
    return hit / n, cov / n


def curve(q: pd.DataFrame, name: str, idxs: np.ndarray) -> pd.DataFrame:
    golds = list(q["gold"])
    b = (q["specificity"] == 0).to_numpy()
    mu = (q["n_gold"] >= 2).to_numpy()
    gb = [g for g, m in zip(golds, b) if m]
    gs = [g for g, m in zip(golds, ~b) if m]
    gm = [g for g, m in zip(golds, mu) if m]
    rows = []
    for N in NS:
        h, c = stat(idxs, golds, N)
        hb, _ = stat(idxs[b], gb, N)
        hs, _ = stat(idxs[~b], gs, N)
        _, mc = stat(idxs[mu], gm, N)
        rows.append(dict(sys=name, N=N, hit=h, Broad=hb, Specific=hs, mcover=mc))
    return pd.DataFrame(rows)


HDR = (f"  {'交付N':>5}{'hit@N':>9}{'Broad hit':>11}{'Specific hit':>14}"
       f"{'多gold全覆盖':>13}   分层涨幅(Broad/Specific)")


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'=' * 108}\n{title}")
    print(HDR)
    base = df.iloc[0]
    for _, r in df.iterrows():
        tag = "  ← 现行(5)" if r["N"] == 5 else ("  ← 候选(20)" if r["N"] == 20 else "")
        gain = (f"+{r['Broad'] - base['Broad']:.1%} / +{r['Specific'] - base['Specific']:.1%}"
                if r["N"] != 1 else "—")
        print(f"  {int(r['N']):>5}{r['hit']:>9.1%}{r['Broad']:>11.1%}{r['Specific']:>14.1%}"
              f"{r['mcover']:>13.1%}   {gain}{tag}")


def main() -> int:
    q, pool, ceord, cache = load()
    nb = int((q["specificity"] == 0).sum())
    nm = int((q["n_gold"] >= 2).sum())
    print(f"\n{'=' * 108}\n【论文检索该交付几篇】LitSearch 597 题 ｜ Broad {nb} ｜ Specific {597 - nb}"
          f" ｜ 多gold(≥2) {nm} ｜ **单 gold {int((q['n_gold'] == 1).sum())}**\n{'=' * 108}")
    print("⚠️ LitSearch 真值是**单答案型**（连 Broad 查询 gold 中位也是 1）→ "
          "`Broad hit` 只说明『论文标的那 1 篇能否被捞到』，**不是覆盖度**。")

    show(curve(q, "hyb0.5", pool), f"① 融合基线 hyb0.5（池 top-{DEPTH}，无精排）")
    show(curve(q, "CE", ceord), f"② CE 重排 `bge-reranker-v2-m3`（同一批 top-{DEPTH} 候选）")

    print(f"\n{'=' * 108}\n③ **生产口径**：CE top-K → LLM listwise（**池深 K 是变量**，交付 N 是另一维）")
    for K in (20, 30, 50, 100):
        d = curve(q, f"K={K}", llm_order(cache, ceord, K))
        n5 = d[d["N"] == 5].iloc[0]
        n20 = d[d["N"] == 20].iloc[0]
        n50 = d[d["N"] == 50].iloc[0] if K >= 50 else None
        print(f"  池 K={K:<4}｜ **交付5** hit {n5['hit']:.1%}（Broad {n5['Broad']:.1%} / "
              f"Specific {n5['Specific']:.1%}）｜ **交付20** hit {n20['hit']:.1%}"
              f"（Broad {n20['Broad']:.1%} / Specific {n20['Specific']:.1%}）｜ "
              f"多gold全覆盖@20 {n20['mcover']:.1%}"
              + (f"｜ 交付50 hit {n50['hit']:.1%}" if n50 is not None else ""))

    print(f"\n{'=' * 108}\n④ 校验（应逐位等于 `LITSEARCH_K_DECISION.md`）")
    for K in (20, 50, 100):
        d = curve(q, f"K={K}", llm_order(cache, ceord, K))
        print(f"    K={K:<4} SpecificR@5 {d[d['N'] == 5]['Specific'].iloc[0] * 100:6.2f}%"
              f"   BroadR@20 {d[d['N'] == 20]['Broad'].iloc[0] * 100:6.2f}%")
    print("    （文档：K=20 → 76.58 / 61.01；K=50 → 79.86 / 70.30；K=100 → 79.41 / 72.88）")

    print(f"\n{'=' * 108}\n⑤ **多 gold 子集**（{nm} 题）：交付 N 篇时『覆盖全部 gold』的比例（跨篇聚合的真判据）")
    for name, idxs in (("hyb0.5", pool), ("CE", ceord), ("CE→LLM(池50)", llm_order(cache, ceord, 50))):
        d = curve(q, name, idxs)
        print(f"    {name:<14} " + " ｜ ".join(
            f"N={int(r['N'])}: {r['mcover']:.0%}" for _, r in d.iterrows() if r["N"] <= 50))

    # ⑥ N_OUT 与交付数 N 的耦合（**线上 `corpus_search.py` 的真实约束**）
    #    线上：POOL=200 → CE → K_CE=50 交 LLM、**N_OUT=10**（LLM 只输出前 10 名）→ 交付 top-5。
    #    交付 5 篇时 N_OUT=10 够用；**交付 20 篇时，第 11~20 名只能退回 CE 序**（CE 覆盖效率仅 47%）。
    print(f"\n{'=' * 108}\n⑥ **N_OUT 与交付数 N 的耦合**（线上 `K_CE=50`、`N_OUT=10`、交付 top-5）")
    full = llm_order(cache, ceord, 50)
    for n_out in (5, 10, 20, 50):
        out = np.zeros_like(full)
        for i in range(len(ceord)):
            head = list(full[i, :n_out])                 # LLM 给出的前 n_out 名
            seen = set(head)
            tail = [int(x) for x in ceord[i] if int(x) not in seen]   # 其余退回 CE 序
            out[i] = (head + tail)[:full.shape[1]]
        d = curve(q, f"N_OUT={n_out}", out)
        n5, n20 = d[d["N"] == 5].iloc[0], d[d["N"] == 20].iloc[0]
        print(f"  N_OUT={n_out:<3} 交付5 hit {n5['hit']:.1%}（Broad {n5['Broad']:.1%}）"
              f" ｜ 交付20 hit {n20['hit']:.1%}（Broad {n20['Broad']:.1%}）"
              f" ｜ 多gold覆盖@20 {n20['mcover']:.1%}")

    print("\n读法：`Broad hit` 与 `Specific hit` 的**涨幅差** = 『研究方向型查询该多给名额』的证据；"
          "\n      `多gold全覆盖` 只在 ≥2 篇 gold 的题上有意义（样本 %d 题，只能看方向）。" % nm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
