"""用户 one-hop 设计的**免费 GPU 验证**：CE 第二趟能否把引用邻居排进前 20？

设计（用户版）：

    第一趟：hyb0.5 top-100 → CE 打分 → 取 CE 的 **top-20 作为 seed**
    扩展  ：seed ∪ (seed 的 outgoing references ∩ 语料)   ← 平均 132.6 篇
    第二趟：CE 对新池**重新打分** → 取 **top-20** → 交给 LLM（LLM 输入仍是 20 篇，成本不变）

**为什么必须实测**：池子从 20 涨到 132.6，引用邻居会与原 20 篇竞争 ——
若 CE 给某个邻居更高分，**原本对的可能被挤出前 20**。
所以「覆盖率 0.8350」只是上界，真实 gold@20 必须算。

对照：
    ce top-20（现用）  gold@20 = 0.7430   ← 基线
    ce top-100         gold@20 = 0.8350   ← 上界参照（但 LLM 成本 4.7×）
    ce top-20 ∪ 邻居    gold@20 = ?        ← 本脚本要算的

纯 GPU，无 API 成本。用法：
    python retrieval/scripts/ce_onehop_eval.py
    python retrieval/scripts/ce_onehop_eval.py --seed-k 20 --limit 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DERIVED = HERE / "data" / "litsearch" / "derived"
DATA = HERE / "data" / "litsearch"
QFILE = DATA / "query" / "full-00000-of-00001.parquet"
CORPUS = DERIVED / "corpus_text.parquet"
TOPK = RESULTS / "topk1000.npz"
CE_SCORES = RESULTS / "rerank1000.npy"
MODEL = "BAAI/bge-reranker-v2-m3"
POOL = 100
KS = (1, 5, 10, 20, 50)
BLOCK = 25          # 每次处理的查询数（池大小不均，按查询块推进）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-k", type=int, default=20, help="用来扩展的 CE top-K seed")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    q = pd.read_parquet(QFILE)
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    n = len(golds)
    if args.limit:
        n = min(n, args.limit)
        golds = golds[:n]

    dfc = pd.read_parquet(CORPUS, columns=["corpusid", "text"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    texts = dfc["text"].astype(str).tolist()

    cit: dict[int, np.ndarray] = {}
    for p in sorted((DATA / "corpus_clean").glob("*.parquet")):
        d = pd.read_parquet(p, columns=["corpusid", "citations"])
        for cid, cs in zip(d["corpusid"].to_numpy(), d["citations"].to_numpy()):
            cit[int(cid)] = cs if cs is not None else np.array([], dtype=np.int64)

    pool = np.load(TOPK)["hyb0.5"][:, :POOL]
    ce0 = np.load(CE_SCORES)[:, :POOL]
    ce_order = [[int(x) for x in pool[i][np.argsort(-ce0[i])]] for i in range(n)]

    # ── 构造新池 ────────────────────────────────────────────
    newpools: list[list[int]] = []
    n_neigh = []
    for i in range(n):
        seeds = ce_order[i][:args.seed_k]
        seen = set(seeds)
        out = list(seeds)
        for r in seeds:
            for c in cit.get(int(ids[r]), np.array([], dtype=np.int64)):
                row = id2row.get(int(c))
                if row is not None and row not in seen:
                    seen.add(row)
                    out.append(row)
        newpools.append(out)
        n_neigh.append(len(out) - len(seeds))
    sizes = np.array([len(p) for p in newpools])
    print(f"查询 {n} 条 | seed-K={args.seed_k} | 新池平均 {sizes.mean():.1f} 篇"
          f"（p50={np.median(sizes):.0f} p90={np.percentile(sizes, 90):.0f} "
          f"max={sizes.max()}）| 其中引用邻居平均 {np.mean(n_neigh):.1f} 篇")
    print(f"CE 需打分 {sizes.sum():,} 对")

    out_cache = RESULTS / f"ce_onehop_k{args.seed_k}.npz"
    if out_cache.exists():
        z = np.load(out_cache, allow_pickle=True)
        if len(z["rows"]) == n:
            print(f"复用已算好的分数 {out_cache.name}")
            rows_arr, sc_arr = z["rows"], z["scores"]
        else:
            rows_arr = sc_arr = None
    else:
        rows_arr = sc_arr = None

    if rows_arr is None:
        import torch
        from sentence_transformers import CrossEncoder
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载 {MODEL}（device={dev}）…")
        ce = CrossEncoder(MODEL, max_length=512, device=dev, trust_remote_code=False)
        if dev == "cuda":
            ce.model.half()

        rows_arr = np.empty(n, dtype=object)
        sc_arr = np.empty(n, dtype=object)
        t_all = time.time()
        done = 0
        for lo in range(0, n, BLOCK):
            hi = min(lo + BLOCK, n)
            pairs, owners = [], []
            for i in range(lo, hi):
                qt = str(q.loc[i, "query"])
                for r in newpools[i]:
                    pairs.append((qt, texts[r]))
                    owners.append(i)
            s = np.asarray(ce.predict(pairs, batch_size=args.batch,
                                      show_progress_bar=False), dtype=np.float32)
            for i in range(lo, hi):
                sc_arr[i] = np.array([], dtype=np.float32)
                rows_arr[i] = np.array(newpools[i], dtype=np.int32)
            k = 0
            for i in owners:
                sc_arr[i] = np.append(sc_arr[i], s[k])
                k += 1
            done += len(pairs)
            el = time.time() - t_all
            print(f"  题 {lo:>4}-{hi:<4} {el:5.1f}s ({done / max(el, 1e-9):5.0f} 对/s) "
                  f"剩余约 {(sizes.sum() - done) / max(done / max(el, 1e-9), 1e-9) / 60:4.1f} 分钟",
                  flush=True)
            np.savez_compressed(out_cache, rows=rows_arr, scores=sc_arr)
        print(f"完成，用时 {time.time() - t_all:.0f}s")

    # ── 指标 ────────────────────────────────────────────────
    def gold_atk(ranked_lists: list[list[int]], kk: int) -> float:
        vals = []
        for i in range(n):
            order = ranked_lists[i][:kk]
            g = golds[i]
            vals.append(len(g & set(order)) / len(g) if g else 0.0)
        return float(np.mean(vals))

    base20 = [[int(x) for x in ce_order[i][:20]] for i in range(n)]
    base100 = [[int(x) for x in ce_order[i][:100]] for i in range(n)]
    newrank = [[int(rows_arr[i][j]) for j in np.argsort(-sc_arr[i])] for i in range(n)]

    print("\n" + "=" * 92)
    print("结果：CE 第二趟（池子 = ce top-20 ∪ 引用邻居）")
    print("=" * 92)
    print(f"  {'方案':<44}" + "".join(f"{'gold@top' + str(k):>12}" for k in KS))
    for name, rl in (("ce top-20（现用基线）", base20),
                     (f"one-hop：ce top-{args.seed_k} + 引用邻居 → CE 重排", newrank),
                     ("ce top-100（上界参照，LLM 成本 4.7×）", base100)):
        print(f"  {name:<44}" + "".join(f"{gold_atk(rl, k):>12.4f}" for k in KS))

    d20 = gold_atk(newrank, 20) - gold_atk(base20, 20)
    print(f"\n  **关键：gold@20 变化 = {d20:+.4f}**（{gold_atk(base20, 20):.4f} → "
          f"{gold_atk(newrank, 20):.4f}）")
    print(f"  对照：ce top-100 的 gold@20 = {gold_atk(base100, 20):.4f}"
          f"（LLM 成本 4.7×）")

    print("\n[ 判读 ]")
    print("  · gold@20 明显 > 0.7430 → **用户的 one-hop 成立**：免费 CE 通道换来更高上界，")
    print("    而 LLM 成本不变 → 严格优于「直接扩到 top-100」（后者成本 4.7×）；")
    print("  · 若 ≈ 或 < 0.7430 → 引用邻居**挤掉了原本对的**，需要加保护（如强制保留原 top-20）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
