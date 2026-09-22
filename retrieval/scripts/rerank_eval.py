"""第 4 步：cross-encoder 精排（在已有 top-k 候选上重排）。

**为什么只重排、不重新召回**：cross-encoder 是 O(候选数) 次前向，不可能扫全库。
标准做法是"召回 → 精排"：这里直接复用 `results/topk.npz`（由 eval_retrieval.py
导出），对某个基线的 top-k 候选逐对打分重排。

**天花板**：精排只能在**已召回集合**内重排 —— 所以 `R@100 = 0.816` 就是上限，
重排不可能把 R@1 提到 0.816 以上。这也是"选 α 要看下游"的原因（有精排 → 选 α=0.5 保 R@100）。

工程要点：
  · **增量保存**：每 50 题存一次 `rerank_scores.npy`，中断后重跑只补缺的部分；
  · 复用 `significance.py` 的 McNemar / 配对 bootstrap，结论直接带显著性。

用法：
    python retrieval/scripts/rerank_eval.py --limit-queries 20     # 冒烟测吞吐
    python retrieval/scripts/rerank_eval.py                        # 全量
    python retrieval/scripts/rerank_eval.py --base hyb0.7 --topk 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))          # 复用显著性检验
DERIVED = HERE / "data" / "litsearch" / "derived"
RESULTS = HERE / "results"
CORPUS = DERIVED / "corpus_text.parquet"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
TOPK = RESULTS / "topk.npz"
SCORES = RESULTS / "rerank_scores.npy"
MODEL = "BAAI/bge-reranker-v2-m3"

BLOCK = 50


def recall_at(ranked: np.ndarray, gold: set[int], k: int) -> float:
    return (len(gold & set(ranked[:k].tolist())) / len(gold)) if gold else 0.0


def mrr_at(ranked: np.ndarray, gold: set[int], k: int) -> float:
    for i, d in enumerate(ranked[:k].tolist(), 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hyb0.5", help="要重排的基线系统（topk.npz 的键）")
    ap.add_argument("--topk", type=int, default=100, help="重排多少个候选")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--topk-file", default="topk.npz",
                    help="候选文件（results/ 下）。扩池子后用 topk1000.npz")
    ap.add_argument("--out-scores", default="rerank_scores.npy",
                    help="分数输出（results/ 下）。**跑探针务必换名**，否则会覆盖正式结果")
    ap.add_argument("--out-md", default="LITSEARCH_RERANK.md",
                    help="报告文件（results/ 下）。**跑探针务必换名**，否则会覆盖正式报告")
    ap.add_argument("--depth-sweep", default="",
                    help="打分深度用 --topk，然后对这些 K_cand 切前缀评估（逗号分隔）。"
                         "一次打分覆盖所有深度 —— 分数按 (题, 候选) 缓存，换深度不重算")
    args = ap.parse_args()
    topk_path = RESULTS / args.topk_file
    scores_path = RESULTS / args.out_scores

    if not topk_path.exists():
        print(f"缺 {topk_path}\n请先运行：python retrieval/scripts/eval_retrieval.py "
              f"--alpha 0.3,0.5,0.7,0.9 --dump-topk")
        return 2

    # ── 数据 ────────────────────────────────────────────────
    dfc = pd.read_parquet(CORPUS, columns=["corpusid", "text"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    texts = dfc["text"].tolist()
    q = pd.read_parquet(QFILE)
    golds = [[int(x) for x in v] for v in q["corpusids"]]
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    gold_rows = [{id2row[g] for g in gs if g in id2row} for gs in golds]

    z = np.load(topk_path)
    if args.base not in z.files:
        print(f"{args.topk_file} 里没有 `{args.base}`；可选：{list(z.files)}")
        return 2
    cand = z[args.base][:, :args.topk]                       # (n_q, topk)
    n_q = len(q)
    if args.limit_queries:
        cand = cand[:args.limit_queries]
        n_q = args.limit_queries
        gold_rows = gold_rows[:n_q]
    print(f"基线 {args.base} | 重排 {n_q} 题 × top-{args.topk} = {n_q * args.topk:,} 对")

    # ── 模型 ────────────────────────────────────────────────
    import torch
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 {MODEL}（device={dev}）…")
    t0 = time.time()
    ce = CrossEncoder(MODEL, max_length=args.max_length, device=dev)
    if dev == "cuda":
        try:
            ce.model.half()
            print("  已切 fp16")
        except Exception as e:  # noqa: BLE001
            print(f"  fp16 失败，用 fp32：{type(e).__name__}")
    print(f"  加载完成 {time.time() - t0:.1f}s")

    # ── 增量打分 ────────────────────────────────────────────
    done = 0
    if scores_path.exists():
        old = np.load(scores_path)
        if old.shape == (n_q, args.topk):
            print(f"发现已算好的分数 {scores_path.name}（{old.shape}）→ 直接复用")
            sc = old
            done = n_q
    if done == 0:
        sc = np.full((n_q, args.topk), np.nan, dtype=np.float32)
        t_all = time.time()
        for lo in range(0, n_q, BLOCK):
            hi = min(lo + BLOCK, n_q)
            t0 = time.time()
            pairs, owners = [], []
            for i in range(lo, hi):
                qt = str(q.loc[i, "query"])
                for j, r in enumerate(cand[i]):
                    pairs.append((qt, texts[int(r)]))
                    owners.append((i - lo, j))
            s = np.asarray(ce.predict(pairs, batch_size=args.batch,
                                      show_progress_bar=False), dtype=np.float32)
            for (ii, jj), v in zip(owners, s):
                sc[lo + ii, jj] = v
            el = time.time() - t0
            left = n_q - hi
            rate = len(pairs) / max(el, 1e-9)                  # 对/秒
            # ⚠️ ETA 要用"剩余对数"= 剩余题数 × topk，别忘了乘 args.topk
            print(f"  题 {lo:>4}-{hi:<4} {el:5.1f}s ({rate:6.0f} 对/s) "
                  f"剩余约 {left * args.topk / rate / 60:5.1f} 分钟"
                  + (f"  [cuda {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB]"
                     if dev == "cuda" else ""), flush=True)
            np.save(scores_path, sc)                # 增量落盘，可中断续跑
            done = hi
        print(f"打分完成，用时 {time.time() - t_all:.1f}s")
    sc = np.nan_to_num(sc, nan=-1e9)

    # ── 深度扫描：一次打分覆盖所有 K_cand ────────────────────────
    # 打分是按 (题, 候选) 存的，所以切列前缀即可评估任意深度，无需重算。
    if args.depth_sweep:
        depths = sorted({int(x) for x in args.depth_sweep.split(",") if x.strip()})
        depths = [d for d in depths if d <= args.topk]
        ceil = {d: float(np.mean([recall_at(cand[i], gold_rows[i], d) for i in range(n_q)]))
                for d in depths}
        rows = []
        for d in depths:
            rl = [cand[i][:d][np.argsort(-sc[i][:d])] for i in range(n_q)]
            row = {"K_cand": d, "池子上界R@K": ceil[d]}
            for k in (1, 5, 10, 20):
                row[f"R@{k}"] = float(np.mean([recall_at(rl[i], gold_rows[i], k)
                                               for i in range(n_q)]))
            row["MRR@10"] = float(np.mean([mrr_at(rl[i], gold_rows[i], 10)
                                           for i in range(n_q)]))
            # 精排吃掉了多少"可用的排序空间" = (R@1_rerank − R@1_池) / (上界 − R@1_池)
            base1 = float(np.mean([recall_at(cand[i], gold_rows[i], 1) for i in range(n_q)]))
            denom = max(ceil[d] - base1, 1e-9)
            row["吃掉排序空间%"] = 100 * (row["R@1"] - base1) / denom
            row["成本(万对)"] = round(n_q * d / 10000, 1)
            rows.append(row)
        t = pd.DataFrame(rows)
        print("\n" + "=" * 96)
        print(f"★ 精排深度扫描（{args.base}，{n_q} 题，模型 {MODEL.split('/')[-1]}，"
              f"max_length={args.max_length}）")
        print("=" * 96)
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n  读法：'池子上界R@K' 是精排 R@1 的天花板；")
        print("        若 R@1 在某个 K_cand 之后不再上升 → 池子加再深也不划算；")
        print("        '吃掉排序空间%' 是精排把'可用的排序空间'利用了多少。")
        from eval_retrieval import md_table      # 本分支在下方 return，import 必须放在这里
        md = RESULTS / args.out_md
        md.write_text(
            f"# LitSearch · 精排深度扫描（{time.strftime('%Y%m%d_%H%M%S')}）\n\n"
            f"- 基线 `{args.base}` 的候选来自 `results/{args.topk_file}`\n"
            f"- 精排模型 `{MODEL}`，`max_length={args.max_length}`，batch={args.batch}\n"
            f"- 题数 {n_q}；**一次打分覆盖所有深度**（分数按 (题, 候选) 缓存）\n"
            f"- `池子上界R@K` = 未精排时 gold 落在前 K 的比例 = **精排 R@1 的理论上界**\n\n"
            + md_table(t) + "\n", encoding="utf-8")
        print(f"\n已写出：{md.relative_to(HERE)}")
        return 0


    # ── 重排 → 指标 ─────────────────────────────────────────
    reranked = [cand[i][np.argsort(-sc[i])] for i in range(n_q)]
    base_ranked = [cand[i] for i in range(n_q)]

    def block(ranked_list: list[np.ndarray]) -> dict:
        out = {}
        for k in (1, 5, 10, 100):
            out[f"R@{k}"] = float(np.mean([recall_at(ranked_list[i], gold_rows[i], k)
                                           for i in range(n_q)]))
        out["MRR@10"] = float(np.mean([mrr_at(ranked_list[i], gold_rows[i], 10)
                                       for i in range(n_q)]))
        out["n"] = n_q
        return out

    rows = [{"配置": f"{args.base}（重排前）", **block(base_ranked)},
            {"配置": f"{args.base} + rerank({MODEL.split('/')[-1]})", **block(reranked)}]
    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"精排效果（{n_q} 题，重排 top-{args.topk}）")
    print("=" * 80)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 显著性（复用 significance.py）────────────────────────
    from eval_retrieval import md_table
    from significance import mcnemar, paired_bootstrap
    print("\n[ 显著性：rerank vs 重排前 ]")
    for k in (1, 5, 10):
        a = np.array([1.0 if recall_at(reranked[i], gold_rows[i], k) == 1.0 else 0.0
                      for i in range(n_q)])
        b = np.array([1.0 if recall_at(base_ranked[i], gold_rows[i], k) == 1.0 else 0.0
                      for i in range(n_q)])
        oa, ob = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        pv = mcnemar(oa, ob)
        print(f"  hit@{k:<3}：仅rerank命中 {oa:3d} | 仅基线命中 {ob:3d} | 净 Δ={oa - ob:+4d} | "
              f"p={pv:.4g}  {'**显著**' if pv < 0.05 else '不显著'}")

    for metric, f in (("recall@1", lambda i, L: recall_at(L[i], gold_rows[i], 1)),
                      ("mrr@10", lambda i, L: mrr_at(L[i], gold_rows[i], 10))):
        x = np.array([f(i, reranked) for i in range(n_q)])
        y = np.array([f(i, base_ranked) for i in range(n_q)])
        d, lo, hi, pv = paired_bootstrap(x, y)
        print(f"  {metric:<9}：Δ={d:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  p≈{pv:.4g}  "
              f"{'**显著**' if (lo > 0 or hi < 0) else '不显著'}")

    # ── 落盘 ────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    md = RESULTS / args.out_md
    md.write_text(
        f"# LitSearch · cross-encoder 精排（{time.strftime('%Y%m%d_%H%M%S')}）\n\n"
        f"- 基线：`{args.base}` 的 top-{args.topk} 候选（来自 `results/{args.topk_file}`）\n"
        f"- 精排模型：`{MODEL}`\n"
        f"- 题数：{n_q}；**精排上限 = 基线的 R@{args.topk}**（重排无法召回新文档）\n\n"
        + md_table(tbl) + "\n", encoding="utf-8")
    print(f"\n已写出：{md.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
