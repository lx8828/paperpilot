"""免费预测试：**粗筛能否用廉价表示？**（纯 GPU，无 API 成本）

背景：K 扫描证明 `R@1` 在 K≥20 后饱和，多花钱只能买 `R@20`。而我们的过滤器
（cross-encoder top-20）的**筛选质量**是上界的瓶颈：
    池子前 20 = 69.8%  <  ce 全文 top-20 = 74.3%  <  LLM 看 100 篇选 20 = 81.3%
所以「怎么把 top-100 筛成 20」是关键。但"让 LLM 看 100 篇全文再筛"比 K=100 单次还贵
（18.8K + 4.1K = 22.9K token），只有在**粗筛能用廉价表示**时才划算。

本脚本用 cross-encoder（免费）做同一件事的代理实验：
    把每篇候选的表示换成 `title` / `short`(标题+前100字符)，重跑 top-100，
    看「gold 落在前 K 的比例」掉多少。

判读：
    · 若 `title` 的 R@20 ≈ 全文（74.3%）→ **标题足够做筛选**
      → 「LLM 看 100 篇标题筛 20 → 20 篇全文精排」路线成立（预计 ~5,982 token/查询，
        比 K=20 只贵 47%，却可能把筛选上界提到 81.3%）
    · 若远低于 → 廉价表示筛选不了 → 该路线放弃，省下后续 API 成本

注意：cross-encoder 只是 **代理**（它和 LLM 不是同一模型），本测试衡量的是
「信息是否**足够**区分」，而不是「LLM 会用这个信息」。

用法：
    python retrieval/scripts/screen_cheap_probe.py
    python retrieval/scripts/screen_cheap_probe.py --views title
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
CORPUS = DERIVED / "corpus_text.parquet"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
TOPK = RESULTS / "topk1000.npz"
CE_FULL = RESULTS / "rerank1000.npy"
MODEL = "BAAI/bge-reranker-v2-m3"

POOL = 100           # 候选池深度（重排范围）
KS = (1, 5, 10, 20, 50, 100)
BLOCK = 50

# 廉价表示：给定 (title, text) 产出喂给 cross-encoder 的文档串
VIEWS = {
    "title": lambda t, x: t,
    "short": lambda t, x: f"{t} {x[:100]}",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="title,short")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    args = ap.parse_args()
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    q = pd.read_parquet(QFILE)
    ids = np.load(DERIVED / "emb" / "corpusid.npy")
    id2row = {int(c): i for i, c in enumerate(ids)}
    golds = [{id2row[g] for g in row if int(g) in id2row} for row in q["corpusids"]]
    n = len(golds)
    if args.limit:
        n = min(n, args.limit)
        golds = golds[:n]

    dfc = pd.read_parquet(CORPUS, columns=["corpusid", "text", "title"])
    dfc = dfc[dfc["text"] != ""].reset_index(drop=True)
    titles = dfc["title"].astype(str).tolist()
    texts = dfc["text"].astype(str).tolist()

    pool = np.load(TOPK)["hyb0.5"][:, :POOL]
    ce_full = np.load(CE_FULL)[:, :POOL]
    print(f"查询 {n} 条 | 候选池 top-{POOL} | 表示 {views}")

    def gold_in_topk(scores: np.ndarray, kk: int) -> float:
        """按 scores 重排后，gold 落在前 kk 的比例（kk=POOL 时即'池子里有没有'）。"""
        vals = []
        for i in range(n):
            order = pool[i][np.argsort(-scores[i])][:kk]
            g = golds[i]
            vals.append(len(g & set(order.tolist())) / len(g) if g else 0.0)
        return float(np.mean(vals))

    import torch
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 {MODEL}（device={dev}）…")
    ce = CrossEncoder(MODEL, max_length=512, device=dev)
    if dev == "cuda":
        ce.model.half()

    rows = [{"表示": "full（现用，基线）", **{f"gold@top{K}": gold_in_topk(ce_full, K)
                                              for K in KS}}]
    for view in views:
        out = RESULTS / f"ce_{view}_scores.npy"
        if out.exists() and np.load(out).shape == (n, POOL):
            sc = np.load(out)
            print(f"[{view}] 复用已算好的分数 {out.name}")
        else:
            fmt = VIEWS[view]
            sc = np.full((n, POOL), np.nan, dtype=np.float32)
            t_all = time.time()
            for lo in range(0, n, BLOCK):
                hi = min(lo + BLOCK, n)
                pairs = []
                for i in range(lo, hi):
                    qt = str(q.loc[i, "query"])
                    for r in pool[i]:
                        rr = int(r)
                        pairs.append((qt, fmt(titles[rr], texts[rr])))
                s = np.asarray(ce.predict(pairs, batch_size=args.batch,
                                          show_progress_bar=False), dtype=np.float32)
                sc[lo:hi] = s.reshape(hi - lo, POOL)
                el = time.time() - t_all
                done = hi * POOL
                print(f"  [{view}] 题 {lo:>4}-{hi:<4} {el:5.1f}s "
                      f"({done / max(el, 1e-9):5.0f} 对/s) 剩余约 "
                      f"{(n * POOL - done) / max(done / max(el, 1e-9), 1e-9) / 60:4.1f} 分钟",
                      flush=True)
                np.save(out, sc)
            print(f"  [{view}] 完成，用时 {time.time() - t_all:.0f}s")
        rows.append({"表示": view, **{f"gold@top{K}": gold_in_topk(sc, K) for K in KS}})

    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("筛选质量：gold 落在重排后前 K 的比例（= 用该表示做筛选时的上界）")
    print("=" * 88)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    base = rows[0]
    print("\n[ 判读 ] 与 full 基线的差值（负 = 掉多少）：")
    for r in rows[1:]:
        d20 = r["gold@top20"] - base["gold@top20"]
        d50 = r["gold@top50"] - base["gold@top50"]
        print(f"  {r['表示']:<8} top-20: {d20:+.4f}   top-50: {d50:+.4f}")
    print("\n  · 若 top-20 掉得很少（比如 < 2pt）→ **廉价表示足够做筛选**，")
    print("    「LLM 看 100 篇廉价表示筛 20 → 全文精排」路线值得投入（约 $2.5）；")
    print("  · 若掉得多 → 该路线放弃，转而改进 cross-encoder 本身（换模型 / 分数融合）。")

    from eval_retrieval import md_table
    (RESULTS / "LITSEARCH_SCREEN_PROBE.md").write_text(
        f"# 廉价表示能否用于粗筛？（免费预测试，{time.strftime('%Y%m%d_%H%M%S')}）\n\n"
        f"- 用 cross-encoder 作代理：把候选表示换成 title/short，重跑 top-{POOL}\n"
        f"- 指标 = gold 落在重排后前 K 的比例（= 用该表示做筛选时的上界）\n\n"
        + md_table(tbl) + "\n", encoding="utf-8")
    print(f"\n已写出：results/LITSEARCH_SCREEN_PROBE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
