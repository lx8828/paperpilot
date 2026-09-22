"""对给定查询并排展示三个视角的 top-5：**旧语料 / 新语料 / 分数融合**。

用途：直观回答"扩语料到底改变了什么" —— 尤其是那类旧语料**完全答不出**的查询
（"最近的 XX 有哪些"）。

用法：
    python retrieval/scripts/demo_search.py
    python retrieval/scripts/demo_search.py --queries "q1;q2"
    python retrieval/scripts/demo_search.py --topk 5 --alpha 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

from eval_multi_corpus import load_side  # noqa: E402
from eval_retrieval import rrf_fuse  # noqa: E402

OLD = HERE / "data" / "litsearch" / "derived"
NEW = HERE / "data" / "arxiv"

DEFAULT_QUERIES = [
    # 用户最初给的三条真实需求
    "Is there any paper that uses contrastive learning for cross-lingual summarization?",
    "How exactly is the evaluation script written in that SemEval task?",
    "Graph neural networks for recommender systems: what recent work uses iterative refinement?",
    # 两条明显"语料外"的（领域外 + 时间外）
    "Diffusion models for protein design: recent advances?",
    "What are the latest methods for long-context LLM inference?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="", help="分号分隔；默认用内置样例")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--pool", default="100,100", help="融合候选池 q_old,q_new")
    args = ap.parse_args()

    queries = ([q.strip() for q in args.queries.split(";") if q.strip()]
               if args.queries else DEFAULT_QUERIES)

    print("载入索引…")
    m_old, bm_old, _ = load_side(OLD, "corpusid")
    N_OLD = m_old.shape[0]
    m_new, bm_new, _ = load_side(NEW, "docid")
    N_NEW = m_new.shape[0]
    t_old = pd.read_parquet(OLD / "corpus_text.parquet", columns=["title"])
    t_old = t_old[pd.read_parquet(OLD / "corpus_text.parquet",
                                  columns=["text"])["text"] != ""]["title"].astype(str).tolist()
    dn = pd.read_parquet(NEW / "meta" / "corpus_text.parquet",
                         columns=["title", "created", "categories"])
    print(f"  旧 {N_OLD} 篇 / 新 {N_NEW} 篇")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    QV = model.encode(queries, batch_size=8, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    pool_sz = [int(x) for x in args.pool.split(",")]
    a = args.alpha
    for qi, qs in enumerate(queries):
        s_old = m_old @ QV[qi]
        s_new = m_new @ QV[qi]
        r_old = rrf_fuse([np.argsort(-s_old)[:args.depth], bm_old.topk(qs, args.depth)],
                         [a, 1 - a], N_OLD)[:args.depth]
        r_new = rrf_fuse([np.argsort(-s_new)[:args.depth], bm_new.topk(qs, args.depth)],
                         [a, 1 - a], N_NEW)[:args.depth]
        s_all = np.concatenate([s_old, s_new])
        pool = np.array(list(dict.fromkeys(
            r_old[:pool_sz[0]].tolist() + (r_new[:pool_sz[1]] + N_OLD).tolist())), dtype=np.int64)
        fused = pool[np.argsort(-s_all[pool])]

        print("\n" + "=" * 108)
        print(f"Q{qi + 1}: {qs}")
        print("=" * 108)
        for label, ranked in (("旧语料 only", r_old[:args.topk]),
                              ("新语料 only", (r_new + N_OLD)[:args.topk]),
                              ("分数融合 top-%d" % args.topk, fused[:args.topk])):
            print(f"\n  【{label}】")
            for rk, g in enumerate(ranked, 1):
                if g < N_OLD:
                    src, title = "old", t_old[int(g)]
                    extra = ""
                else:
                    j = int(g) - N_OLD
                    src, title = "NEW", dn["title"].iloc[j]
                    extra = f"  [{dn['created'].iloc[j]} {dn['categories'].iloc[j][:22]}]"
                print(f"    {rk}. ({src}) {title[:96]}{extra}")
    print("\n注：`old` = LitSearch 旧语料（NLP 场馆、冻结于 2023）；"
          "`NEW` = arXiv 新语料（2026-07~09 的 CS）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
