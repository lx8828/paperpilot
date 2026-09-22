"""第 2 步（arXiv 侧）：把 arXiv 元数据建成与 LitSearch **完全同口径**的检索索引。

为什么必须同口径：fusion 时要和现有 63k 索引一起用（RRF / quota_union），
如果 tokenizer、归一化、向量维度、BM25 公式有任何一处不同，融合就是错的。
所以本脚本**直接 import** 现有实现，而不是另写一套：
  · BM25   → `eval_retrieval.SparseBM25`（同一 tokenize / k1 / b / min_df）
  · 向量   → 沿用 `encode_corpus.py` 的口径（bge-m3 / max_len 512 / **归一化** / fp16 加速 / 分块落盘）

产物（与 litsearch 侧结构对称）：
    data/arxiv/meta/corpus_text.parquet   docid/title/abstract/text/n_chars/arxiv_id/created/categories
    data/arxiv/emb/part_XXXXX.npy         分块向量 (part, 1024) fp32（归一化）
    data/arxiv/emb/docid.npy              行号→docid
    data/arxiv/emb/manifest.json          模型/维度/块清单/口径
    data/arxiv/bm25/{X.npz,stats.npz,vocab.json,params.json}

用法：
    python retrieval/scripts/build_arxiv_index.py --limit 2000   # 冒烟
    python retrieval/scripts/build_arxiv_index.py               # 全量
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
sys.path.insert(0, str(HERE / "scripts"))
from _hfcache import ensure_hf_home  # noqa: E402
from eval_retrieval import SparseBM25  # noqa: E402  同一 BM25 实现（口径保证）

ensure_hf_home()          # 必须在导入 sentence_transformers 之前

ARXIV = HERE / "data" / "arxiv"
SRC = ARXIV / "meta" / "arxiv_meta.parquet"
META = ARXIV / "meta" / "corpus_text.parquet"
EMB = ARXIV / "emb"
BM25DIR = ARXIV / "bm25"

MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 512          # 与 litsearch 侧一致
PART = 5000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--part", type=int, default=PART)
    ap.add_argument("--max-length", type=int, default=MAX_LEN)
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"缺 {SRC}\n请先运行：python retrieval/scripts/fetch_arxiv.py --target 50000")
        return 2

    df = pd.read_parquet(SRC)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    # ── 构语料：与 litsearch 侧同构的 text = title + "\n" + abstract ──
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df = df[(df["title"] != "") & (df["abstract"] != "")].reset_index(drop=True)
    df["text"] = df["title"] + "\n" + df["abstract"]
    df["n_chars"] = df["text"].str.len()
    df.insert(0, "docid", np.arange(len(df), dtype=np.int64))

    META.parent.mkdir(parents=True, exist_ok=True)
    keep = ["docid", "arxiv_id", "title", "abstract", "text", "n_chars",
            "created", "updated", "categories", "comments", "authors", "license"]
    df[keep].to_parquet(META, index=False)
    n = len(df)
    print(f"语料 {n} 篇 → {META.relative_to(HERE)}")
    print(f"  text 长度：p50={int(df['n_chars'].median())} "
          f"p90={int(df['n_chars'].quantile(0.9))} max={int(df['n_chars'].max())}")
    print(f"  年份：{df['created'].str[:4].value_counts().sort_index().to_dict()}")

    # ── 向量（口径与 encode_corpus.py 一致）─────────────────────
    if not args.skip_encode:
        import torch
        from sentence_transformers import SentenceTransformer

        EMB.mkdir(parents=True, exist_ok=True)
        n_parts = (n + args.part - 1) // args.part
        todo = []
        for p in range(n_parts):
            f = EMB / f"part_{p:05d}.npy"
            lo, hi = p * args.part, min((p + 1) * args.part, n)
            if f.exists():
                try:
                    if np.load(f, mmap_mode="r").shape[0] == hi - lo:
                        continue
                except Exception:  # noqa: BLE001
                    pass
            todo.append(p)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n加载 {MODEL_NAME}（device={dev}）…")
        model = SentenceTransformer(MODEL_NAME, device=dev, trust_remote_code=True)
        model.max_seq_length = args.max_length
        use_fp16 = (dev == "cuda")
        if use_fp16:
            model.half()
        print(f"  分 {n_parts} 块，需计算 {len(todo)} 块 | max_len={args.max_length} fp16={use_fp16}")
        texts = df["text"].tolist()
        t_all = time.time()
        for p in todo:
            lo, hi = p * args.part, min((p + 1) * args.part, n)
            t0 = time.time()
            emb = model.encode(texts[lo:hi], batch_size=args.batch,
                               normalize_embeddings=True, show_progress_bar=False,
                               convert_to_numpy=True).astype("float32")
            np.save(EMB / f"part_{p:05d}.npy", emb)
            el, rate = time.time() - t0, 0.0
            rate = (hi - lo) / max(el, 1e-9)
            left = sum(min((q + 1) * args.part, n) - q * args.part for q in todo if q > p)
            print(f"  [part {p + 1:>3}/{n_parts}] 行 {lo:>6}-{hi:<6} {el:5.1f}s "
                  f"({rate:6.0f} 行/s) 剩余约 {left / rate / 60:5.1f} 分钟"
                  + ("  [cuda %.2fGB]" % (torch.cuda.max_memory_allocated() / 1024 ** 3)
                     if dev == "cuda" else ""))
        np.save(EMB / "docid.npy", df["docid"].to_numpy(dtype=np.int64))
        parts = [np.load(EMB / f"part_{p:05d}.npy") for p in range(n_parts)]
        mat = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        (EMB / "manifest.json").write_text(json.dumps({
            "model": MODEL_NAME, "dim": int(mat.shape[1]), "n_docs": int(mat.shape[0]),
            "normalize": True, "max_seq_length": args.max_length, "fp16": use_fp16,
            "part_size": args.part, "n_parts": n_parts,
            "elapsed_s": round(time.time() - t_all, 1),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "arxiv_meta.parquet",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n向量完成：{mat.shape} {mat.dtype} ≈ {mat.nbytes / 1e6:.0f} MB | "
              f"用时 {time.time() - t_all:.0f}s")
        print(f"  自检：范数均值 {np.linalg.norm(mat[:min(1000, len(mat))], axis=1).mean():.4f}"
              f"（归一化后应≈1.0）")

    # ── BM25（同一实现）─────────────────────────────────────────
    if not args.skip_bm25:
        t0 = time.time()
        print(f"\n建 BM25（{n} 篇）…")
        bm = SparseBM25(df["text"].tolist())
        bm.save(BM25DIR)
        print(f"  完成 {time.time() - t0:.1f}s | 矩阵 {bm.X.shape} 非零 {bm.X.nnz / 1e6:.1f}M "
              f"| avgdl={bm.avgdl:.0f} | vocab={len(bm.vocab)}")
        print(f"  → {BM25DIR.relative_to(HERE)}")

    print("\n完成。目录：")
    for p in (META, EMB, BM25DIR):
        print(f"  {p.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
