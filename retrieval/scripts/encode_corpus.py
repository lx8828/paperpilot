"""第 2 步：把 63,269 篇的 title+abstract 编码成向量矩阵。

**没有数据库**：输入是一个 parquet 文件，输出是一批 `.npy` 文件。
索引就是"一个 (N, 1024) 的 float32 矩阵"——63,269 × 1024 × 4B ≈ 259 MB，
一次 matmul 就能算完全部相似度，不需要任何向量库/服务。

设计要点：
  · 模型 `BAAI/bge-m3`（本地已缓存，与 paperpilot 生产同源）
  · **归一化**向量 → 点积即 cosine（与 `paperpilot.agents.embedder.ChunkIndex` 口径一致）
  · **分块增量存盘**（默认 5000 条/块）→ 中断后重跑只补缺的块
  · 支持 `--limit N` 先做小样本冒烟（验证 + 测吞吐，再决定要不要跑全量）

产物（`retrieval/data/litsearch/derived/emb/`）：
    corpusid.npy      (N,) int64        文档顺序，与向量行一一对应
    part_00000.npy    (5000, 1024) fp32 分块向量
    manifest.json     模型 / 维度 / max_length / 块清单 / 耗时

用法：
    python retrieval/scripts/encode_corpus.py --limit 256   # 冒烟（先跑这个）
    python retrieval/scripts/encode_corpus.py               # 全量 63,269 篇
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DERIVED = HERE / "data" / "litsearch" / "derived"
CORPUS = DERIVED / "corpus_text.parquet"
OUT = DERIVED / "emb"

MODEL_NAME = "BAAI/bge-m3"
MAX_LEN = 512
PART = 5000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只编码前 N 篇（冒烟用）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--part", type=int, default=PART, help="每块存多少条")
    ap.add_argument("--max-length", type=int, default=MAX_LEN)
    ap.add_argument("--fp32", action="store_true", help="关掉 fp16（默认用 fp16 加速）")
    args = ap.parse_args()

    if not CORPUS.exists():
        print(f"缺 {CORPUS}\n请先运行：python retrieval/scripts/build_corpus.py")
        return 2

    df = pd.read_parquet(CORPUS, columns=["corpusid", "text", "n_chars"])
    df = df[df["text"] != ""].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    n = len(df)
    print(f"待编码 {n} 篇（无文本的已排除）")

    OUT.mkdir(parents=True, exist_ok=True)
    n_parts = (n + args.part - 1) // args.part

    # 已存在且完整的块 → 跳过（可中断续跑）
    todo = []
    for p in range(n_parts):
        f = OUT / f"part_{p:05d}.npy"
        lo, hi = p * args.part, min((p + 1) * args.part, n)
        if f.exists():
            try:
                if np.load(f, mmap_mode="r").shape[0] == hi - lo:
                    continue
            except Exception:  # noqa: BLE001  损坏的块重算
                pass
        todo.append(p)
    print(f"分 {n_parts} 块，其中需计算 {len(todo)} 块")

    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n加载 {MODEL_NAME}（device={dev}）…")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=dev, trust_remote_code=True)
    model.max_seq_length = args.max_length
    use_fp16 = (dev == "cuda") and not args.fp32
    if use_fp16:
        model.half()
    print(f"  加载完成 {time.time() - t0:.1f}s | max_seq_length={model.max_seq_length}"
          f" | fp16={use_fp16}")

    texts = df["text"].tolist()
    total_chars = int(df["n_chars"].sum())
    trunc_chars = int((df["n_chars"] > args.max_length * 3.2).sum())
    print(f"  文本合计 {total_chars / 1e6:.1f} M 字符 | 可能被截断(> {int(args.max_length * 3.2)} 字符): "
          f"{trunc_chars} 篇 ({100 * trunc_chars / n:.1f}%)")

    print()
    t_all = time.time()
    done_rows = 0
    for p in todo:
        lo, hi = p * args.part, min((p + 1) * args.part, n)
        t0 = time.time()
        emb = model.encode(
            texts[lo:hi],
            batch_size=args.batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        np.save(OUT / f"part_{p:05d}.npy", emb)
        done_rows += hi - lo
        el = time.time() - t0
        rate = (hi - lo) / el
        left = sum(min((q + 1) * args.part, n) - q * args.part for q in todo if q > p)
        print(f"  [part {p + 1:>3}/{n_parts}] 行 {lo:>6}-{hi:<6} {el:5.1f}s "
              f"({rate:6.0f} 行/s)  剩余约 {left / rate / 60:5.1f} 分钟"
              + ("  [cuda %.2fGB]" % (torch.cuda.max_memory_allocated() / 1024 ** 3)
                 if dev == "cuda" else ""))

    # ── 汇总：把所有块读成一个矩阵（若已全部存在则直接加载）──────
    parts = [np.load(OUT / f"part_{p:05d}.npy") for p in range(n_parts)]
    mat = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    np.save(OUT / "corpusid.npy", df["corpusid"].to_numpy(dtype=np.int64))

    manifest = {
        "model": MODEL_NAME,
        "dim": int(mat.shape[1]),
        "n_docs": int(mat.shape[0]),
        "normalize": True,
        "max_seq_length": args.max_length,
        "fp16": use_fp16,
        "batch": args.batch,
        "part_size": args.part,
        "n_parts": n_parts,
        "truncated_est": trunc_chars,
        "elapsed_s": round(time.time() - t_all, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"\n完成：矩阵 {mat.shape} dtype={mat.dtype} "
          f"≈ {mat.nbytes / 1e6:.0f} MB | 用时 {time.time() - t_all:.1f}s")
    print(f"  {OUT.relative_to(HERE)}/corpusid.npy  +  part_*.npy  +  manifest.json")
    print(f"  自检：范数均值 {np.linalg.norm(mat[:min(1000, len(mat))], axis=1).mean():.4f}"
          f"（归一化后应≈1.0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
