r"""验证 `encode_texts` 的**长度排序再批**改动：无回归、有效果。

## 三条腿
| # | 验什么 | 方法 | GPU 风险 |
|---|---|---|---|
| **A** | **效果**：省掉多少 padding token | **解析式**：真 tokenize 全部块 → 按 `ENCODE_BATCH` 分批 → 比较"乱序 vs 排序"的 Σ(batch×批内最长) | 无 |
| **B** | **无回归**：向量是否一致 | 同一批文本，`encode_texts`（排序）vs 裸 `model.encode`（乱序）→ max abs diff / 平均余弦 | 低（只用短块子集） |
| **C** | **无回归**：检索序是否一致 | 两套向量各做一次 RRF 检索 → top-k chunk_id 是否逐位相同 | 低 |

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_verify_encode_sort.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"
N_EQ = 64          # 数值等价性用的块数（取最短的，避免触碰显存上限）


def main() -> int:
    from paperpilot.agents import embedder as E

    d = pd.concat([pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{i}.parquet")
                   for i in range(3)], ignore_index=True)
    texts = d["text"].astype(str).tolist()
    print("=" * 112)
    print(f"语料块 {len(texts)} ｜ 字符 中位 {int(np.median([len(t) for t in texts]))}"
          f" p90 {int(np.percentile([len(t) for t in texts], 90))}"
          f" max {max(len(t) for t in texts)} ｜ ENCODE_BATCH={E.ENCODE_BATCH}")

    model = E._get_model()
    tk = model.tokenizer
    # ── A. 解析式：padding token 浪费 ──
    t0 = time.time()
    lens = []
    for i in range(0, len(texts), 256):
        enc = tk(texts[i:i + 256], truncation=False, add_special_tokens=True)
        lens += [len(x) for x in enc["input_ids"]]
    lens = np.asarray(lens)
    print(f"  tokenize 完成（{time.time() - t0:.0f}s）｜ token 长度 中位 {int(np.median(lens))}"
          f" p90 {int(np.percentile(lens, 90))} max {int(lens.max())}")

    def padded_cost(order: np.ndarray) -> int:
        tot = 0
        for i in range(0, len(order), E.ENCODE_BATCH):
            b = order[i:i + E.ENCODE_BATCH]
            tot += int(lens[b].max()) * len(b)
        return tot

    raw = np.arange(len(lens))
    # 复刻 `encode_texts` 的排序（按**字符长度**，与实现一致）
    sorted_order = np.array(sorted(range(len(texts)), key=lambda i: len(texts[i])))
    c_unsorted, c_sorted = padded_cost(raw), padded_cost(sorted_order)
    real = int(lens.sum())
    print(f"\n【A 效果】按 `ENCODE_BATCH={E.ENCODE_BATCH}` 分批的 padding 总量")
    print(f"  乱序（原）：{c_unsorted:>10,} token（其中真实 {real:,} → "
          f"**padding 占 {1 - real / c_unsorted:.1%}**）")
    print(f"  排序（现）：{c_sorted:>10,} token（其中真实 {real:,} → "
          f"**padding 占 {1 - real / c_sorted:.1%}**）")
    print(f"  ★ 计算量降到 **{c_sorted / c_unsorted:.2f}×**（省 {1 - c_sorted / c_unsorted:.1%}）"
          f" ｜ 峰值批内 token：{max(int(lens[raw[i:i + E.ENCODE_BATCH]].max()) for i in range(0, len(raw), E.ENCODE_BATCH))}"
          f" → {max(int(lens[sorted_order[i:i + E.ENCODE_BATCH]].max()) for i in range(0, len(sorted_order), E.ENCODE_BATCH))}")

    # ── B. 数值等价性（短块子集，**打乱**以让两次批组成真的不同）──
    pick = np.sort(lens)[:N_EQ]
    idx = np.argsort(lens)[:N_EQ].copy()
    np.random.default_rng(0).shuffle(idx)          # ★ 打乱：否则 sub 本身就是长度序，测不出差异
    sub = [texts[i] for i in idx]
    print(f"\n【B 无回归】取 {N_EQ} 个**最短块**并**打乱**"
          f"（token 最长 {int(pick.max())}）→ 仅批组成不同，长度分布不变")
    t0 = time.time()
    v_new = E.encode_texts(sub)
    t_new = time.time() - t0
    t0 = time.time()
    v_old = np.asarray(model.encode(list(sub), normalize_embeddings=True,
                                    batch_size=E.ENCODE_BATCH), dtype="float32")
    t_old = time.time() - t0
    dif = np.abs(v_new - v_old).max()
    cos = float((v_new * v_old).sum(axis=1).min())
    print(f"  max |Δ| = **{dif:.3e}** ｜ 逐条最小余弦 = **{cos:.9f}**"
          f" ｜ 耗时 {t_new:.2f}s vs {t_old:.2f}s")
    print(f"  阈值：与当年 `batch 32→8` 的已接受差异同级（3.2e-07）→ "
          f"{'✅ 等价' if dif < 1e-5 and cos > 0.99999 else '❌ 有差异'}")

    # ── C. 检索序不变 ──
    print(f"\n【C 无回归】用两套向量各做一次 RRF 检索，比 top-k 顺序")
    from paperpilot.agents.embedder import BM25Index

    def topk(vecs: np.ndarray, query: str, k: int = 10) -> list[int]:
        q = E.encode_query(query)
        v = (vecs @ q).astype("float64")
        span = np.arange(len(sub))
        bm = np.asarray(BM25Index(sub).score(query), dtype="float64")
        rrf = np.zeros(len(sub))
        for r, i in enumerate(np.argsort(-v)):
            rrf[int(i)] += 1.0 / (60 + r + 1)
        for r, i in enumerate(np.argsort(-bm)):
            rrf[int(i)] += 1.0 / (60 + r + 1)
        return [int(x) for x in np.argsort(-rrf)[:k]]

    ok = True
    for q in ("which papers report ablation results",
              "efficiency of long context transformers",
              "instruction tuning dataset construction"):
        a, b = topk(v_new, q), topk(v_old, q)
        same = a == b
        ok &= same
        print(f"  「{q[:44]:<46}」 top-10 {'逐位相同 ✅' if same else f'不同 ❌ {a} vs {b}'}")
    print(f"\n  ★ 结论：{'排序改动**不改变**检索结果（可安全上线）' if ok else '有差异，需人工核对'}")

    # ── D. 真实峰值显存与耗时（整簇 1,079 块，这才是硬证据）──
    import torch
    if torch.cuda.is_available():
        big = pd.read_parquet(DEV / "prodchunk50" / "mineru" / "c0.parquet")
        bt = big["text"].astype(str).tolist()
        print(f"\n【D 实测】整簇 {len(bt)} 块（token max ~7,608）")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        E.encode_texts(bt)
        dt = time.time() - t0
        pk = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  新实现（排序 + 每批 token 预算 {E.ENCODE_TOKEN_BUDGET}）："
              f"**峰值 {pk:.0f} MiB** ｜ 耗时 **{dt:.1f}s** ｜ 空闲 "
              f"{(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2 - pk):.0f} MiB")
        print(f"  （对照：改动前实测 **5,634 MiB**、**>1,000s 且不收敛**）")
        print(f"  ★ {'✅ 显存已缓解' if pk < 4500 else '⚠️ 峰值仍高，需继续调预算'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
