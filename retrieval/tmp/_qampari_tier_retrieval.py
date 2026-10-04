"""**零 LLM 档位检索诊断**：128k vs 1m —— 分清"检索掉"还是"reader 掉"

同时给出与论文 **Text Retrieval / QAMPARI / MRecall@5** 直接可比的口径
（论文 128k：Gemini 0.61 / GPT-4o 0.18 / Claude 0.20 / Specialized(Gecko) 0.57）。
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
BASE = HERE / "data" / "loft" / "qampari"
KS = [5, 10, 20, 40, 100]


def tok(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def mrecall_at_k(ranked_pids, gold, k):
    c = len(set(ranked_pids[:k]) & gold)
    m = len(gold)
    if m == 0:
        return float("nan")
    return 1.0 if (c == m if m <= k else c >= k) else 0.0


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    all_rows = []
    for tier in ("128k", "1m"):
        d = BASE / tier
        corpus = [json.loads(l) for l in (d / "corpus.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        queries = [json.loads(l) for l in (d / "test_queries.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        full = [str(c.get("title_text") or "") + " \n " + str(c.get("passage_text") or "")
                for c in corpus]
        pid = [str(c.get("pid")) for c in corpus]
        pid_set = set(pid)
        t0 = time.time()
        D = enc.encode(full, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        Q = enc.encode([q["query_text"] for q in queries], batch_size=16,
                       normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)
        # BM25
        tf = [Counter(tok(t)) for t in full]
        ln = [max(1, sum(t.values())) for t in tf]
        avg = float(np.mean(ln))
        dfc = Counter()
        for t in tf:
            dfc.update(t.keys())
        N = len(tf)
        idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in dfc.items()}

        def bm25s(q):
            out = np.zeros(N)
            for t in set(tok(q)):
                i_ = idf.get(t)
                if i_ is None:
                    continue
                for i, tt in enumerate(tf):
                    f = tt.get(t, 0)
                    if f:
                        out[i] += i_ * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * ln[i] / avg))
            return out

        gold_sizes = []
        per = {k: [] for k in KS}
        mr5, mr5_any = [], []
        for qi, q in enumerate(queries):
            gold = {str(x[0]) for x in (q.get("metadata") or {}).get("qrels") or []}
            gold &= pid_set
            gold_sizes.append(len(gold))
            dense = D @ Q[qi]
            sparse = bm25s(q["query_text"])
            score = (1.0 / (60 + np.argsort(np.argsort(-dense)) + 1)
                     + 1.0 / (60 + np.argsort(np.argsort(-sparse)) + 1))
            order = np.argsort(-score)
            ranked = [pid[i] for i in order]
            for k in KS:
                c = len(set(ranked[:k]) & gold)
                per[k].append(c / len(gold) if gold else float("nan"))
            mr5.append(mrecall_at_k(ranked, gold, 5))
            mr5_any.append(1.0 if (set(ranked[:5]) & gold) else 0.0)

        print(f"\n{'=' * 112}\n【档位 {tier}】语料 {len(corpus):,} 段 ｜ 问题 {len(queries)} ｜ "
              f"gold 段落均 {np.mean(gold_sizes):.1f} ｜ 编码 {time.time() - t0:.0f}s")
        print(f"  {'k':>5}" + "".join(f"{k:>10}" for k in KS))
        print(f"  {'段落召回':>5}" + "".join(f"{np.mean(per[k]):>10.3f}" for k in KS))
        print(f"  **MRecall@5（与论文 Text Retrieval 可比）{np.mean(mr5):.3f}**"
              f"  ｜ 至少命中 1 条 gold 的比例 {np.mean(mr5_any):.3f}")
        all_rows.append(dict(tier=tier, n_corpus=len(corpus), n_q=len(queries),
                             gold_mean=float(np.mean(gold_sizes)),
                             **{f"recall@{k}": float(np.mean(per[k])) for k in KS},
                             MRecall5=float(np.mean(mr5)), any5=float(np.mean(mr5_any))))

    print(f"\n{'=' * 112}\n【对照】")
    print(f"  {'档位':<8}{'语料段数':>10}{'gold均':>8}" + "".join(f"{'R@' + str(k):>9}" for k in KS)
          + f"{'MRecall@5':>11}")
    for r in all_rows:
        print(f"  {r['tier']:<8}{r['n_corpus']:>10,}{r['gold_mean']:>8.1f}"
              + "".join(f"{r[f'recall@{k}']:>9.3f}" for k in KS) + f"{r['MRecall5']:>11.3f}")
    print("\n  论文参照（Text Retrieval / QAMPARI / MRecall@5，128k 档）："
          "Gemini 0.61 ｜ Specialized(Gecko) 0.57 ｜ Claude 0.20 ｜ GPT-4o 0.18")
    print("  注：论文的 RAG 行用的是 subspan_em（另一套），此处对齐的是 **Text Retrieval** 行。")
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(HERE / "results" / "R2_QAMPARI_TIER_RETRIEVAL.csv",
                                  index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
