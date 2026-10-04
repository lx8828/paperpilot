"""**重排诊断 + 导出**：bge-reranker-v2-m3 对 RRF 池重排，看能否救回 top-5 精度

依据（`R2_QAMPARI_TIER1M_20260929.md`）：
  1m 档下 `R@40 = 0.954`（**池子里有**）但 `R@5 = 0.514`、`MRecall@5 = 0.110`（**排不上来**）
  → 重排是直接解。RAG-1 里同款模型曾给 +6.56pt。

产出：
  · 逐档 R@k / MRecall@5 对照（RRF vs RRF+rerank）
  · `data/loft/qampari/<tier>/rerank_top<k>.json`：重排后的 pid 序列（供端到端复用，免重复跑重排）
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
KS = [5, 10, 20, 40]


def tok(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def mrecall_at_k(ranked, gold, k):
    c = len(set(ranked[:k]) & gold)
    m = len(gold)
    return float("nan") if m == 0 else (1.0 if (c == m if m <= k else c >= k) else 0.0)


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--tiers", default="1m")
    ap.add_argument("--pool", type=int, default=100, help="RRF 取多深再重排")
    ap.add_argument("--topk", type=int, default=40, help="重排后保留多少（供端到端用）")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=dev)

    summary = []
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
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

        res = {"rrf": {k: [] for k in KS}, "rerank": {k: [] for k in KS},
               "mr5_rrf": [], "mr5_rr": []}
        export = {}
        t1 = time.time()
        for qi, q in enumerate(queries):
            gold = {str(x[0]) for x in (q.get("metadata") or {}).get("qrels") or []} & pid_set
            dense = D @ Q[qi]
            sparse = bm25s(q["query_text"])
            score = (1.0 / (60 + np.argsort(np.argsort(-dense)) + 1)
                     + 1.0 / (60 + np.argsort(np.argsort(-sparse)) + 1))
            order = np.argsort(-score)
            rrf = [pid[i] for i in order]
            pool_idx = list(order[: args.pool])
            pairs = [(q["query_text"], full[i][:2000]) for i in pool_idx]
            cs = np.asarray(ce.predict(pairs, batch_size=args.batch, show_progress_bar=False))
            rr = [pid[pool_idx[j]] for j in np.argsort(-cs)]
            reranked = rr[: args.topk]
            export[q["qid"]] = reranked
            for k in KS:
                res["rrf"][k].append(len(set(rrf[:k]) & gold) / len(gold))
                res["rerank"][k].append(len(set(reranked[:k]) & gold) / len(gold))
            res["mr5_rrf"].append(mrecall_at_k(rrf, gold, 5))
            res["mr5_rr"].append(mrecall_at_k(reranked, gold, 5))
            if (qi + 1) % 20 == 0:
                print(f"    {tier}: {qi + 1}/{len(queries)}（{time.time() - t1:.0f}s）", flush=True)

        (d / f"rerank_top{args.topk}.json").write_text(
            json.dumps(export, ensure_ascii=False), encoding="utf-8")
        print(f"\n【档位 {tier}】语料 {len(corpus):,} 段 ｜ 池深 {args.pool} ｜ 重排 {time.time() - t1:.0f}s")
        print(f"  {'指标':<12}{'RRF':>10}{'+rerank':>10}{'Δ':>9}")
        for k in KS:
            a, b = float(np.mean(res["rrf"][k])), float(np.mean(res["rerank"][k]))
            print(f"  {'R@' + str(k):<12}{a:>10.3f}{b:>10.3f}{b - a:>+9.3f}")
        a, b = float(np.mean(res["mr5_rrf"])), float(np.mean(res["mr5_rr"]))
        print(f"  {'MRecall@5':<12}{a:>10.3f}{b:>10.3f}{b - a:>+9.3f}"
              f"   ← 论文 Gecko 0.57 / Gemini 0.61")
        summary.append(dict(tier=tier, pool=args.pool,
                            **{f"rrf_R{k}": float(np.mean(res["rrf"][k])) for k in KS},
                            **{f"rr_R{k}": float(np.mean(res["rerank"][k])) for k in KS},
                            rrf_MRecall5=a, rr_MRecall5=b))

    import pandas as pd
    pd.DataFrame(summary).to_csv(HERE / "results" / "R2_QAMPARI_RERANK.csv",
                                 index=False, encoding="utf-8")
    print(f"\n已写 results/R2_QAMPARI_RERANK.csv ｜ 重排序列 → data/loft/qampari/<tier>/rerank_top{args.topk}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
