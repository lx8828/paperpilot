"""**两件事一起查**（50 篇口径，零 LLM）

## A. BM25 通道的价值与质量诊断
· BM25 的 top-k 里**命中 gold 的比例**多少？（与随机基线 k/N 比）
· **独占 gold**（只有 BM25 召回到、dense 前排没有的）有多少？
· BM25 打到 gold 的**名次分布** —— 是"刚好挤不进 k"还是"根本没信号"？
· BM25 的**非零块占比** —— 词法命中是不是太稀疏（= 跨语言 BM25 没建好）

## B. 多查询 × 独立候选池（回应"1 条查询不够"）
现状 `Q_prod` 只有 **1 条英文检索式**；而均 gold **13.00** 是多答案问题。
→ 改为：**每条子查询各自走 dense + BM25**，再**合并候选池**（并集，不是 max）
· 对照三种查询集：`1q`（para）/ `3q`（3 条子查询）/ `3q+zh`（再加中文题面走 dense）
· 报：**池内 gold 覆盖率**（= 完美重排上限）、**完美重排 setF1**、**RRF 实际 setF1**

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_rerank_diag.py --corpus 50
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
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
RRF_K = 60


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2
KS = [5, 10, 13, 20]


def rrf_order(lists: list[tuple[list[str], float]]) -> list[str]:
    sc: dict[str, float] = {}
    for o, w in lists:
        for r, d in enumerate(o):
            sc[d] = sc.get(d, 0.0) + w / (RRF_K + r + 1)
    return sorted(sc, key=lambda d: -sc[d])


def oracle_metrics(pool: list[str], gold: set[str], k: int) -> tuple[float, float]:
    """完美重排：从池里挑最优 k 个 → (setF1, 池内 gold 覆盖率)。"""
    tp = len(set(pool) & gold)
    bk = min(tp, k)
    p = bk / k
    r = bk / len(gold)
    return (2 * p * r / (p + r) if (p + r) else 0.0), tp / len(gold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
    args = ap.parse_args()
    if args.corpus == "50":
        CDIR, KDIR = DEV / "corpus50", DEV / "prodchunk50"
        GF, pmap_f = DEV / "gold_final3.csv", DEV / "corpus50" / "pdf_map_all.json"
    else:
        CDIR, KDIR = DEV / "clusters", DEV / "prodchunk"
        GF, pmap_f = DEV / "gold_final2.csv", DEV / "pdf_map.json"
    pm_ok = {d for d, v in json.loads(pmap_f.read_text(encoding="utf-8")).items()
             if v.get("ok")}
    gg = pd.read_csv(GF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    print("=" * 128)
    print(f"【A｜BM25 通道诊断】+【B｜多查询独立候选池】语料 {args.corpus} ｜ 真值 {GF.name}"
          f" ｜ 切块 {KDIR.name}/{args.chunktag}")

    a_rows, b_rows, detail = [], [], []
    t0 = time.time()
    for ci in range(3):
        ch = pd.read_parquet(KDIR / args.chunktag / f"c{ci}.parquet")
        ls = pd.read_parquet(CDIR / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist()
                if d in set(ch["docid"]) and d in pm_ok]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        idx = {d: np.where(owner == d)[0] for d in docs}
        E = enc.encode(chunks, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {len(chunks)} 块 ｜ {time.time() - t0:.0f}s",
              flush=True)

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            zh, para = str(F2[facet][1]), str(F2[facet][3])
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3]
            # ⚠️ `subqueries.json` 只覆盖 16/19 个 facet（`context_length`/`new_dataset`/`prompt_eng`
            #    是后加的，没有子查询）→ 缺子查询时**回落到 `para`**，否则该题在 `3q` 里会被清零
            #    （第一版就是这个 bug，使 `3q` 的结果被低估）
            SQ_FB = SQ or [para]
            N = len(docs)

            # ── 查询集定义：(名单, 走 dense 的查询, 走 bm25 的英文查询, label) ──
            QS = {
                "1q": ([para], [para]),
                "3q": (list(SQ_FB), list(SQ_FB)),
                "3q+zh": ([zh] + list(SQ_FB), list(SQ_FB)),
            }
            # 逐查询打分（dense 篇级 max；bm25 篇级 max）
            cache: dict[str, tuple[list[str], list[str]]] = {}
            for q in {q for _, (dqs, bqs) in QS.items() for q in list(dqs) + list(bqs)}:
                qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                                convert_to_numpy=True).astype(np.float32)
                Sd = (qv @ E.T)[0]
                Sb = bm.scores(q).astype(np.float32)
                dm_d = np.array([float(Sd[idx[d]].max()) for d in docs])
                dm_b = np.array([float(Sb[idx[d]].max()) for d in docs])
                cache[q] = ([docs[i] for i in np.argsort(-dm_d)],
                            [docs[i] for i in np.argsort(-dm_b)])

            # ══ A：BM25 通道诊断（生产口径 1q）══
            # ⚠️ `cache[q] = (dense_order, bm25_order)` —— 必须**一次解包两个**；
            #    写成 `_, x = cache[q]` 两次会拿到同一个元素（第一版就是这个 bug）
            od, ob = cache[para]
            bm_rank = {d: i for i, d in enumerate(ob)}
            for k in KS:
                tp_b = len(set(ob[:k]) & gold)
                tp_d = len(set(od[:k]) & gold)
                b25_excl = len(set(ob[:k]) & (gold - set(od[:k])))
                a_rows.append(dict(
                    cluster=ci + 1, facet=facet, k=k, n_gold=len(gold), n_docs=N,
                    bm25_StRecall=tp_b / len(gold), dense_StRecall=tp_d / len(gold),
                    random=k / N, bm25_excl=b25_excl,
                    bm25_purity=tp_b / k, dense_purity=tp_d / k,
                    gold_bm25_rank_med=float(np.median(
                        [bm_rank[d] + 1 for d in gold if d in bm_rank] or [np.nan])),
                    bm25_nonzero_blocks=float((bm.scores(para) > 0).mean())))
            detail.append(dict(cluster=ci + 1, facet=facet, n_gold=len(gold),
                               bm25_excl10=len(set(ob[:10]) & gold - set(od[:10])),
                               dense_only10=len(set(od[:10]) & gold - set(ob[:10]))))

            # ══ B：多查询独立候选池 ══
            for lab, (dqs, bqs) in QS.items():
                d_lists = [cache[q][0] for q in dqs]
                b_lists = [cache[q][1] for q in bqs]
                d_lists = [x for x in d_lists if x]
                b_lists = [x for x in b_lists if x]
                for k in KS:
                    pool = list(dict.fromkeys(
                        [x for o in d_lists for x in o[:k]]
                        + [x for o in b_lists for x in o[:k]]))
                    of1, cov = oracle_metrics(pool, gold, k)
                    # RRF：ρ=1 与 ρ=0.25 两种
                    res = {}
                    for rho in (1.0, 0.25):
                        ol = [(o, 1.0) for o in d_lists] + [(o, rho) for o in b_lists]
                        ro = rrf_order(ol)
                        p, r, f1 = _m.setpf(ro, gold, k)
                        res[rho] = (f1, _m.strecall(ro, gold, k))
                    b_rows.append(dict(
                        cluster=ci + 1, facet=facet, qset=lab, k=k, n_gold=len(gold),
                        n_docs=N, n_lists=len(d_lists) + len(b_lists),
                        pool_size=len(pool), pool_cov=cov, oracle_F1=of1,
                        rrf1_F1=res[1.0][0], rrf1_StRecall=res[1.0][1],
                        rrf025_F1=res[0.25][0], rrf025_StRecall=res[0.25][1]))

    A, B, D = pd.DataFrame(a_rows), pd.DataFrame(b_rows), pd.DataFrame(detail)
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    A.to_csv(HERE / "results" / "R2_BM25_DIAG.csv", index=False, encoding="utf-8-sig")
    B.to_csv(HERE / "results" / "R2_MULTIQ_POOL.csv", index=False, encoding="utf-8-sig")

    # ── 表 A：BM25 通道质量 ──
    print("\n" + "=" * 128)
    print("【表 A｜BM25 通道质量（口径 1q，生产形态）】若 BM25 的 top-k 命中率 ≈ 随机，"
          "说明跨语言 BM25 没建好")
    print(f"  {'k':>3}{'随机':>8}{'dense':>8}{'bm25':>8}{'bm25超随机':>11}"
          f"{'bm25纯度':>10}{'dense纯度':>10}{'bm25独占gold':>13}{'gold的bm25名次中位':>19}")
    for k in KS:
        t = A[A.k == k]
        print(f"  {k:>3}{t.random.mean():>8.3f}{t.dense_StRecall.mean():>8.3f}"
              f"{t.bm25_StRecall.mean():>8.3f}"
              f"{t.bm25_StRecall.mean() - t.random.mean():>+11.3f}"
              f"{t.bm25_purity.mean():>10.3f}{t.dense_purity.mean():>10.3f}"
              f"{t.bm25_excl.sum() / t.n_gold.sum():>13.1%}"
              f"{t.gold_bm25_rank_med.median():>19.0f}")
    print(f"\n  BM25 非零块占比 中位 {A.bm25_nonzero_blocks.median():.1%}"
          f"（若极低 = 词法命中太稀疏）")
    print(f"  【判读】bm25超随机 > +0.10 → 通道**有信号**；≈ 0 → 通道坏了。"
          f"`gold的bm25名次中位` 若 ≫ k，说明 gold 在 BM25 里排得很后（挤不进交付）。")

    # ── 表 A2：谁贡献 BM25 的独占 gold ──
    print(f"\n  【表 A2｜BM25 独占 gold（k=10）top 贡献题】")
    dd = D[D.bm25_excl10 > 0].sort_values("bm25_excl10", ascending=False).head(10)
    for _, r in dd.iterrows():
        print(f"    簇{int(r.cluster)} {r.facet:<18}gold {int(r.n_gold):>3}"
              f" ｜ BM25 独占 {int(r.bm25_excl10):>2} ｜ dense 独占 {int(r.dense_only10):>2}")

    # ── 表 B：多查询独立候选池 ──
    print("\n" + "=" * 128)
    print("【表 B｜多查询独立候选池】每条子查询各自走 dense+BM25 → **并集**（不是 max）")
    print(f"  {'查询集':<8}{'k':>4}{'列表数':>7}{'池大小':>8}{'池gold覆盖':>11}"
          f"{'完美重排F1':>11}{'RRF ρ=1 F1':>12}{'RRF ρ=.25 F1':>14}"
          f"{'ρ=.25 StRecall':>15}")
    for lab in ("1q", "3q", "3q+zh"):
        for k in KS:
            t = B[(B.qset == lab) & (B.k == k)]
            if not len(t):
                continue
            print(f"  {lab:<8}{k:>4}{t.n_lists.mean():>7.0f}{t.pool_size.mean():>8.1f}"
                  f"{t.pool_cov.mean():>11.1%}{t.oracle_F1.mean():>11.3f}"
                  f"{t.rrf1_F1.mean():>12.3f}{t.rrf025_F1.mean():>14.3f}"
                  f"{t.rrf025_StRecall.mean():>15.3f}")
    print(f"\n  【Δ 对照 @k=13】")
    for k in (10, 13):
        b1 = B[(B.qset == "1q") & (B.k == k)]
        b3 = B[(B.qset == "3q") & (B.k == k)]
        b4 = B[(B.qset == "3q+zh") & (B.k == k)]
        print(f"    k={k}：池gold覆盖 {b1.pool_cov.mean():.1%} → 3q {b3.pool_cov.mean():.1%}"
              f" → 3q+zh {b4.pool_cov.mean():.1%} ｜ "
              f"RRF(.25)F1 {b1.rrf025_F1.mean():.3f} → {b3.rrf025_F1.mean():.3f}"
              f" → {b4.rrf025_F1.mean():.3f}")
    print(f"\n  → 已写 results/R2_BM25_DIAG.csv ｜ results/R2_MULTIQ_POOL.csv"
          f"（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
