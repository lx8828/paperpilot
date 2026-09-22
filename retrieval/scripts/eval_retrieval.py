"""第 3 步：LitSearch 篇级检索评测（BM25 / dense / hybrid）。

**为什么不用 `paperpilot.agents.embedder.BM25Index`**：它是纯 Python 逐 doc 的
`Counter` 实现（为"单篇 ~20 块"设计）。到 63,269 篇时，构造要数分钟、每题打分要
遍历全部文档 → 不可用。这里改用 **scipy 稀疏矩阵**实现**同一个 Okapi BM25 公式**
（k1=1.5 / b=0.75，同样的 idf 平滑），速度差几个数量级，指标口径不变。

指标（qrels 94.3% 是单正例 → 主指标 Recall@k + MRR@10）：
    Recall@k = |gold ∩ top-k| / |gold|      （单正例时即"是否命中"）
    MRR@10   = 1 / 首个 gold 的位次（不在前 10 记 0）

分层报告（这是"有观点"的关键，不只报一个总分）：
    · query_set（inline_nonacl / manual_acl / inline_acl / manual_iclr）
    · specificity（0/1）、quality（1/2）
    · **是否含精确 token**（大写缩写 / 数字）← 用来解释混合检索的收益边界

用法：
    python retrieval/scripts/eval_retrieval.py              # 全部 597 题
    python retrieval/scripts/eval_retrieval.py --limit 50   # 快测
    python retrieval/scripts/eval_retrieval.py --alpha 0,0.25,0.5,0.75,1
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DERIVED = HERE / "data" / "litsearch" / "derived"
EMB = DERIVED / "emb"
BM25_DIR = DERIVED / "bm25"
CORPUS = DERIVED / "corpus_text.parquet"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
RESULTS = HERE / "results"

K1, B = 1.5, 0.75
RRF_C = 60          # RRF 常数，与 paperpilot 生产口径一致
TOPK = 100          # 报告用（Recall@100 需要）
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(str(s).lower())


# ───────────────────────── BM25（scipy 稀疏）─────────────────────────
class SparseBM25:
    """Okapi BM25，稀疏矩阵实现（公式与 paperpilot 的 BM25Index 一致）。"""

    def __init__(self, texts: list[str], k1: float = K1, b: float = B, min_df: int = 2):
        from sklearn.feature_extraction.text import CountVectorizer

        self.k1, self.b = k1, b
        self.cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+",
                                  min_df=min_df, dtype=np.float32)
        self.X = self.cv.fit_transform(texts)            # (N, V) 词频
        self.vocab = self.cv.vocabulary_
        n = self.X.shape[0]
        df = np.asarray((self.X > 0).sum(axis=0)).ravel()          # 文档频次
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(self.X.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(dl.mean()) if n else 0.0
        # K = k1 * (1 - b + b * dl / avgdl)
        self.K = (k1 * (1 - b + b * dl / max(self.avgdl, 1e-9))).astype(np.float32)

    def _names(self) -> np.ndarray:
        """词表反查（列号 → 词），**必须缓存**。

        ⚠️ 为什么：`get_feature_names_out()` 每次调用都要把整个 `vocabulary_`
        重新排序建成一个 V 长度的字符串数组。本语料 V=155,915 → 实测 **93 ms/次**。
        而 `score()` 原本在**每个 query 词上各调一次**（`[f(i) for i in idx]`），
        于是一个 11 词的查询要 11 次 ≈ 1,025 ms —— 把 BM25 从 ~150ms 拖到 ~1s。

        缓存后**输出逐位不变**（该函数对固定词表是纯函数）。
        """
        n = getattr(self, "_names_cache", None)
        if n is None:
            n = self.cv.get_feature_names_out()
            self._names_cache = n
        return n

    def score(self, query: str) -> np.ndarray:
        """返回全部文档的 BM25 分（未归一），稀疏切片后稠密计算，只用 query 命中的列。"""
        toks = tokenize(query)
        idx = [self.vocab[t] for t in toks if t in self.vocab]
        out = np.zeros(self.X.shape[0], dtype=np.float32)
        if not idx:
            return out
        sub = np.asarray(self.X[:, idx].todense(), dtype=np.float32)   # (N, |q|)
        names = self._names()                     # 缓存，见 _names() 的说明
        w = self.idf[idx] * np.array([toks.count(names[i])
                                      for i in idx], dtype=np.float32)
        out = (((sub * (self.k1 + 1)) / (sub + self.K[:, None])) @ w).astype(np.float32)
        return out

    def topk(self, query: str, k: int) -> np.ndarray:
        s = self.score(query)
        k = min(k, len(s))
        part = np.argpartition(-s, k - 1)[:k]
        return part[np.argsort(-s[part])]

    def save(self, d: Path) -> None:
        from scipy import sparse
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "X.npz", self.X)
        np.savez(d / "stats.npz", idf=self.idf, K=self.K, avgdl=np.array([self.avgdl]))
        (d / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        (d / "params.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b, "n": int(self.X.shape[0]),
                        "V": int(self.X.shape[1])}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "SparseBM25 | None":
        from scipy import sparse
        if not (d / "X.npz").exists():
            return None
        obj = cls.__new__(cls)
        obj.X = sparse.load_npz(d / "X.npz").tocsr()
        st = np.load(d / "stats.npz")
        obj.idf, obj.K = st["idf"], st["K"]
        obj.avgdl = float(st["avgdl"][0])
        obj.vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        p = json.loads((d / "params.json").read_text(encoding="utf-8"))
        obj.k1, obj.b = p["k1"], p["b"]
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+")
        cv.vocabulary_ = obj.vocab
        obj.cv = cv
        return obj


# ───────────────────────── 指标 ─────────────────────────
def recall_at(ranked: np.ndarray, gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(gold & set(ranked[:k].tolist()))
    return hit / len(gold)


def mrr_at(ranked: np.ndarray, gold: set[int], k: int) -> float:
    for i, d in enumerate(ranked[:k].tolist(), 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def rrf_fuse(rank_lists: list[np.ndarray], weights: list[float], n: int) -> np.ndarray:
    """RRF：score(d) = Σ w_i / (C + rank_i(d) + 1)。"""
    score = np.zeros(n, dtype=np.float64)
    for ranks, w in zip(rank_lists, weights):
        score[ranks] += w / (RRF_C + np.arange(len(ranks)) + 1)
    return np.argsort(-score)


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """自己格式化 markdown 表格 —— 避免为 `to_markdown` 引入 tabulate 依赖。"""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", default="0.5",
                    help="向量权重（RRF）；逗号分隔可扫描多值。1=纯向量，0=纯BM25")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--rebuild-bm25", action="store_true")
    ap.add_argument("--dump-topk", action="store_true",
                    help="把各系统的 top-k 候选行号存成 results/<--dump-topk-name>（供 rerank 复用）")
    ap.add_argument("--dump-topk-name", default="topk.npz",
                    help="top-k 候选文件名。**扩池子时务必换名**，否则会覆盖 rerank_eval 依赖的 "
                         "(597,100) 布局")
    ap.add_argument("--fuse-in", type=int, default=1000,
                    help="RRF **组件列表**的深度（默认 1000）。必须显著大于输出深度："
                         "RRF 靠 rank 计分，若组件列表被截断到与输出同长，排在其后的文档"
                         "在融合里得 0 分 → 融合结果系统性偏移（实测 R@100 少 1.9pt）。")
    ap.add_argument("--curve", default="",
                    help="输出 Recall@k 曲线（逗号分隔的 k，需 <= --topk）。"
                         "R@k 即'池子开 k 时的召回率' = 精排的 R@1 理论上界")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in args.alpha.split(",")]

    # ── 语料与向量 ───────────────────────────────────────────
    df = pd.read_parquet(CORPUS, columns=["corpusid", "text", "title"])
    df = df[df["text"] != ""].reset_index(drop=True)
    ids = np.load(EMB / "corpusid.npy")
    assert len(ids) == len(df), f"向量行数({len(ids)})与语料({len(df)})不一致，请重跑 encode_corpus.py"
    assert (ids == df["corpusid"].to_numpy()).all(), "corpusid 顺序不一致"
    mat = np.concatenate([np.load(p) for p in sorted(EMB.glob("part_*.npy"))], axis=0)
    id2row = {int(c): i for i, c in enumerate(ids)}
    n_docs = len(ids)
    print(f"语料 {n_docs} 篇 | 向量 {mat.shape} ({mat.nbytes / 1e6:.0f} MB)")

    # ── BM25（缓存）─────────────────────────────────────────
    bm = None if args.rebuild_bm25 else SparseBM25.load(BM25_DIR)
    if bm is None or bm.X.shape[0] != n_docs:
        t0 = time.time()
        print("构建 BM25（首次较慢，之后走缓存）…")
        bm = SparseBM25(df["text"].tolist())
        bm.save(BM25_DIR)
        print(f"  完成 {time.time() - t0:.1f}s | V={bm.X.shape[1]} | "
              f"非零 {bm.X.nnz / 1e6:.1f}M | avgdl={bm.avgdl:.1f}")
    else:
        print(f"BM25 走缓存 | V={bm.X.shape[1]} | 非零 {bm.X.nnz / 1e6:.1f}M")

    # ── 查询 ────────────────────────────────────────────────
    q = pd.read_parquet(QFILE)
    if args.limit:
        q = q.head(args.limit).reset_index(drop=True)
    golds = [[int(x) for x in v] for v in q["corpusids"]]
    # 只保留 gold 都在语料里的查询（本数据集已校验为 597/597）
    keep = [i for i, g in enumerate(golds) if all(gid in id2row for gid in g)]
    if len(keep) != len(golds):
        print(f"  剔除 gold 不在语料的查询 {len(golds) - len(keep)} 条")
        q = q.iloc[keep].reset_index(drop=True)
        golds = [golds[i] for i in keep]
    gold_rows = [{id2row[g] for g in gs} for gs in golds]

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    print(f"编码 {len(q)} 条查询（{dev}）…")
    t0 = time.time()
    qv = model.encode(q["query"].tolist(), batch_size=32, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    print(f"  完成 {time.time() - t0:.1f}s")

    # 精确 token 分层：查询含大写缩写 / 数字
    has_acr = q["query"].str.contains(r"\b[A-Z]{2,}\b", regex=True)
    has_num = q["query"].str.contains(r"\d", regex=True)
    strata = {
        "全部": np.ones(len(q), dtype=bool),
        "含精确token": (has_acr | has_num).to_numpy(),
        "纯自然语言": ~(has_acr | has_num).to_numpy(),
        "specificity=0": (q["specificity"] == 0).to_numpy(),
        "specificity=1": (q["specificity"] == 1).to_numpy(),
        "quality=1": (q["quality"] == 1).to_numpy(),
        "quality=2": (q["quality"] == 2).to_numpy(),
    }
    for s in q["query_set"].unique():
        strata[f"set:{s}"] = (q["query_set"] == s).to_numpy()

    # ── 逐题检索 ────────────────────────────────────────────
    print(f"\n检索 {len(q)} 题（topk={args.topk}，alpha 扫描 {alphas}）…")
    t0 = time.time()
    res: dict[str, list[np.ndarray]] = {"bm25": [], "dense": []}
    for a in alphas:
        res[f"hyb{a}"] = []
    # ⚠️ 组件列表深度必须 > 输出深度（见 --fuse-in 说明）；否则 RRF 结果系统性偏移
    fuse_in = max(args.fuse_in, args.topk)
    if fuse_in > args.topk:
        print(f"  RRF 组件深度 {fuse_in}（输出深度 {args.topk}）")
    for i in range(len(q)):
        rb = bm.topk(str(q.loc[i, "query"]), fuse_in)
        rd = np.argsort(-(mat @ qv[i]))[:fuse_in]
        res["bm25"].append(rb[:args.topk])
        res["dense"].append(rd[:args.topk])
        for a in alphas:
            res[f"hyb{a}"].append(rrf_fuse([rd, rb], [a, 1 - a], n_docs)[:args.topk])
    print(f"  完成 {time.time() - t0:.1f}s")

    # ── 指标 ────────────────────────────────────────────────
    def metrics(ranked_list: list[np.ndarray], mask: np.ndarray | None = None) -> dict:
        idx = range(len(ranked_list)) if mask is None else np.where(mask)[0]
        idx = list(idx)
        if not idx:
            return {}
        out = {}
        for k in (1, 5, 10, 100):
            out[f"R@{k}"] = float(np.mean([recall_at(ranked_list[i], gold_rows[i], k) for i in idx]))
        out["MRR@10"] = float(np.mean([mrr_at(ranked_list[i], gold_rows[i], 10) for i in idx]))
        out["n"] = len(idx)
        return out

    order = ["bm25", "dense"] + [f"hyb{a}" for a in alphas]
    rows = []
    for name in order:
        m = metrics(res[name])
        rows.append({"配置": name, **m})
    table = pd.DataFrame(rows)

    # ── 逐题结果（供显著性检验：McNemar 用二元命中，bootstrap 用连续指标）──
    pq_rows = []
    for i in range(len(q)):
        gset = gold_rows[i]
        for name in order:
            r = res[name][i]
            ranks = []
            for g in gset:
                pos = np.where(r == g)[0]
                ranks.append(int(pos[0]) + 1 if len(pos) else 0)
            best = min([x for x in ranks if x > 0], default=0)
            row = {"qidx": i, "system": name, "best_rank": best, "n_gold": len(gset)}
            for k in (1, 5, 10, 100):
                row[f"hit@{k}"] = int(0 < best <= k)
                row[f"recall@{k}"] = (sum(1 for x in ranks if 0 < x <= k) / len(gset)) if gset else 0.0
            row["mrr@10"] = (1.0 / best) if 0 < best <= 10 else 0.0
            pq_rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pq_rows).to_parquet(RESULTS / "per_query.parquet", index=False)
    if args.dump_topk:
        np.savez_compressed(
            RESULTS / args.dump_topk_name,
            **{name: np.stack(res[name]).astype(np.int32) for name in order})
        print(f"已导出 top-k 候选：results/{args.dump_topk_name}"
              f"（{len(order)} 系统 × {len(q)} 题 × {args.topk}）")

    print("\n" + "=" * 84)
    print("总体（全部 %d 题）" % len(q))
    print("=" * 84)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    strata_tables = {}
    for sname, mask in strata.items():
        if mask.sum() < 5 or sname == "全部":
            continue
        rows = []
        for name in order:
            m = metrics(res[name], mask)
            rows.append({"配置": name, **m})
        strata_tables[sname] = pd.DataFrame(rows)

    for sname, t in strata_tables.items():
        print(f"\n--- 分层：{sname}（n={int(t['n'].iloc[0])}）---")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 落盘 ────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_md = RESULTS / "LITSEARCH_BASELINE.md"
    lines = [f"# LitSearch 篇级检索 · 基线（{stamp}）", "",
             f"- 语料：{n_docs} 篇（title+abstract，bge-m3 归一化向量）",
             f"- 查询：{len(q)} 条（gold 全在语料内）",
             f"- 指标：Recall@1/5/10/100、MRR@10；RRF α 扫描 {alphas}（α=向量权重）", "",
             "## 总体", "", md_table(table), ""]
    for sname, t in strata_tables.items():
        lines += [f"## 分层：{sname}（n={int(t['n'].iloc[0])}）", "", md_table(t), ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / f"litsearch_raw_{stamp}.json").write_text(json.dumps({
        "alphas": alphas, "topk": args.topk, "n_docs": n_docs, "n_queries": len(q),
        "overall": table.to_dict("records"),
        "strata": {k: v.to_dict("records") for k, v in strata_tables.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写出：\n  {out_md.relative_to(HERE)}\n  results/litsearch_raw_{stamp}.json")

    # ── Recall@k 曲线：池子召回率 = 精排 R@1 的理论上界 ──────────────
    if args.curve:
        ks = [int(x) for x in args.curve.split(",") if x.strip()]
        kk = max(ks)
        if kk > args.topk:
            print(f"\n[警告] --curve 最大 k={kk} 超过 --topk={args.topk} → 曲线尾部会被截断，"
                  f"请用 --topk {kk} 重跑")
        cur = {name: {k: float(np.mean([recall_at(res[name][i], gold_rows[i], k)
                                        for i in range(len(q))])) for k in ks}
               for name in order}

        print("\n" + "=" * 104)
        print("★ Recall@k 曲线：R@k =「池子开 k」时找到 gold 的查询比例 = 精排 R@1 的理论上界")
        print("=" * 104)
        print(f"  {'配置':<10}" + "".join(f"{'R@' + str(k):>9}" for k in ks))
        for name in order:
            print(f"  {name:<10}" + "".join(f"{cur[name][k]:9.4f}" for k in ks))

        print("\n  每个 k 上的最优系统（若随 k 变化 → K_cand 与 α 耦合，必须联合扫）：")
        best_by_k = {}
        for k in ks:
            b = max(order, key=lambda s, k=k: cur[s][k])
            best_by_k[k] = b
            print(f"    k={k:<6} → {b:<9} R@{k}={cur[b][k]:.4f}")

        print(f"\n  相对 k={kk} 的达成率（%）：低于 ~95% 说明该档仍在丢召回")
        print(f"    {'配置':<10}" + "".join(f"{'k=' + str(k):>9}" for k in ks))
        for name in order:
            full = max(cur[name][kk], 1e-9)
            print(f"    {name:<10}" + "".join(f"{100 * cur[name][k] / full:9.1f}" for k in ks))

        print("\n  边际增益（R@k − R@上一档）：")
        print(f"    {'配置':<10}" + "".join(f"{'→' + str(k):>9}" for k in ks[1:]))
        for name in order:
            print(f"    {name:<10}"
                  + "".join(f"{cur[name][k] - cur[name][pv]:+9.4f}"
                            for pv, k in zip(ks[:-1], ks[1:])))

        ct = pd.DataFrame([{"配置": n, **{f"R@{k}": cur[n][k] for k in ks}} for n in order])
        reach = pd.DataFrame([{"配置": n,
                               **{f"k={k}": 100 * cur[n][k] / max(cur[n][kk], 1e-9) for k in ks}}
                              for n in order])
        lines = [f"# LitSearch · Recall@k 曲线（池子深度扫描，{time.strftime('%Y%m%d_%H%M%S')}）", "",
                 f"- 语料 {n_docs} 篇；查询 {len(q)} 条；检索 `--topk {args.topk}`",
                 "- **R@k =「池子开 k」时的召回率，也就是精排 R@1 的理论上界**",
                 "- k 越大池子越深：精排可选更多，但成本线性增长、噪声也增多", "",
                 "## Recall@k 曲线", "", md_table(ct), "",
                 "## 每个 k 上的最优系统（K_cand 与 α 是否耦合）", ""]
        lines += [f"- `k={k}` → `{v}`（R@{k}={cur[v][k]:.4f}）" for k, v in best_by_k.items()]
        lines += ["", f"## 相对 k={kk} 的达成率（%）", "", md_table(reach, floatfmt=".1f")]
        (RESULTS / "LITSEARCH_RECALL_CURVE.md").write_text("\n".join(lines), encoding="utf-8")
        (RESULTS / "recall_curve.json").write_text(json.dumps(
            {"ks": ks, "curves": cur,
             "best_by_k": {str(k): v for k, v in best_by_k.items()},
             "topk": args.topk, "n_queries": len(q), "n_docs": n_docs},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n已写出：\n  results/LITSEARCH_RECALL_CURVE.md\n  results/recall_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
