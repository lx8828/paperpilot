"""把 arXiv 新语料接进检索：**不退化 + 有增益**的双边评测。

背景（为什么必须双边看）：
  · 新语料与旧语料**重叠仅 13 篇（0.02%）** → 对 LitSearch 的 597 题来说，
    新语料**全是干扰项**（gold 一篇都不在里面）→ 只看这个指标会得出"扩语料有害"；
  · 但对"最近的论文"这类真实查询，旧语料**一篇都答不出** → 只看这个会得出"扩语料纯赚"。
  两个方向都必须量。

两种融合（**语义完全不同**，别混）：

  `rrf`   等权/加权 RRF 交错 —— 新语料**能进头部**，所以能帮到 top-5，
          但也**可能把 gold 挤出 top-5**（这是代价）。用权重 w 扫权衡曲线。
  `quota` 复刻 `paperpilot.components.query_optimizer.quota_union`：
          主路（旧语料）前 Q1 **整块在最前**，新语料接在后面
          → **结构上保证不退化**，但新语料永远落在 Q1 名之后
          → 对"交付 top-5"这种场景**等于没接**。这里当**对照**，说明"保守融合"的局限。

统一 id 空间：旧语料 = 0..N_old-1；新语料 = N_old..N_old+N_new-1。
gold 只存在于旧空间 → 新语料的文档天然匹配不上 gold（正是我们要的语义）。

用法：
    python retrieval/scripts/eval_multi_corpus.py                     # 默认扫权重
    python retrieval/scripts/eval_multi_corpus.py --weights 0,0.1,0.3
    python retrieval/scripts/eval_multi_corpus.py --mode quota --quotas 100,10
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

# ⚠️ 必须在任何 transformers / sentence_transformers 导入**之前** ——
# huggingface_hub 在 import 期就把缓存路径固化成常量，之后再设 HF_HOME 无效。
ensure_hf_home()

from eval_retrieval import (  # noqa: E402
    BM25_DIR, EMB, QFILE, RRF_C, RESULTS, SparseBM25, mrr_at, recall_at, rrf_fuse,
)
from significance import mcnemar  # noqa: E402

OLD = HERE / "data" / "litsearch" / "derived"
NEW = HERE / "data" / "arxiv"


def load_side(root: Path, id_col: str):
    """载入一侧索引：(向量矩阵, bm25, 文档 id 数组)。"""
    emb_dir, bm_dir = root / "emb", root / "bm25"
    parts = sorted(emb_dir.glob("part_*.npy"))
    mat = np.concatenate([np.load(p).astype(np.float32) for p in parts], axis=0)
    ids = np.load(emb_dir / f"{id_col}.npy")
    bm = SparseBM25.load(bm_dir)
    return mat, bm, ids


def quota_union(rank_lists: list[list[int]], quotas: list[int]) -> list[int]:
    """复刻 `src/paperpilot/components/query_optimizer.py::quota_union`。

    第 i 路只取其前 quotas[i] 个，**按路顺序整块拼接**并去重。
    ⚠️ 不能交错 —— 那正是"稀释主路"的错误实现（上游实现第一版踩过）。
    """
    out, seen = [], set()
    for rl, q in zip(rank_lists, quotas):
        for d in rl[:max(q, 0)]:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5, help="臂内 hybrid 的向量权重")
    ap.add_argument("--weights", default="0,0.35,0.45,0.5,0.55,0.65",
                    help="跨语料 RRF 中新语料的权重 w（0 = 只用旧语料）。"
                         "⚠️ RRF 是 **rank 计分**：两路都产出 1..depth 的名次，"
                         "所以 w<~0.48 时新语料**根本进不了 top-5**（名次分差不够）"
                         "—— 网格必须覆盖 0.5 附近，否则会误判成'融合无效'")
    ap.add_argument("--gate-th", type=float, default=-0.01,
                    help="mode=gate 的阈值：`旧语料最高分 − 新语料最高分 < 阈值` 时才把新语料"
                         "并入候选。凭据见 gate_probe.py（该信号 AUC≈0.93~0.99）")
    ap.add_argument("--gate-closed-quota", type=int, default=0,
                    help="门控**关掉**时仍保留的新语料候选数（0=完全关闭）。"
                         "g4 实测：完全关闭会砍掉增益长尾（被关的 28 题里 64%% 本可命中）")
    ap.add_argument("--mode", default="rrf",
                    choices=["rrf", "quota", "ce", "score", "gate"],
                    help="rrf=跨语料 RRF 交错（**已证否：悬崖**）；"
                         "quota=quota_union（保证不退化但新语料进不了头部，等于没接）；"
                         "score=**按 dense 余弦分数跨语料融合**（两路同模型同归一化 → 天然同尺度，"
                         "且几乎免费）；ce=候选合并后 cross-encoder 精排（最强但最慢）")
    ap.add_argument("--ce-pool", default="200,200",
                    help="mode=ce 时从两路各取多少候选进入 CE（q1,q2）")
    ap.add_argument("--qset", default="litsearch", choices=["litsearch", "arxiv"],
                    help="litsearch=gold 在旧语料（测不退化）；arxiv=gold 在新语料（测有增益）")
    ap.add_argument("--arxiv-queries", default="arxiv_queries.parquet",
                    help="qset=arxiv 时用的查询集（arxiv_queries_hard.parquet = 更难的那版）")
    ap.add_argument("--quotas", default="100,10", help="mode=quota 时两路的配额 q1,q2")
    ap.add_argument("--depth", type=int, default=1000, help="每路候选深度")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("载入旧语料索引…")
    m_old, bm_old, ids_old = load_side(OLD, "corpusid")
    N_OLD = len(ids_old)
    print(f"  旧：{m_old.shape[0]} 篇 × {m_old.shape[1]} 维")
    print("载入 arXiv 新语料索引…")
    m_new, bm_new, ids_new = load_side(NEW, "docid")
    N_NEW = len(ids_new)
    print(f"  新：{m_new.shape[0]} 篇 × {m_new.shape[1]} 维")
    N_ALL = N_OLD + N_NEW

    # ── 查询集：两个方向都要测 ────────────────────────────────
    if args.qset == "litsearch":
        # gold 全在**旧**语料 → 新语料只会是干扰项（测"不退化"）
        q = pd.read_parquet(QFILE)
        if args.limit:
            q = q.head(args.limit).reset_index(drop=True)
        gold_ids = np.load(OLD / "emb" / "corpusid.npy")
        id2old = {int(c): i for i, c in enumerate(gold_ids)}
        golds = [{id2old[int(g)] for g in row if int(g) in id2old} for row in q["corpusids"]]
        qtext = q["query"].astype(str).tolist()
        print(f"\n查询集 = LitSearch {len(qtext)} 题（gold 在**旧**语料）")
    else:
        # gold 全在**新**语料 → 旧语料一篇都答不出（测"有增益"）
        q = pd.read_parquet(NEW / "meta" / args.arxiv_queries)
        if args.limit:
            q = q.head(args.limit).reset_index(drop=True)
        golds = [{N_OLD + int(x)} for x in q["gold_docid"]]
        qtext = q["query"].astype(str).tolist()
        print(f"\n查询集 = arXiv 生成 {len(qtext)} 题（gold 在**新**语料）")

    # 查询向量（用旧侧模型口径，两侧同模型同维度）
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    QV = model.encode(qtext, batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    # ── mode=ce：载入两侧原文 + cross-encoder ─────────────────────
    ce_model = None
    t_old = t_new = None
    if args.mode in ("ce", "gate"):        # gate 也要跑 CE，同样需要两侧原文
        # ⚠️ 旧侧向量索引是"text 非空"过滤后的 63,269 行 —— 必须做同样的过滤才能行对齐
        _d = pd.read_parquet(OLD / "corpus_text.parquet", columns=["text"])
        t_old = _d[_d["text"] != ""]["text"].astype(str).tolist()
        t_new = pd.read_parquet(NEW / "meta" / "corpus_text.parquet",
                                columns=["text"])["text"].astype(str).tolist()
        assert len(t_old) == N_OLD and len(t_new) == N_NEW, "文本与向量行数不一致"
        from sentence_transformers import CrossEncoder
        ce_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=512)
        if dev == "cuda":
            try:
                # ⚠️ 实测 fp32→fp16 是 **3.4×** 提速（18 → 63 对/s，见 _scratch/bench_ce.py）。
                # 忘了这一行会让整个实验从 55 分钟变成 4 小时。
                ce_model.model.half()
                print("  CE 已切 fp16")
            except Exception as e:  # noqa: BLE001
                print(f"  CE fp16 失败，用 fp32：{type(e).__name__}")
        print(f"  CE 就绪（文本 旧 {len(t_old)} / 新 {len(t_new)}）")

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    quotas = [int(x) for x in args.quotas.split(",") if x.strip()]
    if args.mode in ("ce", "score", "gate"):
        # ⚠️ ce/score/gate 用 --ce-pool（两路各取多少候选），不是 --quotas
        quotas = [int(x) for x in args.ce_pool.split(",") if x.strip()]
        weights = [0.0]                      # 只有一个配置（按候选池大小区分）
    a = args.alpha
    print(f"\nα(臂内)={a} | 融合 mode={args.mode} | "
          f"{'权重' + str(weights) if args.mode == 'rrf' else '配额' + str(quotas)} "
          f"| 深度={args.depth}\n")

    t0 = time.time()
    rows = []
    for qi in range(len(qtext)):
        # ── 两路各自做 hybrid（BM25 ⊕ dense，RRF）──────────────
        qs = qtext[qi]
        s_old = m_old @ QV[qi]                     # 提出来复用：rank 融合要用，score 融合也要用
        s_new = m_new @ QV[qi]
        r_old = rrf_fuse([np.argsort(-s_old)[:args.depth], bm_old.topk(qs, args.depth)],
                         [a, 1 - a], N_OLD)[:args.depth]
        r_new = rrf_fuse([np.argsort(-s_new)[:args.depth], bm_new.topk(qs, args.depth)],
                         [a, 1 - a], N_NEW)[:args.depth]
        # ⚠️ 两侧 dense 分数是**同一模型、同一归一化、同一查询**的余弦 → **天然同尺度**
        #    （这正是 RRF 丢掉的信息：它把分数换成了两套不可比的"名次"）
        s_all = np.concatenate([s_old, s_new])

        use_new = False          # 只有 gate 模式会改写；其它模式在 row 里记 -1
        for w in weights:
            if args.mode == "score":
                # 候选 = 两路 hybrid 的**并集**，但**按 dense 余弦分数**定序。
                # 分数跨语料可比 → 既能让新语料进头部（查"最近论文"），
                # 又不会像 RRF 那样因为"名次同尺度"而一方通吃。
                pool = np.array(list(dict.fromkeys(
                    r_old[:quotas[0]].tolist()
                    + (r_new[:quotas[1]] + N_OLD).tolist())), dtype=np.int64)
                fused = pool[np.argsort(-s_all[pool])]
            elif args.mode in ("ce", "gate"):
                # 合并两路候选**集合**（不是名次），再让 CE 用**分数**决定顺序 ——
                # 这一步绕开了 RRF 的 rank 归一化问题（跨语料名次分不可比）。
                if args.mode == "gate":
                    # 门控：只在"旧语料的最佳匹配**并不优于**新语料"时才放开新语料。
                    # 信号 = s_old.max() − s_new.max()（两路同模型同归一化 → 可比；且**免费**）
                    sig = float(s_old.max() - s_new.max())
                    use_new = sig < args.gate_th
                    # 梯度：关掉时仍可保留一小撮新语料候选（0 = 硬关）。
                    # 硬关会砍掉增益长尾 —— 被关的题里 64% 本可命中。
                    q_new = quotas[1] if use_new else args.gate_closed_quota
                else:
                    use_new, sig, q_new = True, 0.0, quotas[1]
                pool = list(dict.fromkeys(
                    r_old[:quotas[0]].tolist()
                    + ((r_new[:q_new] + N_OLD).tolist() if q_new > 0 else [])))
                docs = [t_old[g] if g < N_OLD else t_new[g - N_OLD] for g in pool]
                sc = ce_model.predict([(qs, d[:2000]) for d in docs],
                                      batch_size=64, show_progress_bar=False)
                sc = np.asarray(sc, dtype=np.float64)
                fused = np.array(pool, dtype=np.int64)[np.argsort(-sc)]
            elif args.mode == "quota":
                fused = quota_union([r_old.tolist(), (r_new + N_OLD).tolist()],
                                    [quotas[0], quotas[1]])
                fused = np.array(fused, dtype=np.int64)
            elif w <= 0:
                fused = r_old                                    # 基线：只用旧语料
            else:
                fused = rrf_fuse([r_old, r_new + N_OLD], [1 - w, w], N_ALL)
            # 两语料在 top-5 里的占比（"入侵率"）—— 看融合是否真的生效
            top5_new = int((fused[:5] >= N_OLD).sum())
            rows.append({
                "q": qi, "w": w, "R@1": recall_at(fused, golds[qi], 1),
                "R@5": recall_at(fused, golds[qi], 5), "R@10": recall_at(fused, golds[qi], 10),
                "R@100": recall_at(fused, golds[qi], 100), "MRR@10": mrr_at(fused, golds[qi], 10),
                "新进top5": top5_new, "旧进top5": 5 - top5_new,
                "新进top100": int((fused[:100] >= N_OLD).sum()),
                "门控放开": int(use_new) if args.mode == "gate" else -1,
            })
        if (qi + 1) % 100 == 0:
            print(f"  {qi + 1}/{len(qtext)} 题  {time.time() - t0:.0f}s")

    d = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"多语料融合评测（查询集={args.qset}，{len(qtext)} 题；旧 {N_OLD} + 新 {N_NEW}）")
    print("=" * 100)
    g = d.groupby("w").agg(R1=("R@1", "mean"), R5=("R@5", "mean"), R10=("R@10", "mean"),
                           R100=("R@100", "mean"), MRR=("MRR@10", "mean"),
                           新进top5=("新进top5", "mean"),
                           新进top100=("新进top100", "mean"),
                           门控放开率=("门控放开", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    base = d[d["w"] == weights[0]].set_index("q")
    print("\n【配对显著性：各 w 相对 w=%.2f（基线）】" % weights[0])
    sig = []
    for w in weights[1:]:
        cur = d[d["w"] == w].set_index("q")
        common = base.index.intersection(cur.index)
        for metric in ("R@5", "R@1"):
            x = (cur.loc[common, metric].to_numpy() >
                 base.loc[common, metric].to_numpy()).astype(int)   # 新赢
            y = (cur.loc[common, metric].to_numpy() <
                 base.loc[common, metric].to_numpy()).astype(int)   # 新输
            o1, o0 = int(x.sum()), int(y.sum())
            pv = mcnemar(o1, o0)
            dr = cur.loc[common, metric].mean() - base.loc[common, metric].mean()
            sig.append({"w": w, "指标": metric, "Δpt": 100 * dr,
                        "新赢": o1, "旧赢": o0, "p": pv,
                        "结论": "显著" if pv < 0.05 else "不显著"})
            print(f"  w={w:<5} [{metric}] Δ={100 * dr:+.2f}pt  新赢 {o1} / 旧赢 {o0}  "
                  f"p={pv:.4g} {'**显著**' if pv < 0.05 else '不显著'}")

    tag = ("LITSEARCH" if args.qset == "litsearch" else "ARXIV")
    if args.mode in ("ce", "score", "gate"):   # 不同候选池要分开存，否则互相覆盖
        tag += f"_{args.mode.upper()}{args.ce_pool.replace(',', '-')}"
        if args.mode == "gate":
            tag += f"_th{args.gate_th}"
            if args.gate_closed_quota:
                tag += f"_cq{args.gate_closed_quota}"
    if args.qset == "arxiv" and args.arxiv_queries != "arxiv_queries.parquet":
        tag += "_HARD"
    elif args.mode == "quota":
        tag += f"_QUOTA{args.quotas.replace(',', '-')}"
    out = RESULTS / f"{tag}_MULTI_CORPUS.md"
    (out).write_text(
        f"# 多语料融合评测（旧 {N_OLD} + arXiv 新 {N_NEW}）\n\n"
        f"- 查询集 **{args.qset}**（{len(qtext)} 题）| α(臂内)={a} | mode={args.mode} "
        f"| 深度={args.depth} | "
        f"{'权重' + str(weights) if args.mode == 'rrf' else '配额' + str(quotas)}\n"
        + ("- gold 全在**旧**语料（新语料 = 纯干扰项）→ 本表测的是**不退化**\n"
           if args.qset == "litsearch" else
           "- gold 全在**新**语料（旧语料一篇都答不出）→ 本表测的是**有增益**\n")
        + "\n## 指标\n\n" + g.to_string(index=False) + "\n\n"
        "## 配对显著性（相对基线）\n\n" + pd.DataFrame(sig).to_string(index=False) + "\n",
        encoding="utf-8")
    d.to_csv(RESULTS / f"{tag}_multi_corpus_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写出：results/{out.name}\n         results/{tag}_multi_corpus_raw.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
