"""离线排序 A/B（零 LLM、零下载）：能不能把"检索排序精度"提上来？

背景：现状 R@16 = 0.992（金块几乎总在候选里）但 gold@1 仅 0.324、MRR@16 0.512、
margin>0 仅 28.9%、AUC 0.747 —— 问题不是"找得到"，是"排不上来"。
本脚本在同一批候选上对比几种**不引入新模型**的重排/校正手段：

  vec        纯向量 cosine（基线）
  hyb0.50    向量+BM25 等权 RRF（**现状线上**）
  hyb_a      加权 RRF（扫 α）
  cap        RRF + 节级配额去重（线上有开关，默认关）
  mean       **去均值**后再归一化（最轻的各向异性校正）
  pc1/pc2/pc3 去掉前 1/2/3 个主成分方向（各向异性校正）
  white      完整 PCA 白化
  mmr        RRF 上做 MMR 多样性重排（λ=0.3）

指标：gold@1 / R@8,12,16 / MRR@16 / **NDCG@16** / AUC（每题各算再平均）

口径：与 cli/run_retrieval_eval.py **完全一致**（复用其 locate_gold / norm，
gold = 全部人工 evidence 段并集），便于与 RECALL_BASELINE 直接对照。

用法：
    uv run python qa/recall/_rerank_ab.py [--limit-papers N] [--alphas 0.3 0.5 0.7]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from run_retrieval_eval import locate_gold, norm  # noqa: E402  评测侧 gold 口径的唯一来源

KS = (8, 12, 16)
MRR_K = 16
NDCG_K = 16


def rrf_ranks(vs: np.ndarray, bs: np.ndarray, alpha: float) -> np.ndarray:
    n = len(vs)
    rv = np.empty(n)
    rv[np.argsort(-vs, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bs, kind="stable")] = np.arange(n)
    return alpha / (60 + rv + 1) + (1 - alpha) / (60 + rb + 1)


def ndcg_at(order: np.ndarray, gold: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, c in enumerate(order[:k]) if int(c) in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def auc_of(scores: np.ndarray, gold: set[int]) -> float:
    """单题 AUC（Mann-Whitney，含并列取 0.5）。"""
    pos = np.array([scores[i] for i in gold], dtype="float64")
    neg = np.array([s for i, s in enumerate(scores) if i not in gold], dtype="float64")
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="qa/recall/recall_set_v1.json")
    ap.add_argument("--out", default="qa/recall/RERANK_AB_20260911.md")
    ap.add_argument("--limit-papers", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())
    if args.limit_papers:
        pids = pids[: args.limit_papers]

    # ── 第一遍：载入 chunks / 向量 / BM25，并收集全部 chunk 向量用于拟合各向异性校正 ──
    t0 = time.time()
    store: dict[str, dict] = {}
    all_vecs: list[np.ndarray] = []
    for pno, pid in enumerate(pids, 1):
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        vecs = ChunkIndex(pdf).vectors()
        if len(chunks) == 0 or len(vecs) != len(chunks):
            continue
        store[pid] = {"chunks": chunks, "vecs": vecs,
                      "bm": BM25Index([c.text for c in chunks]),
                      "ntexts": [norm(c.text) for c in chunks]}
        all_vecs.append(vecs)
        if pno % 50 == 0:
            print(f"  [载入 {pno}/{len(pids)}] t={time.time()-t0:.0f}s", flush=True)

    V = np.vstack(all_vecs).astype("float64")
    mu = V.mean(axis=0, keepdims=True)
    Vc = V - mu
    # 主成分：特征空间的方向是右奇异向量 Vt（shape 1024×1024），不是样本空间的 U
    _U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    print(f"  拟合完成：{V.shape[0]} 个 chunk 向量，top-1 方差占比 "
          f"{(S[0]**2)/ (S**2).sum():.1%}", flush=True)

    def transform(vecs: np.ndarray, mode: str) -> np.ndarray:
        X = vecs.astype("float64") - mu
        if mode == "mean":
            pass
        elif mode.startswith("pc"):
            p = int(mode[2:])
            X = X - (X @ Vt[:p].T) @ Vt[:p]
        elif mode == "white":
            X = ((X @ Vt.T) / np.sqrt(S ** 2 / max(V.shape[0] - 1, 1) + 1e-8)) @ Vt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(nrm, 1e-12)

    results: dict[str, dict[str, list[float]]] = {}
    groups = ("normal", "hard", "all")

    def rec(col: str, g: str, first, auc, nd):
        r = results.setdefault(col, {k: [] for k in groups})
        r[g].append((first, auc, nd))

    cols = ["vec"] + [f"hyb{a:.2f}" for a in args.alphas] + \
           ["cap", "mean", "pc1", "pc2", "pc3", "white", "mmr"]
    acc = {c: {g: {"first": [], "auc": [], "ndcg": []} for g in groups} for c in cols}

    for pno, pid in enumerate(pids, 1):
        st = store.get(pid)
        if not st:
            continue
        chunks, vecs, bm, ntexts = st["chunks"], st["vecs"], st["bm"], st["ntexts"]
        n = len(chunks)
        if n == 0:
            continue
        sec_rank = _section_rank(chunks)
        for it in by_pid[pid]:
            q = next((qq for qq in papers[pid].get("qas") or []
                      if str(qq.get("question_id") or "") == it["qid"]), None)
            if not q:
                continue
            _gold, evs = gold_answer_full(q)
            if not evs:
                continue
            cg: set[int] = set()
            for ev in evs:
                g_i, _ = locate_gold(ntexts, ev)
                if not g_i:
                    try:
                        g_i = {int(np.argmax(bm.score(norm(ev))))}
                    except Exception:  # noqa: BLE001
                        g_i = set()
                cg |= g_i
            if not cg:
                continue
            g = it["group"]
            qv = encode_query(it["question"])
            bs = np.asarray(bm.score(it["question"]), dtype="float64")

            def score_order(scores: np.ndarray) -> np.ndarray:
                return np.argsort(-scores, kind="stable")

            cand: dict[str, np.ndarray] = {"vec": (qv @ vecs.T).astype("float64")}
            for a in args.alphas:
                cand[f"hyb{a:.2f}"] = rrf_ranks(cand["vec"], bs, a)
            cand["cap"] = cand[f"hyb{args.alphas[len(args.alphas)//2]:.2f}"].copy()
            for mode in ("mean", "pc1", "pc2", "pc3", "white"):
                v2 = transform(vecs, mode)
                q2 = transform(qv.reshape(1, -1), mode)
                s = (q2 @ v2.T).ravel().astype("float64")
                cand[mode] = rrf_ranks(s, bs, 0.5)
            # MMR：在等权 RRF 上做多样性重排
            base = cand[f"hyb{0.5:.2f}"]
            sim = (vecs @ vecs.T).astype("float64")
            cand["mmr"] = _mmr_order(base, sim, lam=0.3)

            for col in cols:
                sc = cand[col]
                if col == "cap":
                    order = _cap_order(sc, chunks, cap=1)
                else:
                    order = score_order(sc)
                first = next((p for p in range(n) if int(order[p]) in cg), None)
                auc = auc_of(sc, cg)
                nd = ndcg_at(order, cg, NDCG_K)
                for gg in (g, "all"):
                    acc[col][gg]["first"].append(first if first is not None else 10 ** 6)
                    acc[col][gg]["auc"].append(auc)
                    acc[col][gg]["ndcg"].append(nd)
        if pno % 25 == 0 or pno == len(pids):
            print(f"  [{pno}/{len(pids)}] t={time.time()-t0:.0f}s", flush=True)

    # ── 报告 ──
    L = ["# 离线排序 A/B：重排/各向异性校正能不能提排序精度（2026-09-11）", "",
         f"> 样本 `{data.get('version')}`｜论文 {len([p for p in pids if p in store])} 篇｜"
         f"零 LLM、零新增模型（仅用缓存的 bge-m3 向量 + 本地 BM25）",
         "> 口径与 `cli/run_retrieval_eval.py` 完全一致（复用其 `locate_gold`，gold=全部 evidence 段并集）",
         "> 各向异性校正的参数**在本评测语料上拟合**（无标签，不构成标签泄漏，但对语料有适配性）", "",
         "| 方法 | gold@1 | R@8 | R@12 | R@16 | MRR@16 | **NDCG@16** | AUC |", "|---|---|---|---|---|---|---|---|"]

    def agg(col: str, g: str) -> dict:
        a = acc[col][g]
        n = len(a["first"])
        if n == 0:
            return {}
        first = np.array(a["first"])
        return {"n": n, "gold1": float((first == 0).mean()),
                **{f"r{k}": float((first < k).mean()) for k in KS},
                "mrr": float(np.mean([1.0 / (f + 1) if f < MRR_K else 0.0 for f in first])),
                "ndcg": float(np.mean(a["ndcg"])), "auc": float(np.mean(a["auc"]))}

    for col in cols:
        s = agg(col, "all")
        if not s:
            continue
        L.append(f"| {col} | {s['gold1']:.3f} | {s['r8']:.3f} | {s['r12']:.3f} | {s['r16']:.3f} "
                 f"| {s['mrr']:.3f} | **{s['ndcg']:.3f}** | {s['auc']:.3f} |")
    L += ["", "## 分组（normal / hard）", "",
          "| 方法 | 组 | gold@1 | R@12 | MRR@16 | NDCG@16 | AUC |", "|---|---|---|---|---|---|---|"]
    for col in cols:
        for g in ("normal", "hard"):
            s = agg(col, g)
            if s:
                L.append(f"| {col} | {g} | {s['gold1']:.3f} | {s['r12']:.3f} | {s['mrr']:.3f} "
                         f"| {s['ndcg']:.3f} | {s['auc']:.3f} |")
    txt = "\n".join(L)
    Path(args.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


def _section_rank(chunks) -> list[int]:
    """按章节在正文中出现的顺序给名次（用于节级配额）。"""
    seen: dict[str, int] = {}
    out = []
    for c in chunks:
        key = c.title_path[0] if getattr(c, "title_path", None) else ""
        if key not in seen:
            seen[key] = len(seen)
        out.append(seen[key])
    return out


def _cap_order(scores: np.ndarray, chunks, cap: int = 1) -> np.ndarray:
    """保序 + 每顶层节最多 cap 块（与生产 _cap_select 同义）。"""
    order = np.argsort(-scores, kind="stable")
    srank = _section_rank(chunks)
    used: dict[int, int] = {}
    out, rest = [], []
    for i in order:
        s = srank[int(i)]
        if used.get(s, 0) < cap:
            used[s] = used.get(s, 0) + 1
            out.append(int(i))
        else:
            rest.append(int(i))
    return np.array(out + rest)


def _mmr_order(scores: np.ndarray, sim: np.ndarray, lam: float = 0.3) -> np.ndarray:
    n = len(scores)
    s = scores / (np.max(np.abs(scores)) + 1e-12)
    chosen: list[int] = []
    pool = list(range(n))
    while pool:
        best, best_v = None, -1e9
        for i in pool:
            pen = max((sim[i, j] for j in chosen), default=0.0)
            v = s[i] - lam * pen
            if v > best_v:
                best, best_v = i, v
        chosen.append(best)
        pool.remove(best)
    return np.array(chosen)


if __name__ == "__main__":
    raise SystemExit(main())
