"""离线 Recall@k 检索评估（尺子，零 LLM）。

对 recall_set 逐题计算 vec / hybrid(向量+BM25 RRF) / bm25 三种检索在
gold evidence chunk 上的 Recall@k 与 MRR@k，并输出 gold 跨块/块内位置诊断。

口径（qa/RAG_ENGINE_DESIGN.md §1）：
- chunk = 线上问答同源（document_cache.ordered_chunks）；向量/BM25/RRF 公式与生产一致；
- gold → chunk 映射：整段子串 → 按句串 → 首锚子串；仍空记 mapping_fail，不计分母；
- gold 口径：该题**全部**人工 evidence 段（gold_answer_full）并集；--gold-single 可回退首段旧口径；
- Recall@k = |top_k ∩ C_gold| > 0 占比；MRR@k 以第一个 gold 位次 < k 计。

用法：
    uv run python cli/run_retrieval_eval.py --set qa/recall/recall_set_v1.json \
        --out qa/recall/RECALL_BASELINE.md [--limit-papers N]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import numpy as np  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import (BM25Index, ChunkIndex,  # noqa: E402
                                        encode_query)
from paperpilot.qasper_source import load_papers  # noqa: E402

_REF_RE = re.compile(r"\b[A-Z]+REF\d+\b")
_FIG_RE = re.compile(r"\bFIGURE\d+\b|\bTABLE\d+\b", re.I)
# 列：vec / bm25 / 各 α 的加权混合（hyb0.5 = 现状等权）
# α 为向量权重（RRF: α/(60+rv+1) + (1-α)/(60+rb+1)）；bm 权重 = 1-α
COLS = ("vec", "bm25")


def norm(s: str) -> str:
    t = " ".join(str(s).split())
    t = _REF_RE.sub("", t)
    t = _FIG_RE.sub("", t)
    return " ".join(t.split())


def locate_gold(ntexts: list[str], evidence: str) -> tuple[set[int], dict]:
    ev = norm(evidence)
    diag = {"cross": False, "frac": None}
    if not ev:
        return set(), diag
    # a) 整段
    for i, t in enumerate(ntexts):
        if ev in t:
            diag["frac"] = round(t.find(ev) / max(len(t), 1), 3)
            return {i}, diag
    # b) 按句
    got: set[int] = set()
    for seg in re.split(r"[.;:]\s|\n", evidence):
        seg = seg.strip()
        if len(seg) < 20:
            continue
        ns = norm(seg)
        for i, t in enumerate(ntexts):
            if ns in t:
                got.add(i)
                break
    if got:
        diag["cross"] = len(got) > 1
        if not diag["cross"]:
            idx = next(iter(got))
            anchor = next((norm(s) for s in re.split(r"[.;:]\s|\n", evidence)
                           if len(s.strip()) >= 20 and norm(s) in ntexts[idx]), "")
            diag["frac"] = round(ntexts[idx].find(anchor) / max(len(ntexts[idx]), 1), 3) if anchor else None
        return got, diag
    # c) 首锚（前 80 字）
    anchor = ev[:80]
    for i, t in enumerate(ntexts):
        if anchor in t:
            diag["frac"] = round(t.find(anchor) / max(len(t), 1), 3)
            return {i}, diag
    return set(), diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 12, 16])
    ap.add_argument("--mrr-k", type=int, default=16)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5],
                    help="加权 RRF 的向量权重 α（bm 权重 = 1-α）。传多个=一次扫多组，0.5=现状等权")
    ap.add_argument("--limit-papers", type=int, default=None)
    ap.add_argument("--skip-missing", action="store_true",
                    help="跳过无 cvec 缓存的论文（预览口径，报告标注覆盖）")
    ap.add_argument("--gold-single", action="store_true",
                    help="回退旧口径：只用 evidence[0] 作 gold（默认=全部 evidence 段并集）")
    args = ap.parse_args()

    data = json.loads(Path(args.set).read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())
    if args.skip_missing:
        kept = [p for p in pids
                if (ROOT / f"assets/artifacts/out_views/qasper_{p}.cvec.npy").exists()]
        print(f"[skip-missing] {len(pids)} → {len(kept)} 篇（其余留待预编码后补全）")
        pids = kept
    if args.limit_papers:
        pids = pids[: args.limit_papers]

    # 列：vec / bm25 / 各 α 加权混合（hyb0.5 = 现状等权）
    hyb_cols = [f"hyb{a:.2f}" for a in args.alphas]
    cols = list(COLS) + hyb_cols
    # group -> col -> k -> hits
    hit = {g: {c: {k: 0 for k in args.ks} for c in cols}
           for g in ("normal", "hard", "all")}
    mrr = {g: {c: 0.0 for c in cols} for g in ("normal", "hard", "all")}
    denom = {g: 0 for g in ("normal", "hard", "all")}
    stat = {"cross": 0, "fracs": [], "fallback": 0}
    map_fail: list[str] = []

    t0 = time.time()
    for pno, pid in enumerate(pids, 1):
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
        vecs = ChunkIndex(pdf).vectors()
        bm = BM25Index([c.text for c in chunks])
        ntexts = [norm(c.text) for c in chunks]
        for it in by_pid[pid]:
            q = next((qq for qq in papers[pid].get("qas") or []
                      if str(qq.get("question_id") or "") == it["qid"]), None)
            if not q:
                continue
            from paperpilot.qasper_source import gold_answer_full
            _gold, evs = gold_answer_full(q)
            if not evs:
                map_fail.append(f"{pid}:{it['qid'][:8]}(no-ev)")
                continue
            # 默认口径：全部 evidence 段并集为 gold（--gold-single 可回退首段旧口径做 A/B）
            if args.gold_single:
                evs = evs[:1]
            cg: set[int] = set()
            diag = {"cross": False, "frac": None}
            used_fallback = False
            for ev in evs:
                g_i, d_i = locate_gold(ntexts, ev)
                if not g_i and bm is not None:
                    # 兜底：用 evidence 本身做 BM25 自检索，取 top1（口径标记 fallback）
                    try:
                        g_i = {int(np.argmax(np.asarray(bm.score(norm(ev)), dtype="float64")))}
                        d_i["frac"] = None
                        used_fallback = True
                    except Exception:
                        g_i = set()
                if g_i:
                    cg |= g_i
                    diag["cross"] = diag["cross"] or d_i["cross"]
                    if diag["frac"] is None:
                        diag["frac"] = d_i["frac"]
            if used_fallback:
                stat["fallback"] += 1
            if not cg:
                map_fail.append(f"{pid}:{it['qid'][:8]}(no-locate)")
                continue
            grp = it["group"]
            for g in (grp, "all"):
                denom[g] += 1
            if diag["cross"]:
                stat["cross"] += 1
            if diag["frac"] is not None:
                stat["fracs"].append(diag["frac"])
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            order_v = np.argsort(-vs, kind="stable")
            order_b = np.argsort(-bs, kind="stable")
            rv = np.empty(n); rb = np.empty(n)
            rv[order_v] = np.arange(n); rb[order_b] = np.arange(n)
            # 加权 RRF：α=向量权重（每路由 rank 贡献 1/(60+rank) 再乘权重）
            orders: dict[str, np.ndarray] = {"vec": order_v, "bm25": order_b}
            for a in args.alphas:
                rrf = a / (60 + rv + 1) + (1 - a) / (60 + rb + 1)
                orders[f"hyb{a:.2f}"] = np.argsort(-rrf, kind="stable")
            for col, order in orders.items():
                first = next((int(p) for p in range(n) if int(order[p]) in cg), None)
                if first is None:
                    continue
                for g in (grp, "all"):
                    for k in args.ks:
                        if first < k:
                            hit[g][col][k] += 1
                    if first < args.mrr_k:
                        mrr[g][col] += 1.0 / (first + 1)
        if pno % 25 == 0 or pno == len(pids):
            print(f"  [{pno}/{len(pids)}] 篇 q={sum(len(v) for v in by_pid.values()):,} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    # 报告
    kk = args.ks
    L = ["# 离线 Recall@k × RRF 权重扫描（检索尺子）", ""]
    L.append(f"> 样本 `{data['version']}`（seed {data.get('seed')}）｜ 论文 {len(pids)} 篇")
    L.append(f"> 加权 RRF：α·1/(60+rank_vec) + (1-α)·1/(60+rank_bm)；**hyb0.50 = 现状等权**，"
             f"α 越小 BM25 权重越大")
    L.append(f"> gold 口径：{'**仅 evidence[0]**（旧）' if args.gold_single else '**全部 evidence 段并集**（官方）'}")
    L.append(f"> gold 定位成功/失败：`{denom['all']}` / `{len(map_fail)}`"
             f"（失败例：{'; '.join(map_fail[:4])}）")
    L.append(f"> 映射口径：文本定位为主，`{stat['fallback']}` 条走 evidence-BM25 兜底"
             f"（占比 {stat['fallback']/max(denom['all'],1):.0%}，兜底会略偏乐观，已单列）")
    L.append("")
    for g in ("normal", "hard", "all"):
        d = denom[g]
        L.append(f"### group = {g}（分母 {d}）")
        L.append(f"| 检索器 | R@{kk[0]} | R@{kk[1]} | R@{kk[2]} | MRR@{args.mrr_k} |")
        L.append("|---|---|---|---|---|")
        for c in cols:
            if d == 0:
                L.append(f"| {c} | — | — | — | — |")
                continue
            rr = [f"{hit[g][c][k]/d:.3f}" for k in kk]
            mm = f"{mrr[g][c]/d:.3f}"
            tag = " ← 现状" if c == "hyb0.50" else ""
            L.append(f"| {c} | {' | '.join(rr)} | {mm} |{tag}")
        L.append("")
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print("已写", args.out, f"| mapped/all={sum(denom.values())//2} α={args.alphas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
