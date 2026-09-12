"""临时：大块干扰探针——查证三条假设（base 检索，官方口径，零新 encode）。

假设（用户提出，逐个验证）：
  H1 大块噪声多、embedding 被拉向均值 → 用"gold evidence 只占其所在大块极小比例"作代理；
  H2 大块降低检索精度 → 按 gold 块大小分组 MRR/NDCG/gold@1（已有 B2_CAUSE 数据，这里复核+细）；
  H3 大块因含词多/句多总在 topK 高位（霸榜）→ 关键新证据：
     比较"全库大块占比" vs "top12 命中中大块占比"。若后者显著更高 → 霸榜成立，
     并量化"霸榜但非 gold 的大块挤占了几个名额"。

输出：
  1) 大块(>2500 字符)在全库占比 vs 在 top12 占比（vector 路 / bm25 路 / 混合路）
  2) gold 落大块的题：gold evidence 占该块字符比例（稀释度）；按稀释度分组看 MRR
  3) 按 gold 块大小分组：top12 内无关大块数、gold rank
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

BIG = 2500
K = 12


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    # 聚合器
    frac_all_big = []            # 每篇全库大块占比
    topk_big = {"vec": [], "bm": [], "hyb": []}   # 每题 top12 大块占比
    rows = []                    # 每题一行
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
        big = [len(c.text) > BIG for c in chunks]
        frac_all_big.append(sum(big) / n)
        vecs = ChunkIndex(pdf).vectors()
        bm = BM25Index([c.text for c in chunks])
        ntexts = [ree.norm(c.text) for c in chunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _ = ree.locate_gold(ntexts, ev)
            if not cg:
                continue
            gi = next(iter(cg))
            glen = len(chunks[gi].text)
            # gold evidence 在块中的字符占比（稀释代理）：evidence 若子串命中
            evn = ree.norm(ev)
            denom_frac = None
            if evn in ntexts[gi]:
                denom_frac = len(evn) / max(len(ntexts[gi]), 1)
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")

            def topk_big_ratio(order) -> float:
                topk = order[:K]
                return sum(1 for i in topk if big[int(i)]) / K

            order_v = list(np.argsort(-vs))
            order_b = list(np.argsort(-bs))
            rrf = np.zeros(n)
            for r, i in enumerate(order_v):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(order_b):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order_h = list(np.argsort(-rrf))
            first_h = next((p for p in range(len(order_h)) if order_h[p] in cg), None)
            topk_big["vec"].append(topk_big_ratio(order_v))
            topk_big["bm"].append(topk_big_ratio(order_b))
            topk_big["hyb"].append(topk_big_ratio(order_h))
            # top12 内无关大块数（非 gold 的大块）
            gold_is_big = glen > BIG
            topk = order_h[:K]
            n_big_in_topk = sum(1 for i in topk if big[int(i)] and i not in cg)
            rows.append({"grp": it["group"], "gold_len": glen,
                         "gold_big": gold_is_big, "dilute": denom_frac,
                         "rank": first_h, "n_big_noise": n_big_in_topk})
    L = ["# 大块干扰探针：H1 噪声稀释 / H2 精度 / H3 topK 霸榜", ""]
    L.append(f"> BIG={BIG} 字符 | K={K} | 论文 {len(frac_all_big)} | 题 {len(rows)}")
    L.append("")
    # H3: 全库占比 vs topK 占比
    m_all = np.mean(frac_all_big)
    L.append("### H3 霸榜检验：大块占比 全库 vs top12")
    L.append("| 检索路 | 全库大块占比 | top12 大块占比 | 倍率 |")
    L.append("|---|---|---|---|")
    for name in ("vec", "bm", "hyb"):
        m = np.mean(topk_big[name])
        L.append(f"| {name} | {m_all:.3f} | {m:.3f} | {m/max(m_all,1e-9):.2f}× |")
    L.append("")
    L.append("### H3b 按 gold 是否大块：top12 中'无关大块'占的名额（平均）")
    for gb in (False, True):
        sub = [r for r in rows if r["gold_big"] == gb]
        if sub:
            L.append(f"- gold{'是大块' if gb else '不是大块'}（n={len(sub)}）："
                     f"top12 内无关大块 {np.mean([r['n_big_noise'] for r in sub]):.1f} 个 / 12")
    L.append("")
    # H2: 按 gold 块大小分组
    L.append("### H2 检索精度 × gold 块大小")
    L.append("| gold块 | n | MRR | NDCG@12 | gold@1 | top12无关大块 |")
    L.append("|---|---|---|---|---|---|")
    for lo, hi, lab in [(0, 1000, "≤1000"), (1000, 2000, "1000-2000"),
                        (2000, 3000, "2000-3000"), (3000, 10**9, ">3000")]:
        sub = [r for r in rows if lo < r["gold_len"] <= hi or (lo == 0 and r["gold_len"] <= hi)]
        if lo == 0:
            sub = [r for r in rows if r["gold_len"] <= hi]
        elif hi == 10**9:
            sub = [r for r in rows if r["gold_len"] > lo]
        else:
            sub = [r for r in rows if lo < r["gold_len"] <= hi]
        d = len(sub)
        if not d:
            continue
        mrr = sum(1 / (r["rank"] + 1) for r in sub if r["rank"] is not None and r["rank"] < 16) / d
        ndcg = sum(1 / math.log2(r["rank"] + 2) for r in sub if r["rank"] is not None and r["rank"] < 12) / d
        top1 = sum(1 for r in sub if r["rank"] == 0) / d
        noise = np.mean([r["n_big_noise"] for r in sub])
        L.append(f"| {lab} | {d} | {mrr:.3f} | {ndcg:.3f} | {top1:.3f} | {noise:.1f} |")
    L.append("")
    # H1: gold 落大块时的稀释
    gb = [r for r in rows if r["gold_big"] and r["dilute"] is not None]
    L.append("### H1 噪声稀释：gold 落大块(>2500)时，evidence 占该块字符比例")
    if gb:
        ds = sorted(r["dilute"] for r in gb)
        L.append(f"- 有效 n={len(gb)}：evidence 占块 中位 {np.median(ds)*100:.1f}% | "
                 f"p25 {ds[len(ds)//4]*100:.1f}% | 均值 {np.mean(ds)*100:.1f}%")
        L.append("  → 若中位<10%：大块 90% 是无关内容，向量确实被稀释（H1 成立）")
        # 稀释度 vs rank 相关
        rr = [r["rank"] for r in gb if r["rank"] is not None]
        dd = [r["dilute"] for r in gb if r["rank"] is not None]
        if len(rr) > 3:
            corr = float(np.corrcoef(rr, dd)[0, 1])
            L.append(f"- 稀释度 × gold rank 相关 {corr:+.2f}（正=越稀释越靠后）")
    L.append("")
    L.append("> 对照：小块 evidence 往往占 30-100%。")
    txt = "\n".join(L)
    print(txt, flush=True)
    Path("qa/recall/BIG_PROBE_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
