"""临时：查"排前面的是些啥"——把 gold 排位差的题，把它们 top12 的真实块内容打出来。

目的：base/merge 都排不好时，看挤在 topK 前面的是哪类块（章节类别/大小/是否与 gold 同节），
      以判断"排不准"的机制是 (a) 同节错块 还是 (b) 跨节无关块 还是 (c) 噪声块（Ref/Appendix）。

输出：
  A) gold rank 直方图（base vs merge）
  B) top1 与 gold 的"节关系"：同节 / 同大节 / 不同节 / 噪声节
  C) 最差 25 题的 top6 明细（节 / 长度 / RRF分 / 文本首 90 字），GOLD 标注
零新 encode（base 用 ChunkIndex 缓存，merge 用 mrg 缓存）。
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

VIEW = ROOT / "assets/artifacts/out_views"
NOISE = ("reference", "bibliograph", "acknowledg", "appendix", "supplement")


def leaf(tp) -> str:
    return (list(tp)[-1] if tp else "") or ""


def is_noise(tp) -> bool:
    l = leaf(tp).lower()
    return any(k in l for k in NOISE)


def big_sec(tp) -> str:
    return (list(tp)[0] if tp else "") or ""


def load_mrg(pdf: str, th: int):
    stem = Path(pdf).stem
    vf = VIEW / f"{stem}.mrg_t{th}.cvec.npy"
    if not vf.exists():
        return None
    return np.load(vf)


def hybrid_order(qv, vecs, bm, q, n):
    vs = (qv @ vecs.T).astype("float64")
    bs = np.asarray(bm.score(q), dtype="float64")
    rrf = np.zeros(n)
    for r, i in enumerate(np.argsort(-vs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    for r, i in enumerate(np.argsort(-bs)):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    return list(np.argsort(-rrf)), rrf


def main() -> int:
    th = 900
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    hist = {"base": {}, "merge": {}}
    rel = {"same": 0, "same_big": 0, "diff": 0, "noise": 0}
    cases = []
    n_q = 0
    n_have_merge = 0
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        bchunks = ordered_chunks(pdf)
        nb = len(bchunks)
        vB = ChunkIndex(pdf).vectors()
        bmB = BM25Index([c.text for c in bchunks])
        ntB = [ree.norm(c.text) for c in bchunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _ = ree.locate_gold(ntB, ev)
            if not cg:
                try:
                    best = int(np.argmax(np.asarray(bmB.score(ree.norm(ev)), dtype="float64")))
                    cg = {best}
                except Exception:
                    continue
            gi = next(iter(cg))
            qv = encode_query(it["question"])
            order, rrf = hybrid_order(qv, vB, bmB, it["question"], nb)
            rank = next((p for p in range(len(order)) if order[p] in cg), None)
            hist["base"][rank] = hist["base"].get(rank, 0) + 1
            n_q += 1
            if rank is not None and rank > 0:
                t1 = order[0]
                if is_noise(bchunks[t1].title_path):
                    rel["noise"] += 1
                elif leaf(bchunks[t1].title_path) == leaf(bchunks[gi].title_path):
                    rel["same"] += 1
                elif big_sec(bchunks[t1].title_path) == big_sec(bchunks[gi].title_path):
                    rel["same_big"] += 1
                else:
                    rel["diff"] += 1
            if rank is None or rank >= 3:
                top = []
                for p in range(min(6, len(order))):
                    ci = int(order[p])
                    c = bchunks[ci]
                    top.append({
                        "i": p + 1, "ci": ci, "sec": leaf(c.title_path),
                        "len": len(c.text), "rrf": float(rrf[ci]),
                        "gold": ci in cg,
                        "snip": " ".join(c.text.split())[:90],
                    })
                cases.append({"qid": it["qid"][:10], "grp": it["group"],
                              "rank": rank, "q": it["question"],
                              "gold_sec": leaf(bchunks[gi].title_path),
                              "gold_len": len(bchunks[gi].text), "top": top})

    cases.sort(key=lambda r: -(r["rank"] if r["rank"] is not None else 99))
    L = ["# TOP-DUMP：gold 排位差的题，topK 前排是些啥（base 官方口径）", ""]
    L.append(f"> 题 {n_q} | merge 缓存可用 {n_have_merge}")
    L.append("")
    L.append("## A) gold rank 直方图（base）")
    L.append("| rank | 题数 |")
    L.append("|---|---|")
    for rk in sorted([k for k in hist["base"] if k is not None]):
        L.append(f"| #{rk+1} | {hist['base'][rk]} |")
    L.append(f"| miss@16 | {hist['base'].get(None, 0)} |")
    L.append("")
    L.append("## B) gold 非#1 时，top1 与 gold 的节关系")
    tot = sum(rel.values()) or 1
    L.append("| 关系 | 题数 | 占比 |")
    L.append("|---|---|---|")
    for k, lab in [("same", "同节(同 leaf 标题)"), ("same_big", "同大节不同子节"),
                   ("diff", "完全不同的节"), ("noise", "噪声节(Ref/Appendix/Supp)")]:
        L.append(f"| {lab} | {rel[k]} | {rel[k]/tot*100:.1f}% |")
    L.append("")
    L.append(f"## C) 最差 {min(25,len(cases))} 题 top6 明细")
    for r in cases[:25]:
        rk = "MISS" if r["rank"] is None else f"#{r['rank']+1}"
        L.append("")
        L.append(f"### {r['qid']} [{r['grp']}] gold={rk} gold节={r['gold_sec']!r} goldlen={r['gold_len']}")
        L.append(f"Q: {r['q'][:150]}")
        for t in r["top"]:
            mark = "  <<< GOLD" if t["gold"] else ""
            L.append(f"  [{t['i']}] rrf={t['rrf']:.4f} len={t['len']:5d} {t['sec']!r}{mark}")
            L.append(f"      {t['snip']}")
    txt = "\n".join(L)
    Path("qa/recall/TOP_DUMP_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(f"written TOP_DUMP_20260910.md | 题={n_q} | gold@1={hist['base'].get(0,0)} "
          f"| gold@2={hist['base'].get(1,0)} | gold@3={hist['base'].get(2,0)} "
          f"| miss16={hist['base'].get(None,0)}")
    print("节关系:", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
