"""临时：解剖 hard 组 gold 的检索位次——gold 排第几、前面压着哪些块。

复用 cli/run_retrieval_eval.py 的 norm/locate_gold（同口径），对 recall_set hard 题
算 hybrid(α=0.5 现状) 全序 rank；输出 gold_rank 分布 + 压位块 top_section 类型统计
+ 每题 top6 明细（chunk_id/section/文本首 50 字 + gold 标记），供人看"通用段是否挤占"。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402  (norm / locate_gold)

from paperpilot.agents.document_cache import ordered_chunks, top_section  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    hard = [it for it in data["items"] if it["group"] == "hard"]
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in hard:
        by_pid.setdefault(it["pid"], []).append(it)

    rows = []          # 每题一行（gold_rank / top6）
    rank_all = []      # gold 全序 rank
    rank_top = []      # gold 是否在 top12
    press_section: dict[str, int] = {}   # gold 前压位块的 top_section
    n_no_locate = 0
    detail = []
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n = len(chunks)
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
            cg, _diag = ree.locate_gold(ntexts, ev)
            if not cg:
                n_no_locate += 1
                continue
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            order_v = np.argsort(-vs, kind="stable")
            order_b = np.argsort(-bs, kind="stable")
            rv = np.empty(n); rb = np.empty(n)
            rv[order_v] = np.arange(n); rb[order_b] = np.arange(n)
            rrf = 0.5 / (60 + rv + 1) + 0.5 / (60 + rb + 1)
            order = np.argsort(-rrf, kind="stable")
            # gold rank（0-based）
            gr = next((int(p) for p in range(n) if int(order[p]) in cg), None)
            if gr is None:
                n_no_locate += 1
                continue
            rank_all.append(gr)
            rank_top.append(gr < 12)
            top = []
            for j, idx in enumerate(order[: max(gr + 1, 6)]):
                c = chunks[int(idx)]
                is_g = int(idx) in cg
                top.append({"rank": j + 1, "cid": c.chunk_id, "sec": top_section(c),
                            "gold": is_g, "txt": c.text[:50].replace("\n", " ")})
            # 压位块：gold 之前的块
            for j, idx in enumerate(order[:gr]):
                c = chunks[int(idx)]
                sec = top_section(c)
                press_section[sec] = press_section.get(sec, 0) + 1
            rows.append({"pid": pid[:8], "qid": it["qid"][:10], "gold_rank": gr + 1,
                         "q": it["question"][:70], "top": top})

    nq = len(rows)
    print(f"hard 可定位 {nq} 题（no-locate {n_no_locate}）")
    print(f"gold 在 top12 内: {sum(rank_top)}/{nq} ({sum(rank_top)/nq:.0%})")
    print(f"gold rank 均值 {sum(rank_all)/nq:.2f} 中位 {sorted(rank_all)[nq//2]+1}（1-based）")
    from collections import Counter
    rk = Counter(r["gold_rank"] for r in rows)
    print("gold rank 分布(1-based):", {k: rk[k] for k in sorted(rk)})
    print(f"gold=1: {rk.get(1,0)} | =2-3: {sum(rk.get(k,0) for k in (2,3))} | >=4: "
          f"{sum(v for k,v in rk.items() if k>=4)}")
    print("—" * 60)
    print("gold 前压位块的 top_section 统计（仅 gold_rank>1 的题）:")
    for s, cnt in sorted(press_section.items(), key=lambda x: -x[1])[:15]:
        print(f"  {s!r}: {cnt}")
    print("=" * 60)
    # 输出明细到文件供人工看（gold_rank<=6 全列 + 其余示例）
    Path("qa/recall/_gold_rank_detail.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    # 打印前 20 题 gold 位次与压位块
    for r in rows[:20]:
        pre = [f"{t['rank']}:{t['sec'] or '(no-sec)'}{'★GOLD' if t['gold'] else ''}"
               for t in r["top"] if not t["gold"] or t["rank"] <= r["gold_rank"]]
        print(f"[{r['gold_rank']:>2}] {r['qid']} | {r['q'][:45]}")
        print(f"     压位/序列: {pre[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
