"""临时：切块质量审计——chunk 粒度是不是排序难的一个原因？

核心问题：评测 chunk = 整章节一坨（≤4000 字符，按段落打包）。
如果 gold 常落在"大而杂"的块里，embedding 被同块他段稀释 → 排序被拖低，
说明切细（段落级/更小窗口）有空间；若 gold 块普遍小而准、仍排不高 → 问题不在切块。

审计项：
  A. recall_set 全部论文的 chunk 字符数/段落数分布、超大块占比
  B. 每题 gold 所在 chunk 的尺寸 → 按 gold 块大小分组 MRR/NDCG/hit@12
  C. gold 块 段数(同节段落打包数) 分组
  D. 抽样超大 gold 块的原文开头，人工看"稀释"是否成立
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

BIG = 2500  # "大块"判定阈值（字符）


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    # A: chunk 尺寸分布
    all_lens: list[int] = []
    all_blocks: list[int] = []
    rows: list[dict] = []   # 每题一行
    n_big = 0
    n_ch = 0
    sample_big: list[str] = []   # 超大 gold 块样本
    for pid, its in by_pid.items():
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n_ch += len(chunks)
        for c in chunks:
            all_lens.append(len(c.text))
            all_blocks.append(c.n_blocks)
        n_big += sum(1 for c in chunks if len(c.text) > BIG)
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
                try:
                    best = int(np.argmax(np.asarray(bm.score(ree.norm(ev)), dtype="float64")))
                    cg = {best}
                except Exception:
                    continue
            if not cg:
                continue
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = np.asarray(bm.score(it["question"]), dtype="float64")
            rrf = np.zeros(len(chunks))
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order = list(np.argsort(-rrf))
            first = next((p for p in range(len(order)) if order[p] in cg), None)
            g = next(iter(cg))
            gc = chunks[g]
            rows.append({"grp": it["group"], "rank": first,
                         "glen": len(gc.text), "gblocks": gc.n_blocks,
                         "gpart": gc.part or "",
                         "sec": gc.title_path[-1] if gc.title_path else "",
                         "q": it["question"][:90],
                         "pid": pid[:8]})
            if len(gc.text) > BIG and len(sample_big) < 5 and first is not None and first > 2:
                sample_big.append(f"[{it['group']}][rank#{first+1}] {it['question'][:70]}\n"
                                  f"    sec={gc.title_path[-1]} len={len(gc.text)} blocks={gc.n_blocks} "
                                  f"part={gc.part}\n"
                                  f"    {gc.text[:220].replace(chr(10),' ')}")

    def pct(xs: list[int], p: float) -> int:
        return int(sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))])

    L = ["# 切块质量审计（base 检索，官方口径，250 题）", ""]
    L.append(f"> chunk 总数 {n_ch} | 超大块(>{BIG}字符) {n_big} ({n_big/n_ch*100:.0f}%)")
    L.append("")
    L.append("### A. chunk 尺寸分布（全部 recall_set 论文）")
    L.append(f"- 字符数：中位 {pct(all_lens,50)} | p75 {pct(all_lens,75)} | p90 {pct(all_lens,90)} | max {max(all_lens)}")
    L.append(f"- 段落数：中位 {pct(all_blocks,50)} | p75 {pct(all_blocks,75)} | p90 {pct(all_blocks,90)} | max {max(all_blocks)}")
    L.append("")

    def stats(sub: list[dict], name: str):
        d = len(sub)
        if not d:
            return
        mrr = sum(1.0 / (r["rank"] + 1) for r in sub if r["rank"] is not None and r["rank"] < 16) / d
        ndcg = sum(1.0 / math.log2(r["rank"] + 2) for r in sub if r["rank"] is not None and r["rank"] < 12) / d
        hit12 = sum(1 for r in sub if r["rank"] is not None and r["rank"] < 12) / d
        top1 = sum(1 for r in sub if r["rank"] == 0) / d
        L.append(f"| {name} | {d} | {mrr:.3f} | {ndcg:.3f} | {hit12:.3f} | {top1:.3f} |")

    L.append("### B. 按 gold 所在 chunk 的字符数分组")
    L.append("| 分组 | n | MRR | NDCG@12 | hit@12 | gold@1 |")
    L.append("|---|---|---|---|---|---|")
    stats([r for r in rows if r["glen"] <= 1000], "gold块 ≤1000字符")
    stats([r for r in rows if 1000 < r["glen"] <= 2500], "gold块 1001-2500")
    stats([r for r in rows if 2500 < r["glen"] <= 4000], "gold块 2501-4000")
    stats([r for r in rows if r["glen"] > 4000], "gold块 >4000（多段二级切）")
    L.append("")
    L.append("### C. 按 gold 所在 chunk 的段落数分组")
    L.append("| 分组 | n | MRR | NDCG@12 | hit@12 | gold@1 |")
    L.append("|---|---|---|---|---|---|")
    stats([r for r in rows if r["gblocks"] == 1], "gold块=1段")
    stats([r for r in rows if r["gblocks"] == 2], "gold块=2段")
    stats([r for r in rows if 3 <= r["gblocks"] <= 4], "gold块=3-4段")
    stats([r for r in rows if r["gblocks"] >= 5], "gold块≥5段")
    L.append("")
    L.append("### D. gold 落在超大块的样例（rank 靠后）")
    for s in sample_big:
        L.append("- " + s.replace("\n", "\n  "))
    L.append("")

    # 相关强度
    import numpy as _np
    rr = [r["rank"] for r in rows if r["rank"] is not None]
    ll = [r["glen"] for r in rows if r["rank"] is not None]
    if len(rr) > 5:
        corr = _np.corrcoef(rr, ll)[0, 1]
        L.append(f"> gold块字符数 × gold rank 相关系数：{corr:+.3f} "
                 f"（正=块越大排越靠后）")
    txt = "\n".join(L)
    print(txt, flush=True)
    Path("qa/recall/CHUNK_AUDIT_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
