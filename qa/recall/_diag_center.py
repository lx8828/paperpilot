"""圆心质量离线诊断：L2 失败是"检索层没召回答案块"还是"召回了但圆心没选对"？

对 67 题 L2 目标群逐题（无 LLM）：
    gold_chunk = 含 gold evidence 句的正文 chunk（句子 60-char key 子串命中，取命中最多）
    L1 池      = retrieve_claims top12 的 rep-claim chunk_id 集合
    targets    = 历史 judge_l1 输出（ab_l2target_result 的 A.j1_targets）
    圆心(每target节) = 该节内 L1 最高 score 命中 claim 的 chunk_id（= _pick_centers 逻辑）
指标：
    a) gold_chunk ∈ L1 池（chunk 级召回）
    b) gold_chunk 的节 ∈ targets（节级命中）
    c) 圆心 == gold_chunk；圆心与 gold_chunk 同节距离（块数差）/不同节
分桶（A挂B过=净亏 / A过B挂=净赢 / 双过 / 双挂）看圆心质量差异 → 判断该修哪层。
"""
from __future__ import annotations
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import ordered_chunks, section_from_path  # noqa: E402
from paperpilot.agents.nodes.retrieve import retrieve_claims  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

VIEW = ROOT / "assets/artifacts/out_views"


def ev_sent_keys(ev: str, k: int = 60) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s", ev or "") if len(s.strip()) > 25]
    keys = []
    for s in sents[:8]:
        key = s[:k].lower()
        if key not in keys:
            keys.append(key)
    return keys


def locate_gold_chunk(chunks, ev: str):
    """返回含 gold evidence 的 chunk（句子 key 命中数最多者）。"""
    keys = ev_sent_keys(ev)
    if not keys:
        return None, 0
    best, best_n = None, 0
    for c in chunks:
        t = c.text.lower()
        n = sum(1 for k in keys if k in t)
        if n > best_n:
            best, best_n = c, n
    return best, best_n


def main() -> int:
    papers = load_papers()
    recs = json.load(open("qa/recall/ab_l2target_result.json", encoding="utf-8"))
    items = [r for r in recs if r["l2_target"]]

    stats = {}
    for it in items:
        qid = it["qid"]
        pid = next((p for p, pp in papers.items()
                    if any(str(q.get("question_id") or "") == qid for q in (pp.get("qas") or []))), None)
        q = next(q for q in papers[pid].get("qas") or [] if str(q.get("question_id") or "") == qid)
        _, ev = gold_answer(q)
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        # 含 gold 的正文块
        gchunk, gn = locate_gold_chunk(chunks, ev)
        # L1 检索（离线，同检索器）
        ret = retrieve_claims({"question": q.get("question", ""), "pdf": pdf}).get("retrieved") or []
        l1_pool = {r.get("chunk_id") for r in ret if r.get("chunk_id")}
        # 历史 judge_l1 targets
        targets = list(it.get("A", {}).get("j1_targets") or [])
        sec2best = {}
        for r in ret:
            sec = r.get("home_section") or ""
            if sec not in sec2best or float(r.get("score", 0)) > sec2best[sec][0]:
                sec2best[sec] = (float(r.get("score", 0)), r.get("chunk_id") or "")
        centers = [sec2best[s][1] for s in targets if s in sec2best]
        gold_sec = section_from_path(list(gchunk.title_path)) if gchunk else ""
        # 圆心质量
        in_pool = bool(gchunk) and (gchunk.chunk_id in l1_pool)
        sec_hit = bool(gold_sec) and gold_sec in targets
        center_hit = bool(gchunk) and (gchunk.chunk_id in centers)
        same_sec_dist = None
        if gchunk and centers and not center_hit:
            gids = [c.chunk_id for c in chunks if section_from_path(list(c.title_path)) == gold_sec]
            cids_in = [c for c in centers if c in set(gids)]
            if cids_in and gold_sec:
                try:
                    gi = gids.index(gchunk.chunk_id)
                    ci = gids.index(cids_in[0])
                    same_sec_dist = abs(gi - ci)
                except ValueError:
                    pass
        row = {"in_pool": in_pool, "sec_hit": sec_hit, "center_hit": center_hit,
               "same_sec_dist": same_sec_dist, "gold_sec": bool(gold_sec),
               "n_centers": len(centers), "gn": gn}
        stats[qid] = {"row": row, "bucket": "净亏(A挂B过)" if not it["A"]["pass"] and it["B"]["pass"]
                      else "净赢(A过B挂)" if it["A"]["pass"] and not it["B"]["pass"]
                      else "双过" if it["A"]["pass"] and it["B"]["pass"] else "双挂",
                      "qid": qid}
        print(f"{qid[:10]} pool={int(in_pool)} sec={int(sec_hit)} center={int(center_hit)} "
              f"dist={same_sec_dist}  {stats[qid]['bucket']}", flush=True)

    print()
    print("=" * 100)
    print("汇总（按桶）：n | L1池含gold块% | targets含gold节% | 圆心=gold块% | "
          "同节距(非0圆心|均值) | gold块未定位")
    for bucket in ("净亏(A挂B过)", "净赢(A过B挂)", "双过", "双挂"):
        sub = [v for v in stats.values() if v["bucket"] == bucket]
        if not sub:
            continue
        n = len(sub)
        ip = sum(1 for v in sub if v["row"]["in_pool"])
        sh = sum(1 for v in sub if v["row"]["sec_hit"])
        ch = sum(1 for v in sub if v["row"]["center_hit"])
        no_gold = sum(1 for v in sub if not v["row"]["gold_sec"] or v["row"]["gn"] == 0)
        dists = [v["row"]["same_sec_dist"] for v in sub if v["row"]["same_sec_dist"] is not None]
        dmean = round(sum(dists) / len(dists), 1) if dists else None
        print(f"{bucket}: n={n} | L1池含gold {ip/n:.0%} | 节命中 {sh/n:.0%} | "
              f"圆心命中 {ch/n:.0%} | 同节均距 {dmean} | gold未定位 {no_gold}")
    print("=" * 100)
    print("解读：L1池含gold低 → 检索/claims覆盖问题；圆心命中低但池含gold高 → 圆心选择(策略④/投票)；"
          "圆心同节距大 → 扩窗半径不足。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
