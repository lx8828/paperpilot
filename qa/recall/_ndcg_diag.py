"""临时：NDCG 低的原因诊断——按题目宽泛度/锚点分组看 MRR/NDCG。

假设：宽泛题（泛问 approach/contribution，无专名/数字锚）检索排序天然难 → NDCG 低。
验证：按多个特征分组统计 base 检索的 MRR/NDCG，看差异是否显著。

分组维度（都是可自动判定的代理）：
  qhead : What/How/Which/Is/Do/Why...
  anchored : 问题含 大写专名 / 数字 / 引号词组 → 有强检索锚
  宽泛词 : question 命中 contribution/approach/main idea/what is/overview 类
  qlen   : 词数
交叉输出各组 NDCG/MRR/hit@12/样本数，并算"金标在 top1/top2-3/top4+"占比。
"""
from __future__ import annotations
import json
import math
import re
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

_OV_WORDS = ("main contribution", "primary contribution", "core idea", "main idea",
             "what is the paper about", "overall", "summary", "goal of",
             "what does the paper", "how does the paper", "briefly",
             "the paper propose", "does this paper", "main goal", "key idea",
             "what kind of", "what are the main", "main findings",
             "purpose of", "main method", "overview")

_NUM_RE = re.compile(r"\d")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]+\b|(?:[A-Z]{2,})")


def bucket_meta(q: str) -> dict:
    qq = q.strip()
    head = qq.split()[0].lower() if qq.split() else ""
    has_num = bool(_NUM_RE.search(qq))
    has_cap = bool(re.search(r"\b[A-Z][a-z]+[A-Z]|\b[A-Z]{2,}", qq)) or bool(
        re.search(r"\"[^\"]+\"", qq))
    ov = any(w in qq.lower() for w in _OV_WORDS)
    return {"head": head, "num": has_num, "cap": has_cap, "ov": ov,
            "qlen": len(qq.split())}


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    # 每题算 base 检索的 gold rank（0-based, 命中 top16 才算）
    per = []  # {qid,grp,meta,rank(0-based)|None}
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
            cg, _ = ree.locate_gold(ntexts, ev)
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
            rrf = np.zeros(n)
            for r, i in enumerate(np.argsort(-vs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            for r, i in enumerate(np.argsort(-bs)):
                rrf[int(i)] += 1.0 / (60 + r + 1)
            order = list(np.argsort(-rrf))
            first = next((p for p in range(len(order)) if order[p] in cg), None)
            per.append({"qid": it["qid"], "grp": it["group"], "rank": first,
                        **bucket_meta(it["question"])})

    hit = [p for p in per if p["rank"] is not None]
    denom = len(per)

    def stats(sub):
        d = len(sub)
        if d == 0:
            return None
        mrr = sum(1.0 / (p["rank"] + 1) for p in sub if p["rank"] < 16) / d
        ndcg = sum(1.0 / math.log2(p["rank"] + 2) for p in sub if p["rank"] < 12) / d
        hit12 = sum(1 for p in sub if p["rank"] is not None and p["rank"] < 12) / d
        miss = sum(1 for p in sub if p["rank"] is None or p["rank"] >= 12) / d
        top1 = sum(1 for p in sub if p["rank"] == 0) / d
        top23 = sum(1 for p in sub if p["rank"] in (1, 2)) / d
        return {"n": d, "MRR": mrr, "NDCG12": ndcg, "hit12": hit12,
                "top1": top1, "top23": top23, "miss12": miss}

    L = ["# NDCG 低的原因诊断：题目宽泛度 × 检索排序（base，官方口径）", ""]
    L.append(f"> 全部 {denom} 题（gold 可定位）；rank 为 base 混合检索首个命中位次(0-based)")
    L.append(f"| 分组 | n | MRR@16 | NDCG@12 | hit@12 | gold@1 | gold@2-3 | miss@12 |")
    L.append("|---|---|---|---|---|---|---|---|")

    def emit(name, sub):
        s = stats(sub)
        if s:
            L.append(f"| {name} | {s['n']} | {s['MRR']:.3f} | {s['NDCG12']:.3f} | "
                     f"{s['hit12']:.3f} | {s['top1']:.3f} | {s['top23']:.3f} | "
                     f"{s['miss12']:.3f} |")

    emit("ALL", per)
    emit("ALL-hit-only", hit)
    L.append("")
    # 按疑问词
    heads = {}
    for p in per:
        heads.setdefault(p["head"], []).append(p)
    L.append("### 按疑问词")
    for h in sorted(heads, key=lambda x: -len(heads[x]))[:10]:
        emit(f"{h}?", heads[h])
    # 按锚点
    L.append("### 按锚点强弱")
    emit("有数字锚", [p for p in per if p["num"]])
    emit("无数字锚", [p for p in per if not p["num"]])
    emit("有大写专名/引号", [p for p in per if p["cap"]])
    emit("无大写锚", [p for p in per if not p["cap"]])
    emit("宽泛词(contribution等)", [p for p in per if p["ov"]])
    emit("非宽泛词", [p for p in per if not p["ov"]])
    emit("无数字且无大写(最宽泛)", [p for p in per if not p["num"] and not p["cap"] and not p["ov"]])
    emit("有数字或大写(有锚)", [p for p in per if p["num"] or p["cap"]])
    # 题目长度
    L.append("### 按题目长度")
    for lo, hi, lab in [(0, 7, f"≤7词"), (8, 11, "8-11词"), (12, 99, "≥12词")]:
        emit(lab, [p for p in per if lo <= p["qlen"] <= hi])
    # normal vs hard
    L.append("### normal vs hard")
    for g in ("normal", "hard"):
        emit(g, [p for p in per if p["grp"] == g])

    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/NDCG_DIAG_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
