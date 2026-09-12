"""S_merge A/B：只并超小块、不切任何健康块（base 保留）vs 现状 base。

规则（与 _merge_plan 同语义，确定性）：
    base chunks 顺序遍历；块 len < TH 且非末块 → 并入下一个非 merged 接收块；
    接收块超 cap(4500) → 放弃并入（该碎块原样保留，dropped）；
    接收块 text = 各被并块文本（按正文序在前）+ 自身文本；title_path 取接收块。
    不产生任何"新切点"——健康块(≥TH)原样不动（除非被前向碎块并入而变大）。

断点续跑：独立缓存 assets/artifacts/out_views/<stem>.mrg_tTH.cvec.npy(.sig.json)；rows 追加
    _mrg_rows_tTH.jsonl；--report 汇总。

用法：
    uv run python qa/recall/_merge_ab.py --th 900 --limit-papers 40
    uv run python qa/recall/_merge_ab.py --th 900 --report
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import (BM25Index, ChunkIndex,  # noqa: E402
                                        encode_texts, encode_query)
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

VIEW = ROOT / "assets/artifacts/out_views"
ROWS_DIR = ROOT / "qa" / "recall"
# cap=4000：合并后不造 base 本不存在的 >4000 巨块（base 原始仅 1 块>4000），
# 且与 chunker MAX_CHUNK_LEN 一致（2026-09-10 cap 权衡扫描：4000 灭 90% 碎块、0 新巨块）
CAP = 4000


def merge_chunks(chunks, th: int) -> list[dict]:
    n = len(chunks)
    lens = [len(c.text) for c in chunks]
    merged: set[int] = set()
    recv_src: dict[int, list[int]] = {}
    i = 0
    while i < n:
        if i in merged or lens[i] >= th:
            i += 1
            continue
        # 找下一个非 merged 接收者（向后；到尾部则向前）
        j = i + 1
        while j < n and j in merged:
            j += 1
        forward = j < n
        if not forward:
            j = i - 1
            while j >= 0 and j in merged:
                j -= 1
        if j < 0:
            i += 1
            continue
        if lens[j] + lens[i] > CAP:
            i += 1
            continue
        recv_src.setdefault(j, []).append(i)
        lens[j] += lens[i]
        merged.add(i)
        i += 1
    out: list[dict] = []
    for idx in range(n):
        if idx in merged:
            continue
        srcs = sorted(recv_src.get(idx, []))
        parts = []
        tp = list(chunks[idx].title_path)
        for s in srcs:
            parts.append(chunks[s].text)
            if s < idx and not tp:
                tp = list(chunks[s].title_path)
        parts.append(chunks[idx].text)
        out.append({"title_path": tp, "text": "\n".join(parts),
                    "n_blocks": chunks[idx].n_blocks + len(srcs)})
    return out


def _sig(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def mrg_vecs(pdf: str, p2: list[dict], th: int) -> np.ndarray:
    stem = Path(pdf).stem
    vf = VIEW / f"{stem}.mrg_t{th}.cvec.npy"
    sf = VIEW / f"{stem}.mrg_t{th}.sig.json"
    sigs = [_sig(c["text"]) for c in p2]
    if vf.exists() and sf.exists():
        try:
            if json.loads(sf.read_text(encoding="utf-8")) == sigs:
                return np.load(vf)
        except Exception:  # noqa: BLE001
            pass
    vecs = encode_texts([c["text"] for c in p2])
    VIEW.mkdir(parents=True, exist_ok=True)
    np.save(vf, vecs)
    sf.write_text(json.dumps(sigs), encoding="utf-8")
    return vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=int, default=900)
    ap.add_argument("--limit-papers", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())
    rows_file = ROWS_DIR / f"_mrg_rows_t{args.th}.jsonl"

    if args.report:
        rows = [json.loads(l) for l in rows_file.read_text(encoding="utf-8").splitlines()
                if l.strip()] if rows_file.exists() else []
        idx = {r["qid"]: r for r in rows}
        miss = 0

        def fresh():
            return {"hit": {8: 0, 12: 0, 16: 0}, "mrr": 0.0, "ndcg": 0.0,
                    "top1": 0, "d": 0, "miss12": 0}
        acc = {"base": fresh(), "merge": fresh()}
        pairs = []
        for it in data["items"]:
            r = idx.get(it["qid"])
            if not r or (r.get("base") is None and r.get("merge") is None):
                miss += 1
                continue
            pairs.append(r)
            for name in ("base", "merge"):
                a = acc[name]
                f = r.get(name)
                a["d"] += 1
                if f is None:
                    a["miss12"] += 1
                    continue
                if f == 0:
                    a["top1"] += 1
                for k in (8, 12, 16):
                    if f < k:
                        a["hit"][k] += 1
                if f < 16:
                    a["mrr"] += 1.0 / (f + 1)
                if f < 12:
                    a["ndcg"] += 1.0 / math.log2(f + 2)
        med = lambda xs: float(np.median(xs)) if xs else 0  # noqa: E731
        L = [f"# S_merge A/B：只并 <{args.th} 超小块（base 其余不动）", ""]
        L.append(f"> 论文 {len(pids)} | 记录题 {len(rows)}（缺 {miss}）")
        L.append("| 策略 | R@8 | R@12 | R@16 | MRR@16 | NDCG@12 | gold@1 | miss@12 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for n in ("base", "merge"):
            a = acc[n]
            d = a["d"] or 1
            L.append(f"| {n} | {a['hit'][8]/d:.3f} | {a['hit'][12]/d:.3f} | "
                     f"{a['hit'][16]/d:.3f} | {a['mrr']/d:.3f} | {a['ndcg']/d:.3f} | "
                     f"{a['top1']/d:.3f} | {a['miss12']/d:.3f} |")
        # 块分布（merge 后）
        mlen = []
        for pid in pids:
            mlen += [len(c["text"]) for c in
                     merge_chunks(ordered_chunks(f"qasper_{pid}.qpdf"), args.th)]
        L.append("")
        L.append(f"> merge 后块数 {len(mlen)}（base 3247）| 字符 中位 {med(mlen):.0f} | "
                 f"p90 {sorted(mlen)[min(len(mlen)-1,int(0.9*len(mlen)))]} | <900 剩 "
                 f"{sum(1 for x in mlen if x<args.th)}")
        diff = [(r, (r["base"] if r["base"] is not None else 99)
                 - (r["merge"] if r["merge"] is not None else 99)) for r in pairs]
        gain = [r for r, dv in diff if dv > 0]
        lose = [r for r, dv in diff if dv < 0]
        L.append("")
        L.append(f"> 逐题：变好 {len(gain)} / 变差 {len(lose)} / 持平 "
                 f"{len(pairs)-len(gain)-len(lose)}")
        def short(r):
            return f"  {r['qid'][:10]} [{r.get('grp','')}] base=" + (
                "miss" if r["base"] is None else f"#{r['base']+1}") + " merge=" + (
                "miss" if r["merge"] is None else f"#{r['merge']+1}")
        lose.sort(key=lambda r: -((r["merge"] if r["merge"] is not None else 99)
                                  - (r["base"] if r["base"] is not None else 99)))
        gain.sort(key=lambda r: -((r["base"] if r["base"] is not None else 99)
                                  - (r["merge"] if r["merge"] is not None else 99)))
        L.append("")
        L.append("### merge 变差 Top15")
        for r in lose[:15]:
            L.append(short(r))
        L.append("")
        L.append("### merge 变好 Top15")
        for r in gain[:15]:
            L.append(short(r))
        txt = "\n".join(L)
        print(txt, flush=True)
        Path(f"qa/recall/MERGE_AB_t{args.th}.md").write_text(txt + "\n", encoding="utf-8")
        return 0

    done_qids = set()
    if rows_file.exists():
        for l in rows_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done_qids.add(json.loads(l)["qid"])
    if args.limit_papers:
        pids = pids[: args.limit_papers]
    t0 = time.time()
    n_new = 0
    with open(rows_file, "a", encoding="utf-8") as fout:
        for pno, pid in enumerate(pids, 1):
            pdf = f"qasper_{pid}.qpdf"
            bchunks = ordered_chunks(pdf)
            p2 = merge_chunks(bchunks, args.th)
            vecs_b = ChunkIndex(pdf).vectors()
            vecs_2 = mrg_vecs(pdf, p2, args.th)
            bm_b = BM25Index([c.text for c in bchunks])
            bm_2 = BM25Index([c["text"] for c in p2])
            ntexts_b = [ree.norm(c.text) for c in bchunks]
            ntexts_2 = [ree.norm(c["text"]) for c in p2]
            qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
            paper_new = 0
            for it in by_pid[pid]:
                if it["qid"] in done_qids:
                    continue
                q = qas.get(it["qid"])
                if not q:
                    continue
                _g, ev = gold_answer(q)
                if not ev:
                    continue
                cg_b, _ = ree.locate_gold(ntexts_b, ev)
                if not cg_b:
                    try:
                        cg_b = {int(np.argmax(np.asarray(bm_b.score(ree.norm(ev)),
                                                         dtype="float64")))}
                    except Exception:  # noqa: BLE001
                        continue
                cg_2, _ = ree.locate_gold(ntexts_2, ev)
                if not cg_2:
                    try:
                        cg_2 = {int(np.argmax(np.asarray(bm_2.score(ree.norm(ev)),
                                                         dtype="float64")))}
                    except Exception:  # noqa: BLE001
                        continue
                qv = encode_query(it["question"])

                def first_rank(vecs, texts, bm, cg):
                    vs = (qv @ vecs.T).astype("float64")
                    bs = np.asarray(bm.score(it["question"]), dtype="float64")
                    rrf = np.zeros(len(texts))
                    for r, i in enumerate(np.argsort(-vs)):
                        rrf[int(i)] += 1.0 / (60 + r + 1)
                    for r, i in enumerate(np.argsort(-bs)):
                        rrf[int(i)] += 1.0 / (60 + r + 1)
                    order = list(np.argsort(-rrf))
                    return next((p for p in range(len(order)) if order[p] in cg), None)

                fb = first_rank(vecs_b, bchunks, bm_b, cg_b)
                f2 = first_rank(vecs_2, p2, bm_2, cg_2)
                fout.write(json.dumps({"qid": it["qid"], "grp": it["group"],
                                       "base": fb, "merge": f2}) + "\n")
                fout.flush()
                done_qids.add(it["qid"])
                paper_new += 1
                n_new += 1
            print(f"  [{pno}/{len(pids)}] {pid[:8]} 新增{paper_new}题 累计{n_new} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    print(f"本次新增 {n_new} 题。再跑 --report 汇总。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
