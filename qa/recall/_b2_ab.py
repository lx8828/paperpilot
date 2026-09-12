"""B2 A/B：段落流均匀重打包（跨标题自由打包） vs 现状按标题切块。可断点续跑。

B2 语义（彻底解耦 report 与检索）：
    - 正文段流：full_text 全部段落（滤 PREAMBLE），段落为原子、顺序不变；
    - 均匀打包：目标 ~target 字符，跨叶标题自由并流（细标题 3.1.1/3.1.2 正文并入邻块）；
    - title_path：记块首段所属标题（溯源展示用），块间不保证标题一致；
    - References/Acknowledgments 段与正文隔离（不混块，但 ref 段之间可自由打包）；
    - 单段超 1.5×target：自成一块（段落原子，不硬拆）。

断点续跑：
    - B2 向量按论文落盘 assets/artifacts/out_views/<stem>.b2_t<target>.cvec.npy + .sig.json
      （sig=每块 sha1 全文；与当前 repack 不一致即重建）→ 中断后再跑只补缺失篇；
    - 每题结果追加 qa/recall/_b2_rows_t<target>.jsonl（qid 已存在则跳过）；
    - 全部跑完用 --report 汇总（此时向量已缓存，秒级）。

用法：
    uv run python qa/recall/_b2_ab.py --target 1600 --limit-papers 40   # 分批续跑
    uv run python qa/recall/_b2_ab.py --target 1600 --report            # 汇总
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
from paperpilot.qasper_source import (_clean_para, _split_title,  # noqa: E402
                                      gold_answer, load_papers)

VIEW = ROOT / "assets/artifacts/out_views"
ROWS_DIR = ROOT / "qa" / "recall"


def repack_paper(paper: dict, target: int) -> list[dict]:
    """QASPER full_text → 均匀段落块 [{title_path, text, n_blocks}]。"""
    flow: list[tuple[list[str], str]] = []
    for sec in paper.get("full_text") or []:
        parts = _split_title(sec.get("section_name") or "")
        path = parts or ["(PREAMBLE)"]
        if path == ["(PREAMBLE)"]:
            continue
        for p in (sec.get("paragraphs") or []):
            cl = _clean_para(p)
            if cl:
                flow.append((list(path), cl))
    chunks: list[dict] = []
    buf: list[str] = []
    buf_path: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_path, buf_len
        if buf:
            chunks.append({"title_path": list(buf_path), "text": "\n".join(buf),
                           "n_blocks": len(buf)})
        buf, buf_path, buf_len = [], [], 0

    for path, para in flow:
        leaf = path[-1].lower()
        is_ref = ("reference" in leaf or "bibliograph" in leaf
                  or "acknowledg" in leaf or "appendix" in leaf)
        if is_ref and buf:
            flush()
        plen = len(para) + 1
        if plen > int(target * 1.5):
            flush()
            chunks.append({"title_path": list(path), "text": para, "n_blocks": 1})
            continue
        if buf and buf_len + plen > target:
            flush()
        if not buf:
            buf_path = list(path)
        buf.append(para)
        buf_len += plen
    flush()
    return chunks


def _sig(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def b2_vecs(pdf: str, p2: list[dict], target: int) -> np.ndarray:
    """B2 chunk 向量：落盘缓存（key=块全文 sha1），命中复用、否则 encode。"""
    stem = Path(pdf).stem
    vf = VIEW / f"{stem}.b2_t{target}.cvec.npy"
    sf = VIEW / f"{stem}.b2_t{target}.sig.json"
    sigs = [_sig(c["text"]) for c in p2]
    if vf.exists() and sf.exists():
        try:
            old = json.loads(sf.read_text(encoding="utf-8"))
            if old == sigs:
                return np.load(vf)
        except Exception:  # noqa: BLE001 损坏则重建
            pass
    vecs = encode_texts([c["text"] for c in p2])
    VIEW.mkdir(parents=True, exist_ok=True)
    np.save(vf, vecs)
    sf.write_text(json.dumps(sigs), encoding="utf-8")
    return vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--limit-papers", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())
    rows_file = ROWS_DIR / f"_b2_rows_t{args.target}.jsonl"

    if args.report:
        rows = [json.loads(l) for l in rows_file.read_text(encoding="utf-8").splitlines()
                if l.strip()] if rows_file.exists() else []
        # 对齐 recall_set 题目序
        qorder = {it["qid"]: (it["group"], it["pid"]) for it in data["items"]}
        idx = {r["qid"]: r for r in rows}
        miss = 0
        # 汇总
        def fresh():
            return {"hit": {8: 0, 12: 0, 16: 0}, "mrr": 0.0, "ndcg": 0.0,
                    "top1": 0, "d": 0, "miss12": 0}
        acc = {"base": fresh(), "b2": fresh()}
        pairs = []
        for it in data["items"]:
            r = idx.get(it["qid"])
            if not r or (r.get("base") is None and r.get("b2") is None):
                miss += 1
                continue
            pairs.append(r)
            for name in ("base", "b2"):
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
        L = ["# B2 A/B：段落流均匀重打包 vs 现状按标题切（检索尺子，官方口径）", ""]
        L.append(f"> target={args.target} ｜ 论文 {len(pids)} ｜ 已记录题目 {len(rows)}（缺 {miss}）")
        L.append("| 策略 | R@8 | R@12 | R@16 | MRR@16 | NDCG@12 | gold@1 | miss@12 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for n in ("base", "b2"):
            a = acc[n]
            d = a["d"] or 1
            L.append(f"| {n} | {a['hit'][8]/d:.3f} | {a['hit'][12]/d:.3f} | "
                     f"{a['hit'][16]/d:.3f} | {a['mrr']/d:.3f} | {a['ndcg']/d:.3f} | "
                     f"{a['top1']/d:.3f} | {a['miss12']/d:.3f} |")
        # 块分布
        b2lens = []
        for pid in pids:
            p2 = repack_paper(papers[pid], args.target)
            b2lens += [len(c["text"]) for c in p2]
        base_lens = []
        for pid in pids:
            for c in ordered_chunks(f"qasper_{pid}.qpdf"):
                base_lens.append(len(c.text))
        def q9(xs):
            return sorted(xs)[min(len(xs) - 1, int(0.9 * len(xs)))]
        L.append("")
        L.append(f"> 现状块：{len(base_lens)}，字符中位 {med(base_lens):.0f} | p90 {q9(base_lens)}")
        L.append(f"> B2 块：{len(b2lens)}，字符中位 {med(b2lens):.0f} | p90 {q9(b2lens)}")
        # 逐题位移
        diff = [(r, (r["base"] if r["base"] is not None else 99)
                 - (r["b2"] if r["b2"] is not None else 99)) for r in pairs]
        gain = [r for r, dv in diff if dv > 0]
        lose = [r for r, dv in diff if dv < 0]
        L.append("")
        L.append(f"> 逐题：变好 {len(gain)} / 变差 {len(lose)} / 持平 "
                 f"{len(pairs)-len(gain)-len(lose)}")
        def short(r):
            return f"  {r['qid'][:10]} [{r.get('grp','')}] base=" + (
                "miss" if r["base"] is None else f"#{r['base']+1}") + " b2=" + (
                "miss" if r["b2"] is None else f"#{r['b2']+1}")
        lose.sort(key=lambda r: -((r["b2"] if r["b2"] is not None else 99)
                                  - (r["base"] if r["base"] is not None else 99)))
        gain.sort(key=lambda r: -((r["base"] if r["base"] is not None else 99)
                                  - (r["b2"] if r["b2"] is not None else 99)))
        L.append("")
        L.append("### B2 变差 Top15")
        for r in lose[:15]:
            L.append(short(r))
        L.append("")
        L.append("### B2 变好 Top15")
        for r in gain[:15]:
            L.append(short(r))
        txt = "\n".join(L)
        print(txt, flush=True)
        Path(f"qa/recall/B2_AB_t{args.target}.md").write_text(txt + "\n", encoding="utf-8")
        return 0

    # ── 分批续跑 ──────────────────────────────
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
            p2 = repack_paper(papers[pid], args.target)
            vecs_b = ChunkIndex(pdf).vectors()          # base 缓存（已有）
            vecs_2 = b2_vecs(pdf, p2, args.target)      # B2 缓存（缺失才 encode）
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
                                       "base": fb, "b2": f2}) + "\n")
                fout.flush()
                done_qids.add(it["qid"])
                paper_new += 1
                n_new += 1
            print(f"  [{pno}/{len(pids)}] {pid[:8]} 新增{paper_new}题 "
                  f"累计{n_new} t={time.time()-t0:.0f}s", flush=True)
    print(f"本次新增 {n_new} 题。再跑 --report 汇总。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
