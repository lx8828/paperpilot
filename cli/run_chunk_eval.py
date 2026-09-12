"""chunk 质量评估：向量空间判据（尺子，零 LLM）。

目的：为"我们的切分是否合理"提供**分量化的数字**——补齐此前只有"检索结果层"
（gold rank / MRR / NDCG，见 CHUNK_AUDIT）而缺"embedding 空间层"的空白。

四组指标：
  A 内在质量（不需要 query，全量 chunk）
    A1 尺寸分布：字符/段落 percentile + CV（变异系数，衡量粒度是否均匀）
    A2 冗余/多样性：相邻块平均余弦（越高=邻块冗余）、全对平均余弦（越低=越多样）、
                   高相似块对(cos≥0.9)占比（重复块风险）
    A3 边界完整性：块尾是否断在句中、块首是否小写续接（切分是否"切坏了句子"）
    A4 块内聚度：块内**段落间**平均余弦（越高=块内容语义单一；低=一坨多主题）
  B 检索可分性（250 题，query→chunk 余弦）
    B1 分布：query→gold_best 与 query→other 的 mean/median/p10/p90/max
    B2 margin：s(gold_best) − s(other_max)（>0=gold 就是最高分）；正比例
    B3 排名：gold@1 / R@8 / R@12 / R@16、MRR@16（向量 + 混合 RRF）
    B4 可分性：池化 AUC（threshold-free）+ Cohen's d + 干扰项超越 gold 的比例
    B5 gold 与其最相似非 gold 块余弦（越高=存在"近似重复块"，gold 不唯一）
  C 切分策略对照（同 gold 口径）：现状 vs 固定窗口 512/1024/2048 vs 段落打包 1024
  D 跨块诊断：gold 证据跨多块的比例、gold 在块内落点（头/中/尾）

口径（与线上一致，可复现）：
- chunk = document_cache.ordered_chunks（qasper_* 走 qasper_source.build_chunks）
- 向量 = ChunkIndex(pdf).vectors()（bge-m3，已归一化，点积=cosine）
- gold → chunk 映射 = run_retrieval_eval.locate_gold（整段→按句→首锚）+ BM25 兜底
- gold = 全部人工 evidence 段并集（gold_answer_full）

用法：
    uv run python cli/run_chunk_eval.py                       # 全量
    uv run python cli/run_chunk_eval.py --limit-papers 40     # 冒烟
    uv run python cli/run_chunk_eval.py --strat-papers 60     # C 组对照论文数
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import numpy as np  # noqa: E402

import run_retrieval_eval as ree  # noqa: E402
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import (BM25Index, ChunkIndex,  # noqa: E402
                                        encode_query, encode_texts)
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

TERMINAL = set('.!?。！？”"\')]}…')
_BIN_EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
STRAT_CACHE = ROOT / "assets/artifacts/out_views" / "_strat_cache"
# 等预算召回口径：按 rank 顺序取块、累计字符达 B 即停，看 gold 是否落在已取集合内。
# 固定 k 会天然偏向"大块"（每 slot 装更多字、覆盖更大比例正文），必须用等字符预算归一。
BUDGETS = (8000, 16000, 24000)


# ── 通用统计 ──────────────────────────────────────────────────────────────────

def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, int(p / 100 * len(s)))
    return s[i]


def pooled_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """池化 ROC-AUC（Mann-Whitney，含 ties）：P(s_pos > s_neg)。"""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def mean_auc(list_scores: list[np.ndarray],
             list_labels: list[np.ndarray]) -> tuple[float, float, float]:
    """**每题各算 AUC 再平均**（mean per-query AUC）——IR 标准口径。

    不能把所有题拼成一个大数组算池化 AUC：不同题的 cosine 分数尺度不同
    （有的题整体偏高/偏低），池化会把"某题的高分干扰项"与"另一题的低分 gold"
    混在一起比，系统性低估可分性。
    返回 (mean, median, p10)。
    """
    vals = [pooled_auc(np.asarray(s), np.asarray(l))
            for s, l in zip(list_scores, list_labels)
            if np.asarray(l).any() and not np.asarray(l).all()]
    vals = [v for v in vals if v == v]  # 去 NaN
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(vals)
    return float(a.mean()), float(np.median(a)), float(pct(a.tolist(), 10))


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / max(len(pos) + len(neg) - 2, 1))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else float("nan")


def hist_lines(name: str, xs: list[float]) -> list[str]:
    if not xs:
        return [f"- {name}: （无样本）"]
    a = np.asarray(xs, dtype="float64")
    cnt, _ = np.histogram(a, bins=_BIN_EDGES)
    tot = max(len(xs), 1)
    cells = " | ".join(f"{c}({c / tot:.0%})" for c in cnt)
    edges = " | ".join(f"[{_BIN_EDGES[i]:.1f},{_BIN_EDGES[i + 1]:.1f})"
                       for i in range(len(cnt)))
    return [f"- {name} 分桶 {edges}", f"  计数 {cells}"]


# ── gold 映射（多 evidence 并集 + BM25 兜底）──────────────────────────────────

def gold_chunks(ntexts: list[str], evs: list[str],
                bm: BM25Index | None) -> tuple[set[int], bool]:
    cg: set[int] = set()
    fallback = False
    for ev in evs:
        gi, _ = ree.locate_gold(ntexts, ev)
        if not gi and bm is not None:
            try:
                gi = {int(np.argmax(np.asarray(bm.score(ree.norm(ev)), dtype="float64")))}
                fallback = True
            except Exception:  # noqa: BLE001
                gi = set()
        cg |= gi
    return cg, fallback


def rrf_rank(vec_scores: np.ndarray, bm_scores: np.ndarray) -> np.ndarray:
    n = len(vec_scores)
    rrf = np.zeros(n, dtype="float64")
    for r, i in enumerate(np.argsort(-vec_scores, kind="stable")):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    for r, i in enumerate(np.argsort(-bm_scores, kind="stable")):
        rrf[int(i)] += 1.0 / (60 + r + 1)
    return rrf


def first_gold_rank(order: np.ndarray, cg: set[int]) -> int | None:
    for p in range(len(order)):
        if int(order[p]) in cg:
            return p
    return None


# ── C 组：切分策略构造 ────────────────────────────────────────────────────────

def strat_fixed(text: str, size: int) -> list[str]:
    """硬窗口（按字符，含边界空白清理）——最朴素的切分基线。"""
    out = []
    for i in range(0, len(text), size):
        t = text[i:i + size].strip()
        if len(t) >= 40:
            out.append(t)
    return out


def strat_para(text: str, size: int) -> list[str]:
    """段落打包到 ~size（无标题感知，仅段落原子）——比硬窗口更接近我们的做法。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        if cur and cur_len + len(p) + 1 > size:
            out.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def eval_strategy(name: str, texts_by_pid: dict[str, list[str]],
                  by_pid: dict[str, list[dict]], papers: dict,
                  vecs_by_pid: dict[str, np.ndarray]) -> dict:
    """给定策略下的 chunk 文本，按同口径算 hit/MRR/AUC/margin/跨块。"""
    ks = (1, 8, 12, 16)
    hits = {k: 0 for k in ks}
    hitb = {b: 0 for b in BUDGETS}
    mrr = 0.0
    denom = 0
    cross = 0
    auc_s, auc_l = [], []
    margins = []
    map_fail = 0
    n_chunks = 0
    for pid, its in by_pid.items():
        texts = texts_by_pid.get(pid)
        if not texts:
            continue
        vecs = vecs_by_pid.get(pid)
        if vecs is None:
            # 磁盘缓存：同 (pid, 策略, 块数) 直接复用（重跑秒回）
            STRAT_CACHE.mkdir(parents=True, exist_ok=True)
            key = hashlib.md5(f"{pid}|{name}|{len(texts)}".encode()).hexdigest()
            cf = STRAT_CACHE / f"{key}.npy"
            if cf.exists():
                vecs = np.load(cf)
            else:
                vecs = encode_texts(texts)
                np.save(cf, vecs)
            vecs_by_pid[pid] = vecs
        ntexts = [ree.norm(t) for t in texts]
        tlens = [len(t) for t in texts]
        n_chunks += len(texts)
        bm = BM25Index(texts)
        for it in its:
            q = it.get("_q")
            if q is None:
                continue
            _g, evs = gold_answer_full(q)
            if not evs:
                continue
            cg, _fb = gold_chunks(ntexts, evs, bm)
            if not cg:
                map_fail += 1
                continue
            denom += 1
            if len(cg) > 1:
                cross += 1
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            order = np.argsort(-vs, kind="stable")
            first = first_gold_rank(order, cg)
            if first is not None:
                for k in ks:
                    if first < k:
                        hits[k] += 1
                if first < 16:
                    mrr += 1.0 / (first + 1)
            # 等预算召回：rank 序累计字符达 B 前是否已遇 gold
            for b in BUDGETS:
                c = 0
                for idx in order:
                    if int(idx) in cg:
                        hitb[b] += 1
                        break
                    c += tlens[int(idx)]
                    if c >= b:
                        break
            best_g = max(float(vs[i]) for i in cg)
            oth = np.asarray([float(vs[i]) for i in range(len(texts)) if i not in cg])
            lab = np.zeros(len(texts), dtype=bool)
            for i in cg:
                lab[i] = True
            auc_s.append(vs)
            auc_l.append(lab)
            if len(oth):
                margins.append(best_g - float(oth.max()))
    d = max(denom, 1)
    mauc, _mmed, _mp10 = mean_auc(auc_s, auc_l)
    return {
        "name": name, "denom": denom, "map_fail": map_fail,
        "hit@1": hits[1] / d, "hit@8": hits[8] / d, "hit@12": hits[12] / d,
        "hit@16": hits[16] / d, "mrr@16": mrr / d,
        "hitb": {b: hitb[b] / d for b in BUDGETS},
        "n_chunks": n_chunks,
        "cross": cross / d,
        "auc": mauc,
        "margin_pos": (sum(1 for m in margins if m > 0) / len(margins)) if margins else float("nan"),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="qa/recall/recall_set_v1.json")
    ap.add_argument("--out", default="qa/recall/CHUNK_METRICS_20260910.md")
    ap.add_argument("--limit-papers", type=int, default=None)
    ap.add_argument("--strat-papers", type=int, default=25, help="C 组对照论文数（0=关）")
    ap.add_argument("--cohesion-sample", type=int, default=300, help="A4 抽样块数")
    args = ap.parse_args()

    print("[chunk-eval] 读取题集…", flush=True)
    data = json.loads((ROOT / args.set).read_text(encoding="utf-8"))
    print("[chunk-eval] 加载 QASPER 语料（较大，约 10-30s）…", flush=True)
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())
    if args.limit_papers:
        pids = pids[: args.limit_papers]
        by_pid = {p: by_pid[p] for p in pids}

    print(f"[chunk-eval] 就绪：论文 {len(pids)} 篇｜题目 "
          f"{sum(len(v) for v in by_pid.values())}｜C 组 {args.strat_papers} 篇｜"
          f"A4 抽样 {args.cohesion_sample} 块｜开始 A/B…", flush=True)

    # ── A/Z 采集 ──
    lens: list[int] = []
    nblocks: list[int] = []
    adj_cos: list[float] = []
    allpair_cos: list[float] = []
    dup_pairs = 0
    total_pairs = 0
    end_mid = start_low = n_ch = 0
    cohesion: list[float] = []
    cohesion_budget = args.cohesion_sample

    # ── B 采集 ──
    gold_best: list[float] = []
    others: list[float] = []
    margins: list[float] = []
    gold_nn: list[float] = []          # gold 块 ↔ 最相似非 gold 块
    ranks_vec: list[int] = []
    ranks_hyb: list[int] = []
    n_other_above: list[float] = []
    cross_cnt = 0
    fracs: list[float] = []
    b_denom = 0
    map_fail_all = 0
    fb_all = 0
    auc_s, auc_l = [], []
    grp_stat = {"normal": {"n": 0, "top1": 0, "mrr": 0.0, "auc_s": [], "auc_l": [],
                           "gb": [], "ot": []},
                "hard": {"n": 0, "top1": 0, "mrr": 0.0, "auc_s": [], "auc_l": [],
                         "gb": [], "ot": []}}

    t0 = time.time()
    for pno, pid in enumerate(pids, 1):
        pdf = f"qasper_{pid}.qpdf"
        chunks = ordered_chunks(pdf)
        n_ch += len(chunks)
        vecs = ChunkIndex(pdf).vectors()
        n = len(chunks)
        for i, c in enumerate(chunks):
            t = c.text
            lens.append(len(t))
            nblocks.append(c.n_blocks)
            st = t.strip()
            if st and st[-1] not in TERMINAL:
                end_mid += 1
            if st and st[0].islower():
                start_low += 1
        # A2 冗余/多样性
        if n >= 2:
            sim = vecs @ vecs.T
            iu = np.triu_indices(n, k=1)
            pair = sim[iu]
            allpair_cos.extend(pair.tolist())
            total_pairs += int(len(pair))
            dup_pairs += int((pair >= 0.9).sum())
            for i in range(n - 1):
                adj_cos.append(float(sim[i, i + 1]))
        # A4 块内聚度（抽样）
        if cohesion_budget > 0:
            for c in chunks[: min(4, n)]:
                if cohesion_budget <= 0:
                    break
                paras = [p.strip() for p in c.text.split("\n") if len(p.strip()) >= 80]
                if len(paras) < 2:
                    continue
                pv = encode_texts(paras[:12])
                if len(pv) >= 2:
                    s = pv @ pv.T
                    iu = np.triu_indices(len(pv), k=1)
                    cohesion.append(float(s[iu].mean()))
                cohesion_budget -= 1

        ntexts = [ree.norm(c.text) for c in chunks]
        bm = BM25Index([c.text for c in chunks])
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        for it in by_pid[pid]:
            q = qas.get(it["qid"])
            if not q:
                continue
            it["_q"] = q
            _g, evs = gold_answer_full(q)
            if not evs:
                continue
            cg, fb = gold_chunks(ntexts, evs, bm)
            if fb:
                fb_all += 1
            if not cg:
                map_fail_all += 1
                continue
            b_denom += 1
            if len(cg) > 1:
                cross_cnt += 1
            qv = encode_query(it["question"])
            vs = (qv @ vecs.T).astype("float64")
            bs = bm.score(it["question"])
            order_v = np.argsort(-vs, kind="stable")
            order_h = np.argsort(-rrf_rank(vs, bs), kind="stable")
            rv = first_gold_rank(order_v, cg)
            rh = first_gold_rank(order_h, cg)
            if rv is not None:
                ranks_vec.append(rv)
            if rh is not None:
                ranks_hyb.append(rh)
            bg = max(float(vs[i]) for i in cg)
            oth = np.asarray([float(vs[i]) for i in range(n) if i not in cg])
            gold_best.append(bg)
            if len(oth):
                others.extend(oth.tolist())
                margins.append(bg - float(oth.max()))
                n_other_above.append(float((oth > bg).mean()))
                if len(cg) == 1:
                    gi = next(iter(cg))
                    nz = [float(vs[i]) for i in range(n) if i != gi]
                    if nz:
                        gold_nn.append(max(nz))
            lab = np.zeros(n, dtype=bool)
            for i in cg:
                lab[i] = True
            grp = it["group"]
            auc_s.append(vs); auc_l.append(lab)
            g = grp_stat.get(grp)
            if g is not None:
                g["n"] += 1
                g["auc_s"].append(vs); g["auc_l"].append(lab)
                g["gb"].append(bg); g["ot"].extend(oth.tolist())
                if rv == 0:
                    g["top1"] += 1
                if rv is not None and rv < 16:
                    g["mrr"] += 1.0 / (rv + 1)
        if pno % 10 == 0 or pno == len(pids) or pno == 1:
            el = time.time() - t0
            eta = el / pno * (len(pids) - pno)
            print(f"  [A/B {pno}/{len(pids)}] chunks={n_ch:,} q={b_denom} "
                  f"t={el:.0f}s ETA≈{eta:.0f}s", flush=True)

    # ── C 组：策略对照 ──
    strat_rows: list[dict] = []
    if args.strat_papers:
        sub_pids = list(by_pid.keys())[: args.strat_papers]
        cur_texts: dict[str, list[str]] = {}
        s512: dict[str, list[str]] = {}
        s1024: dict[str, list[str]] = {}
        s2048: dict[str, list[str]] = {}
        para: dict[str, list[str]] = {}
        for pid in sub_pids:
            full = "\n".join(c.text for c in ordered_chunks(f"qasper_{pid}.qpdf"))
            cur_texts[pid] = [c.text for c in ordered_chunks(f"qasper_{pid}.qpdf")]
            s512[pid] = strat_fixed(full, 512)
            s1024[pid] = strat_fixed(full, 1024)
            s2048[pid] = strat_fixed(full, 2048)
            para[pid] = strat_para(full, 1024)
        sub_by = {p: by_pid[p] for p in sub_pids}
        print(f"  [C 组] {len(sub_pids)} 篇 × 5 策略 encode…", flush=True)
        for nm, tt in (("现状(标题+段落打包≤4000)", cur_texts), ("固定窗口 512", s512),
                       ("固定窗口 1024", s1024), ("固定窗口 2048", s2048),
                       ("段落打包≤1024(无标题)", para)):
            print(f"    [{nm}] encode+检索…", flush=True)
            r = eval_strategy(nm, tt, sub_by, papers, {})
            strat_rows.append(r)
            print(f"    {nm}: hit@12={r['hit@12']:.3f} mrr={r['mrr@16']:.3f} "
                  f"auc={r['auc']:.3f} margin+={r['margin_pos']:.3f}", flush=True)

    # ── 报告 ──
    da = max(b_denom, 1)
    b_auc, b_auc_med, b_auc_p10 = mean_auc(auc_s, auc_l)
    L: list[str] = []
    L.append("# chunk 质量评估：向量空间判据（recall_set 250 题）")
    L.append("")
    L.append(f"> 数据 `{data.get('version')}`｜论文 {len(pids)} 篇｜chunk {n_ch:,} 个｜"
             f"题目映射成功 {b_denom}（fail {map_fail_all}，BM25 兜底 {fb_all}）")
    L.append(f"> 向量 bge-m3（已归一化，点积=cosine）｜gold=全部人工 evidence 并集｜"
             f"定位口径同 `run_retrieval_eval.locate_gold`")
    L.append("")

    L.append("## A. 内在质量（无 query）")
    L.append("")
    L.append("### A1 尺寸分布")
    L.append(f"- 字符：中位 {pct(lens,50):.0f} | p90 {pct(lens,90):.0f} | "
             f"p99 {pct(lens,99):.0f} | max {max(lens)} | "
             f"均值 {np.mean(lens):.0f} | std {np.std(lens):.0f} | "
             f"**CV {np.std(lens)/np.mean(lens):.2f}**（0.3-0.5=均匀，>1=极不均匀）")
    L.append(f"- 段落数：中位 {pct(nblocks,50):.0f} | p90 {pct(nblocks,90):.0f} | max {max(nblocks)}")
    L.append("")
    L.append("### A2 冗余 / 多样性")
    if adj_cos:
        L.append(f"- **相邻块平均余弦 {np.mean(adj_cos):.3f}**（越高=相邻块越冗余）｜"
                 f"中位 {np.median(adj_cos):.3f}｜p90 {pct(adj_cos,90):.3f}")
    if allpair_cos:
        L.append(f"- **全对平均余弦 {np.mean(allpair_cos):.3f}**"
                 f"（多样性 = {1 - np.mean(allpair_cos):.3f}）｜"
                 f"配对 {total_pairs:,} 对")
        L.append(f"- 高相似块对(cos≥0.9)：**{dup_pairs} 对 / {total_pairs:,} = "
                 f"{dup_pairs/max(total_pairs,1):.2%}**（重复块风险）")
    L.append("")
    L.append("### A3 边界完整性")
    L.append(f"- 块尾**未以句末标点结束**（疑似截断）：{end_mid:,} / {n_ch:,} = "
             f"**{end_mid/max(n_ch,1):.1%}**")
    L.append(f"- 块首**小写开头**（疑似承接上文）：{start_low:,} / {n_ch:,} = "
             f"**{start_low/max(n_ch,1):.1%}**")
    L.append("")
    L.append("### A4 块内聚度（块内段落间平均余弦，抽样）")
    if cohesion:
        L.append(f"- 样本 {len(cohesion)} 块：**均值 {np.mean(cohesion):.3f}** | "
                 f"中位 {np.median(cohesion):.3f} | p10 {pct(cohesion,10):.3f} | "
                 f"p90 {pct(cohesion,90):.3f}")
        L.append("  （越高=块内语义越单一；显著偏低说明块是「多主题拼盘」）")
    else:
        L.append("- （未抽样；--cohesion-sample 0 关闭）")
    L.append("")

    L.append("## B. 检索可分性（query → chunk 余弦）")
    L.append("")
    L.append("### B1 分布")
    L.append(f"- query→**gold 块**（每题取最高）：mean {np.mean(gold_best):.3f} | "
             f"median {np.median(gold_best):.3f} | p10 {pct(gold_best,10):.3f} | "
             f"p90 {pct(gold_best,90):.3f}")
    L.append(f"- query→**其余块**（全部干扰项）：mean {np.mean(others):.3f} | "
             f"median {np.median(others):.3f} | p90 {pct(others,90):.3f} | "
             f"p99 {pct(others,99):.3f}")
    L.append("")
    L.extend(hist_lines("gold 相似度", gold_best))
    L.extend(hist_lines("干扰项相似度", others))
    L.append("")
    L.append("### B2 margin（分开度）")
    L.append(f"- margin = s(gold_best) − s(干扰项最高)：mean **{np.mean(margins):+.3f}** | "
             f"median {np.median(margins):+.3f} | "
             f"p10 {pct(margins,10):+.3f} | p90 {pct(margins,90):+.3f}")
    L.append(f"- margin>0（gold 就是最高分）比例：**{sum(1 for m in margins if m > 0)/max(len(margins),1):.1%}**")
    L.append(f"- 干扰项超越 gold_best 的比例（均）：{np.mean(n_other_above):.1%}"
             f"（0=完美可分）")
    if gold_nn:
        L.append(f"- gold 块 ↔ 最相似非 gold 块余弦：mean {np.mean(gold_nn):.3f} | "
                 f"中位 {np.median(gold_nn):.3f}（越高=存在近似重复块，gold 不唯一）")
    L.append("")
    L.append("### B3 排名（命中率）")
    L.append("| 检索器 | gold@1 | R@8 | R@12 | R@16 | MRR@16 |")
    L.append("|---|---|---|---|---|---|")
    for nm, rr in (("向量 cosine", ranks_vec), ("混合 RRF", ranks_hyb)):
        d = max(len(rr), 1)
        L.append(f"| {nm} | {sum(1 for x in rr if x==0)/d:.3f} | "
                 f"{sum(1 for x in rr if x<8)/d:.3f} | {sum(1 for x in rr if x<12)/d:.3f} | "
                 f"{sum(1 for x in rr if x<16)/d:.3f} | "
                 f"{sum(1.0/(x+1) for x in rr if x<16)/d:.3f} |")
    L.append("")
    L.append("### B4 可分性（threshold-free）")
    L.append(f"- **AUC（每题各算再平均）= {b_auc:.3f}**｜中位 {b_auc_med:.3f}｜"
             f"p10 {b_auc_p10:.3f}（0.5=随机，1.0=完美可分）")
    L.append(f"  （口径说明：不同题的 cosine 尺度不同，**不能**把所有题拼成一个大数组算池化 AUC）")
    L.append(f"- Cohen's d = {cohens_d(np.asarray(gold_best), np.asarray(others)):.2f}"
             f"（>0.8 大效应）")
    L.append("")
    L.append("### B5 分组")
    L.append("| 组 | n | gold@1 | MRR@16 | AUC | gold均 | 干扰均 | margin均 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for gname, g in grp_stat.items():
        if not g["n"]:
            continue
        gs, _gm, _gp = mean_auc(g["auc_s"], g["auc_l"])
        gb = np.asarray(g["gb"]); ot = np.asarray(g["ot"])
        L.append(f"| {gname} | {g['n']} | {g['top1']/g['n']:.3f} | "
                 f"{g['mrr']/g['n']:.3f} | {gs:.3f} | "
                 f"{gb.mean():.3f} | {ot.mean():.3f} | {np.mean(gb)-ot.mean():+.3f} |")
    L.append("")

    L.append("## C. 切分策略对照（同 gold 口径，向量检索）")
    L.append("")
    if strat_rows:
        nsub = args.strat_papers or len(by_pid)
        L.append("**主表（固定 k=12，注意：偏向大块）**")
        L.append("")
        L.append("| 策略 | 块/篇 | 映射 | gold@1 | R@12 | MRR@16 | AUC | margin>0 | 跨块率 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in strat_rows:
            L.append(f"| {r['name']} | {r['n_chunks']/max(nsub,1):.0f} | {r['denom']} | "
                     f"{r['hit@1']:.3f} | {r['hit@12']:.3f} | "
                     f"{r['mrr@16']:.3f} | {r['auc']:.3f} | {r['margin_pos']:.3f} | "
                     f"{r['cross']:.3f} |")
        L.append("")
        L.append("**等预算表（rank 序累计字符达 B 前命中 gold；消除「大块更占便宜」的口径偏差）**")
        L.append("")
        L.append("| 策略 | R@8k字符 | R@16k字符 | R@24k字符 |")
        L.append("|---|---|---|---|")
        for r in strat_rows:
            hb = r["hitb"]
            L.append(f"| {r['name']} | {hb[8000]:.3f} | {hb[16000]:.3f} | {hb[24000]:.3f} |")
        L.append("")
        L.append(f"> 对照论文 {nsub} 篇；固定窗口按字符硬切（最朴素基线），段落打包=段落原子累加。")
        L.append("> 固定 k 下大块因「每 slot 装更多字/覆盖更大比例正文」而虚高，**判断切分优劣看等预算表**。")
    else:
        L.append("- （关闭）")
    L.append("")

    L.append("## D. 跨块诊断")
    L.append(f"- gold 证据**跨多块**的比例：{cross_cnt}/{b_denom} = "
             f"**{cross_cnt/da:.1%}**（越低=切分越少把答案切断）")
    L.append("")

    txt = "\n".join(L)
    (ROOT / args.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
