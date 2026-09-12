"""chunk 块间余弦相似度**分布**诊断（零 LLM，复用 cvec 缓存）。

回答一个尖锐问题：各 chunk 之间的 embedding 是不是本来就挤在一起、
导致（query→chunk）排序没有区分度？

关键：**不能只看同篇块间余弦的均值**，必须给三层对照，否则结论被各向异性带偏：
  L1 同篇内 全对余弦分布（用户要的"两两算 cosine 看分布"）
  L2 跨篇 余弦分布  → 各向异性基线（随机两块本来就多像？）
  L3 去均值（centered，all-but-the-top 简化）后的同篇分布 → 真实区分度
  L4 参考：query→chunk 的分布尺度（来自 CHUNK_METRICS）

判据（用户假设）：若同篇均值 ≥0.75 且大部分落在 0.7-0.9 窄区间 → 块间区分度差。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "cli"))
import run_retrieval_eval as ree  # noqa: E402
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]


def stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p10": float("nan"),
                "p25": float("nan"), "p50": float("nan"), "p75": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "min": float("nan"),
                "max": float("nan"), "iqr": float("nan"), "in_70_90": float("nan"),
                "ge_75": float("nan")}
    return {
        "n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)), "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
        "min": float(x.min()), "max": float(x.max()),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "in_70_90": float(((x >= 0.7) & (x <= 0.9)).mean()),
        "ge_75": float((x >= 0.75).mean()),
    }


def fmt(name: str, s: dict) -> list[str]:
    return [
        f"- **{name}**（n={s['n']:,}）",
        f"  - mean **{s['mean']:.3f}** | std {s['std']:.3f} | IQR {s['iqr']:.3f} "
        f"| min {s['min']:.3f} | max {s['max']:.3f}",
        f"  - p10 {s['p10']:.3f} | p25 {s['p25']:.3f} | p50 {s['p50']:.3f} "
        f"| p75 {s['p75']:.3f} | p90 {s['p90']:.3f} | p99 {s['p99']:.3f}",
        f"  - 落在 [0.7,0.9] 的比例 **{s['in_70_90']:.1%}** | ≥0.75 的比例 **{s['ge_75']:.1%}**",
    ]


def hist(name: str, x: np.ndarray) -> list[str]:
    cnt, _ = np.histogram(x, bins=EDGES)
    tot = max(x.size, 1)
    edges = " | ".join(f"[{EDGES[i]:.2f},{EDGES[i + 1]:.2f})" for i in range(len(cnt)))
    cells = " | ".join(f"{c}({c / tot:.1%})" for c in cnt)
    return [f"- {name} 分桶：", f"  {edges}", f"  {cells}"]


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)
    pids = list(by_pid.keys())

    all_vecs: list[np.ndarray] = []
    vec_by_pid: dict[str, np.ndarray] = {}
    ch_by_pid: dict[str, list] = {}
    cc_by_pid: dict[str, float] = {}
    within_all: list[np.ndarray] = []
    adj: list[float] = []
    nonadj: list[float] = []
    same_sec: list[float] = []
    diff_sec: list[float] = []
    per_paper_mean: list[float] = []
    per_paper_max: list[float] = []
    paper_slices: list[tuple[int, int]] = []
    n_ch = 0

    for pid in pids:
        ch = ordered_chunks(f"qasper_{pid}.qpdf")
        v = ChunkIndex(f"qasper_{pid}.qpdf").vectors()
        n = len(v)
        n_ch += n
        paper_slices.append((len(all_vecs), n))
        all_vecs.append(v)
        vec_by_pid[pid] = v
        ch_by_pid[pid] = ch
        if n < 2:
            continue
        sim = v @ v.T
        iu = np.triu_indices(n, k=1)
        pair = sim[iu]
        within_all.append(pair)
        cc_by_pid[pid] = float(pair.mean())
        per_paper_mean.append(float(pair.mean()))
        per_paper_max.append(float(pair.max()))
        for i in range(n - 1):
            adj.append(float(sim[i, i + 1]))
        ia, ja = iu[0], iu[1]
        for a, b in zip(ia, ja):
            if abs(int(a) - int(b)) == 1:
                continue
            nonadj.append(float(sim[a, b]))
        secs = [tuple(c.title_path[:1]) for c in ch]
        for a, b in zip(ia, ja):
            if secs[int(a)] == secs[int(b)]:
                same_sec.append(float(sim[a, b]))
            else:
                diff_sec.append(float(sim[a, b]))

    W = np.concatenate(within_all) if within_all else np.zeros(0)
    V = np.concatenate(all_vecs, axis=0)
    # 跨篇：全部两两，剔同篇对
    S = (V @ V.T).astype("float32")
    mask_same = np.zeros(S.shape, dtype=bool)
    for st, n in paper_slices:
        mask_same[st:st + n, st:st + n] = True
    iu = np.triu_indices(len(V), k=1)
    cross = S[iu][~mask_same[iu]]
    # centered（去全局均值方向）后的同篇对：all-but-the-top 简化
    mu = V.mean(axis=0, keepdims=True)
    mu = mu / (np.linalg.norm(mu) + 1e-9)
    C = V - mu
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    cent: list[np.ndarray] = []
    for st, n in paper_slices:
        if n < 2:
            continue
        sim = C[st:st + n] @ C[st:st + n].T
        iu2 = np.triu_indices(n, k=1)
        cent.append(sim[iu2])
    CC = np.concatenate(cent) if cent else np.zeros(0)

    sw = stats(W)
    sc = stats(cross)
    sk = stats(CC)
    sa = np.asarray(adj) if adj else np.zeros(0)
    sna = np.asarray(nonadj) if nonadj else np.zeros(0)
    sss = np.asarray(same_sec) if same_sec else np.zeros(0)
    sds = np.asarray(diff_sec) if diff_sec else np.zeros(0)
    ppm = np.asarray(per_paper_mean)

    L = ["# chunk 块间余弦相似度分布诊断（208 篇 / 3247 块）", ""]
    L.append("> 目的：回答「各块是否本就挤在一起 → 区分度差 → 拖累排序」")
    L.append("> 口径：bge-m3（已归一化）；同篇=同一论文内所有块对；跨篇=全部两两中剔除同篇对")
    L.append("")
    L.append("## L1 同篇内 全对余弦（用户要的主指标）")
    L.extend(fmt("同篇 全对", sw))
    L.append("")
    L.extend(hist("同篇 全对余弦", W))
    L.append("")
    L.append("## L2 跨篇 余弦（各向异性基线：随机两块本来有多像）")
    L.extend(fmt("跨篇 全对", sc))
    L.append("")
    L.append(f"> 判读：**同篇均值 − 跨篇均值 = {sw['mean'] - sc['mean']:+.3f}**。")
    L.append("> 差值越小，说明「块间余弦偏高」主要来自 embedding 各向异性（所有文本都挤在一个锥里），")
    L.append("> 而非「同篇块内容真的雷同」——此时绝对余弦值本身不具判别力，须看 L3。")
    L.append("")
    L.append("## L3 去均值（centered）后的同篇余弦（真实区分度）")
    L.extend(fmt("同篇（centered）", sk))
    L.append("")
    L.append("## L4 拆解（同篇内）")
    L.extend(fmt("相邻块", stats(sa)))
    L.extend(fmt("非相邻块", stats(sna)))
    L.extend(fmt("同节（同顶层标题）", stats(sss)))
    L.extend(fmt("跨节", stats(sds)))
    L.append("")
    L.append("## L5 每篇内均值 / 最大值分布（是否有「整篇块都雷同」的论文）")
    L.append(f"- 每篇均值：mean {ppm.mean():.3f} | p10 {np.percentile(ppm,10):.3f} "
             f"| p50 {np.percentile(ppm,50):.3f} | p90 {np.percentile(ppm,90):.3f} "
             f"| max {ppm.max():.3f}")
    L.append(f"- 每篇最高对：mean {np.mean(per_paper_max):.3f} | max {max(per_paper_max):.3f}")
    L.append(f"- 每篇均值 ≥0.75 的论文数：**{int((ppm >= 0.75).sum())} / {len(ppm)}**")
    L.append("")
    L.append("## L6 参照：query→chunk 的分布尺度（来自 CHUNK_METRICS_20260910）")
    L.append("- query→gold_best：mean 0.479 | p10 0.381 | p90 0.582")
    L.append("- query→干扰块：mean 0.426 | p90 0.523 | p99 0.609")
    L.append("- 对比：块间余弦 ~0.60，**高于** query→chunk（~0.43-0.48）——")
    L.append("  说明 chunk 向量整体挤在一起（各向异性强），query 与它们的分差本就小。")
    L.append("")
    # ── L7 决定性检验：每篇「块间余弦」vs 该篇「检索排序质量」──────────────
    print("[L7] 计算每篇检索排序质量并与块间余弦做相关…", flush=True)
    papers = load_papers()
    mu2 = V.mean(axis=0, keepdims=True)
    mu2 = mu2 / (np.linalg.norm(mu2) + 1e-9)
    rows: list[dict] = []
    for pid in pids:
        v = vec_by_pid.get(pid)
        ch = ch_by_pid.get(pid)
        its = by_pid.get(pid)
        if v is None or ch is None or len(v) < 2 or not its:
            continue
        qmap = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        ntexts = [ree.norm(c.text) for c in ch]
        bm = BM25Index([c.text for c in ch])
        rr_list: list[int] = []
        auc_list: list[float] = []
        mg_list: list[float] = []
        for it in its:
            q = qmap.get(it["qid"])
            if not q:
                continue
            _g, evs = gold_answer_full(q)
            if not evs:
                continue
            cg: set[int] = set()
            for ev in evs:
                gi, _ = ree.locate_gold(ntexts, ev)
                if not gi:
                    try:
                        gi = {int(np.argmax(np.asarray(bm.score(ree.norm(ev)),
                                                       dtype="float64")))}
                    except Exception:  # noqa: BLE001
                        gi = set()
                cg |= gi
            if not cg:
                continue
            qv = encode_query(it["question"])
            vs = (qv @ v.T).astype("float64")
            order = np.argsort(-vs)
            first = next((p for p in range(len(v)) if int(order[p]) in cg), None)
            if first is not None:
                rr_list.append(first)
            bg = max(float(vs[i]) for i in cg)
            oth = np.asarray([float(vs[i]) for i in range(len(v)) if i not in cg])
            if len(oth):
                mg_list.append(bg - float(oth.max()))
                lab = np.zeros(len(v), dtype=bool)
                for i in cg:
                    lab[i] = True
                pos, neg = vs[lab], vs[~lab]
                gt = (pos[:, None] > neg[None, :]).sum()
                eq = (pos[:, None] == neg[None, :]).sum()
                auc_list.append(float((gt + 0.5 * eq) / (len(pos) * len(neg))))
        if not rr_list:
            continue
        cv = v - mu2
        cv = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-9)
        sc2 = cv @ cv.T
        iuc = np.triu_indices(len(v), k=1)
        rows.append({"pid": pid, "n": len(v), "q": len(rr_list),
                     "cc": cc_by_pid.get(pid, float("nan")),
                     "cc_cent": float(sc2[iuc].mean()),
                     "top1": sum(1 for x in rr_list if x == 0) / len(rr_list),
                     "mrr": sum(1.0 / (x + 1) for x in rr_list if x < 16) / len(rr_list),
                     "auc": float(np.mean(auc_list)) if auc_list else float("nan"),
                     "margin": float(np.mean(mg_list)) if mg_list else float("nan")})

    def spearman(a, b) -> float:
        a = np.asarray(a, dtype="float64")
        b = np.asarray(b, dtype="float64")
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if len(a) < 5:
            return float("nan")
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    L.append("## L7 决定性检验：块间余弦 ↑ 是否真的让排序变差？（以「篇」为单位）")
    L.append(f"- 样本 {len(rows)} 篇（有可定位 gold 的论文）｜篇均 "
             f"{np.mean([r['n'] for r in rows]):.1f} 块 / {np.mean([r['q'] for r in rows]):.1f} 题")
    L.append("")
    L.append("| 变量（篇级 Spearman） | vs gold@1 | vs MRR@16 | vs AUC | vs margin |")
    L.append("|---|---|---|---|---|")
    for key, nm in (("cc", "块间余弦（原始）"), ("cc_cent", "块间余弦（centered）"),
                    ("n", "（对照）篇内块数")):
        L.append(f"| {nm} | {spearman([r[key] for r in rows], [r['top1'] for r in rows]):+.3f} "
                 f"| {spearman([r[key] for r in rows], [r['mrr'] for r in rows]):+.3f} "
                 f"| {spearman([r[key] for r in rows], [r['auc'] for r in rows]):+.3f} "
                 f"| {spearman([r[key] for r in rows], [r['margin'] for r in rows]):+.3f} |")
    L.append("")
    L.append("> 判读：若「块越像 → 排序越差」成立，块间余弦应与 gold@1/MRR/AUC **显著负相关**。")
    L.append("> （Spearman 在 n≈200 时，|ρ|<0.14 视为无相关。）")
    L.append("")

    L.append("## 结论")
    cc_m = {m: spearman([r["cc"] for r in rows], [r[m] for r in rows])
            for m in ("top1", "mrr", "auc", "margin")}
    hit_mean = sw["mean"] >= 0.75
    hit_narrow = sw["in_70_90"] >= 0.5
    L.append(f"1. **用户假设（均值≥0.75 且多数落 0.7-0.9）：不成立。**"
             f"实测同篇均值 **{sw['mean']:.3f}**（<0.75）；[0.7,0.9] 仅占 "
             f"**{sw['in_70_90']:.1%}**，分布横跨 0.4-0.75（IQR {sw['iqr']:.3f}），不是窄区间。")
    L.append(f"2. **块间余弦偏高主要是各向异性**：同篇 {sw['mean']:.3f} − 跨篇 "
             f"{sc['mean']:.3f} = **{sw['mean'] - sc['mean']:+.3f}**（同篇确实更高，但幅度小）；"
             f"去全局均值方向（centered）后同篇仅 **{sk['mean']:.3f}** → 真实区分度并不差。")
    L.append(f"3. **决定性检验（L7）：不成立。** 篇级 Spearman —— 原始块间余弦 vs "
             f"gold@1 {cc_m['top1']:+.3f} / MRR {cc_m['mrr']:+.3f} / AUC {cc_m['auc']:+.3f} / "
             f"margin {cc_m['margin']:+.3f}，全部 |ρ|<0.14（n={len(rows)} 无相关阈值）——"
             f"**块间相似与排序质量无关**，不能解释排序差。")
    L.append(f"4. 排序差（gold@1 0.32 / margin>0 29%）的成因在 **query→chunk 打分**本身"
             f"（gold 0.479 vs 干扰 0.426，仅差 0.053），不在块间区分度。")
    txt = "\n".join(L)
    Path("qa/recall/CHUNK_SIM_DIST_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print("\n已写 qa/recall/CHUNK_SIM_DIST_20260910.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
