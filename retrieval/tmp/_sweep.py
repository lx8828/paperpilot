"""打分线（A）+ 融合线（B）扫描 —— 沙盒内，**零 LLM / 零模型**。

方法学（见对话记录）：
  · **先 OFAT 再二维**：A 只动 BM25 `k1/b`；B1 只动 RRF `k` 与 `w_bm`；
    B2 只动 `K_zh/K_en` 与并集策略；B3 才做交互。
  · **交叉验证（免费、严格）**：group1 与 group2 的题集/语料**不同** →
    两组的指标分别报，**只有两边都不差的配置才进候选**（挡"在 120 题上偶然好看"）。
  · **守卫指标**：候选占语料比（防"塞满→召回虚高"）+ gold 密度 lift（别掉到 ~1×）。

指标口径：gold 块进前 12 的比例（分母 = 该题**全部** gold 块，含没进候选的）；
M1 另报**篇级全篇覆盖**（块级对 M1 过严，见 `_sandbox.py --chunk`）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))
sys.path.insert(0, str(ROOT / "src"))

import _sandbox as sbx  # noqa: E402
from paperpilot.agents.embedder import _bm_or_none, rrf_order  # noqa: E402

GROUPS = ("group1", "group2")
SBS = {g: sbx.Sandbox(g) for g in GROUPS}
GOLD = {(g, q["qid"]): sbx._gold_of(SBS[g], q["qid"]) for g in GROUPS for q in SBS[g].queries}
_ORD: dict = {}


def orders_of(g: str, qid: str, qlang: str, *, k: float, w: float,
              k1: float, b: float, use_bm: bool = True):
    """(全局全量序, 每篇全量序)。RRF 只看名次 → 换 K 不必重算，故按 (k,w,k1,b) 缓存。"""
    key = (g, qid, qlang, k, w, k1, b, use_bm)
    hit = _ORD.get(key)
    if hit is None:
        sb = SBS[g]
        v = sb.vec_scores(qid, qlang)
        n = len(v)
        bsig = _bm_or_none(sb.bm_global(qid, qlang, k1=k1, b=b)) if use_bm else None
        glob = rrf_order(v, bsig, k=k, w_vec=np.ones(n), w_bm=np.full(n, w))
        per: dict[str, list[int]] = {}
        for p in sb.pdfs:
            s = sb.seg[p]
            bs = _bm_or_none(sb.bm_scores(qid, qlang, seg=p, k1=k1, b=b)) if use_bm else None
            o = rrf_order(v[s], bs, k=k, w_vec=np.ones(len(s)), w_bm=np.full(len(s), w))
            per[p] = [int(s[int(i)]) for i in o]
        hit = (glob, per)
        _ORD[key] = hit
    return hit


def arm_sel(g: str, qid: str, qlang: str, K: int, **kw) -> list[int]:
    sb = SBS[g]
    glob, per = orders_of(g, qid, qlang, **kw)
    return sb.quota(glob, per, top_k=K, floor=1)


def merge_append(zh: list[int], en: list[int]) -> list[int]:
    seen = set(zh)
    return zh + [i for i in en if i not in seen]


def merge_inter(zh: list[int], en: list[int], *, w_en: float, k: float, top_n: int
                ) -> list[int]:
    """加权交错：把英文路当"第三路 RRF"（路内名次加权），再取前 top_n（**上下文不涨**）。"""
    sc: dict[int, float] = {}
    for lst, wgt in ((zh, 1.0), (en, w_en)):
        for r, i in enumerate(lst):
            sc[i] = sc.get(i, 0.0) + wgt / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])[:top_n]


def evaluate(sel_of) -> dict:
    """sel_of(g, qid) -> 候选块下标列表。返回汇总指标（分组建模 + 合计）。"""
    acc = {g: {"gold": 0, "in12": 0, "ceil12": 0, "m1g": 0, "m1_12": 0,
               "m1_gold_all": 0, "m1_pap_tot": 0,
               "m1_pap_full": 0, "ncand": 0, "nq": 0, "chars": 0,
               "gold_in_cand": 0, "corpus": 0} for g in GROUPS}
    for g in GROUPS:
        sb = SBS[g]
        for q in sb.queries:
            qid = q["qid"]
            gp = GOLD[(g, qid)]
            gold = gp["qchunks"]
            gset = set(gold)
            sel = sel_of(g, qid)
            pos = {int(i): r for r, i in enumerate(sel)}
            a = acc[g]
            a["nq"] += 1
            a["gold"] += len(gold)
            a["in12"] += sum(1 for i in gold if i in pos and pos[i] < 12)
            # 候选池的排序上界：把池内 gold 全排到最前，能有多少进前 12
            # （当前@12 与它的差 = 留给**重排**的空间；池里根本没有的看不到）
            a["ceil12"] += min(len(gset & set(pos)), 12)
            a["ncand"] += len(sel)
            a["chars"] += int(sum(int(sb.tlens[i]) for i in sel))
            a["gold_in_cand"] += len(gset & set(pos))
            a["corpus"] += sb.N
            if qid.startswith(("G1-M1", "G2-M1")):
                papers = {sb.chunk_meta[i]["pdf"] for i in gold}
                top12 = {sb.chunk_meta[i]["pdf"] for i in sel[:12]}
                a["m1g"] += 1
                a["m1_gold_all"] = a.get("m1_gold_all", 0) + len(gold)
                a["m1_12"] += sum(1 for i in gold if i in pos and pos[i] < 12)
                a["m1_pap_tot"] += len(papers)
                a["m1_pap_full"] += 1 if papers <= top12 else 0
    return acc


def row(name: str, acc: dict) -> dict:
    tot = {kk: sum(acc[g][kk] for g in GROUPS) for kk in acc[GROUPS[0]]}
    per = {g: acc[g]["in12"] / max(acc[g]["gold"], 1) for g in GROUPS}
    return {
        "name": name,
        "gold12": tot["in12"] / max(tot["gold"], 1),
        "m1_12": tot["m1_12"] / max(tot.get("m1_gold_all", 0), 1),
        "ceil12": tot["ceil12"] / max(tot["gold"], 1),
        "m1_full": tot["m1_pap_full"] / max(tot["m1g"], 1) if tot["m1g"] else 0.0,
        "ncand": tot["ncand"] / max(tot["nq"], 1),
        "ratio": tot["ncand"] / max(tot["corpus"], 1),
        "lift": (tot["gold_in_cand"] / max(tot["ncand"], 1))
                / (tot["gold"] / max(tot["corpus"], 1)),
        "chars": tot["chars"] / max(tot["nq"], 1),
        "g1": per["group1"], "g2": per["group2"],
    }


HDR = (f"  {'配置':<32}{'gold@12':>8}{'池上界':>8}{'M1块@12':>9}{'M1篇全':>8}"
       f"{'候选数':>7}{'占比':>7}{'lift':>7}{'字符':>8}{'g1':>7}{'g2':>7}")


def show(rows: list[dict], title: str, top: int = 0) -> None:
    print(f"\n########## {title}")
    print(HDR)
    for r in (rows[:top] if top else rows):
        ok = r["g1"] >= BASE["g1"] - 0.005 and r["g2"] >= BASE["g2"] - 0.005
        flag = "" if ok else " ⚠️CV"
        print(f"  {r['name']:<30}{r['gold12']:>8.1%}{r['ceil12']:>8.1%}{r['m1_12']:>9.1%}"
              f"{r['m1_full']:>8.0%}{r['ncand']:>7.1f}{r['ratio']:>7.1%}"
              f"{r['lift']:>7.1f}×{r['chars']:>8.0f}{r['g1']:>7.1%}{r['g2']:>7.1%}{flag}")


def main() -> int:
    global BASE
    t0 = time.time()
    K0, XK0 = 24, 24
    base_of = lambda g, q: merge_append(arm_sel(g, q, "zh", K0, k=60, w=1.0, k1=1.5, b=0.75),
                                        arm_sel(g, q, "en", XK0, k=60, w=1.0, k1=1.5, b=0.75))
    BASE = row("BASE（现状：k1=1.5 b=0.75 k=60 w=1 K=24+24）", evaluate(base_of))
    show([BASE], "基线（用于对照）")

    # ── A · 打分线：BM25 k1 × b ──
    rows = []
    for k1 in (0.8, 1.2, 1.5, 2.0, 2.5):
        for b in (0.3, 0.5, 0.75, 1.0):
            f = lambda g, q, k1=k1, b=b: merge_append(
                arm_sel(g, q, "zh", K0, k=60, w=1.0, k1=k1, b=b),
                arm_sel(g, q, "en", XK0, k=60, w=1.0, k1=k1, b=b))
            rows.append(row(f"k1={k1} b={b}", evaluate(f)))
    rows.sort(key=lambda r: -r["gold12"])
    show(rows, "A · 打分线：BM25 k1 × b（20 组；其余固定为现状）", top=8)

    # ── B1 · 融合线：RRF k × BM25 权重 ──
    rows = []
    for k in (10, 20, 30, 60, 120, 200):
        for w in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            f = lambda g, q, k=k, w=w: merge_append(
                arm_sel(g, q, "zh", K0, k=k, w=w, k1=1.5, b=0.75),
                arm_sel(g, q, "en", XK0, k=k, w=w, k1=1.5, b=0.75))
            rows.append(row(f"RRF k={k} w_bm={w}", evaluate(f)))
    rows.sort(key=lambda r: -r["gold12"])
    show(rows, "B1 · 融合线：RRF k × BM25 权重 w（36 组；K 固定 24+24）", top=10)

    # ── B2 · 候选集：K_zh × K_en × 并集策略 ──
    rows = []
    for Kz in (12, 16, 20, 24, 32, 48):
        for Ke in (0, 8, 16, 24):
            for pol in ("append", "inter0.5", "inter1.0"):
                if Ke == 0 and pol != "append":
                    continue
                def f(g, q, Kz=Kz, Ke=Ke, pol=pol):
                    zh = arm_sel(g, q, "zh", Kz, k=60, w=1.0, k1=1.5, b=0.75)
                    if Ke == 0:
                        return zh
                    en = arm_sel(g, q, "en", Ke, k=60, w=1.0, k1=1.5, b=0.75)
                    if pol == "append":
                        return merge_append(zh, en)
                    return merge_inter(zh, en, w_en=float(pol[5:]), k=60, top_n=Kz)
                rows.append(row(f"Kz={Kz} Ke={Ke} {pol}", evaluate(f)))
    rows.sort(key=lambda r: -r["gold12"])
    show(rows, "B2 · 候选集：K_zh × K_en × 并集策略（30 组）", top=14)

    # ── B3 · 交互：RRF k × (Kz, Ke) × inter ──
    rows = []
    for k in (10, 20, 60):
        for Kz, Ke in ((12, 16), (16, 16), (24, 16), (32, 16), (24, 24), (32, 24)):
            f = lambda g, q, Kz=Kz, Ke=Ke, k=k: merge_inter(
                arm_sel(g, q, "zh", Kz, k=k, w=1.0, k1=1.5, b=0.75),
                arm_sel(g, q, "en", Ke, k=k, w=1.0, k1=1.5, b=0.75),
                w_en=1.0, k=k, top_n=Kz)
            rows.append(row(f"k={k} Kz={Kz} Ke={Ke} inter1.0", evaluate(f)))
    rows.sort(key=lambda r: (-r["gold12"], -r["ceil12"]))
    show(rows, "B3 · 交互：RRF k × (K_zh, K_en) × 加权交错（18 组）", top=12)

    print(f"\n[耗时] {time.time() - t0:.1f}s ｜ 序缓存 {len(_ORD)} 条")
    print("注：'⚠️CV' = 该配置在 group1 或 group2 上低于基线 → 交叉验证不通过，别采纳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
