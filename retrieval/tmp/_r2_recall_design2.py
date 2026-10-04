"""**按「通道类型」分别标定截断深度 `d`**（50 篇口径，零 LLM）

## 为什么要拆
上一版把所有列表共用一个 `d`（最优 d=50 块）。但三类列表的**质量差异很大**：

| 类 | 列表 | 单独 StRecall@13（`R2_SUBQ_CONTRIB`） |
|---|---|---|
| `zh-dense` | 中文题面 ×1 | 0.448 |
| `sq-dense` | 3 条英文子查询 ×3 | 0.443 ~ 0.475 |
| `sq-bm25` | 3 条英文子查询 ×3 | **0.473 ~ 0.529** |

→ 共用一个 `d` 是**把不同的东西强行对齐**。本脚本让三类各有 `d_zh / d_sd / d_sb`。

## 三个阶段
| 阶段 | 做什么 | 输出 |
|---|---|---|
| **① 逐类 `d` 曲线**（诊断） | 每类的「各列表 top-d 块的并集」覆盖多少 gold？边际增益在哪拐？ | 三类曲线 → 定候选区间 |
| **② 三维 `d` × `ρ` 网格** | 在两个区间候选上做网格（3×3×3 × ρ） | 最优 `(d_zh, d_sd, d_sb, ρ)` |
| **③ 复核段级聚合** | 最优配置下再扫 `n_keep` × `agg` | 确认篇内 max |

**查询 = `subqueries_v2.json`**（已修：`prompt_eng` 补 3 条、`knowledge_distill` 净化、
`case_study` 去冗余）；**去 `para`**（dense≈zh、bm25 最弱）。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_recall_design2.py --corpus 50
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"
RRF_C = 60
D_CURVE = [5, 10, 20, 30, 50, 75, 100, 150, 300, 0]      # 0 = 不截（全库）
GRID_VALS = [20, 50, 100]                                 # 三维网格候选
RHO_GRID: list = [0.0, 0.5, 1.0, 2.0, 4.0, None]          # 0 = 纯 dense；None = 纯 bm25
RHO_LBL = {0.0: "dense", 0.5: "2:1", 1.0: "1:1", 2.0: "1:2", 4.0: "1:4", None: "bm25"}
NKEEP = [1, 3, 6]
AGGS = ["max", "sum", "mean"]
RNG = np.random.default_rng(20260930)


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def boot(a: pd.DataFrame, b: pd.DataFrame, col: str = "setF1"):
    keys = ["cluster", "facet", "k"]
    j = (a.set_index(keys)[col].to_frame("a").join(
        b.set_index(keys)[col].to_frame("b"), how="inner"))
    if not len(j):
        return None
    d = (j["a"] - j["b"]).to_numpy()
    m = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float((m <= 0).mean()), len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--subqueries", default="subqueries_v2.json")
    args = ap.parse_args()
    if args.corpus == "50":
        CDIR, KDIR = DEV / "corpus50", DEV / "prodchunk50"
        GF, pmap_f = DEV / "gold_final3.csv", DEV / "corpus50" / "pdf_map_all.json"
    else:
        CDIR, KDIR = DEV / "clusters", DEV / "prodchunk"
        GF, pmap_f = DEV / "gold_final2.csv", DEV / "pdf_map.json"
    pm_ok = {d for d, v in json.loads(pmap_f.read_text(encoding="utf-8")).items()
             if v.get("ok")}
    gg = pd.read_csv(GF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    subq = json.loads((DEV / args.subqueries).read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    print("=" * 122)
    print(f"【按通道类型分别标定 d】语料 {args.corpus} ｜ 真值 {GF.name}"
          f" ｜ 查询 {args.subqueries}（已修 3 个 facet）｜ RRF_C={RRF_C}")
    print(f"  三类：`zh-dense`(1 列表) ｜ `sq-dense`(n) ｜ `sq-bm25`(n) ｜ **去 `para`**")
    print(f"  曲线 d∈{D_CURVE}（0=不截）｜ 网格 d∈{GRID_VALS}³ × ρ∈{[RHO_LBL[r] for r in RHO_GRID]}")

    curve, grid, aggd = [], [], []
    t0 = time.time()
    for ci in range(3):
        ch = pd.read_parquet(KDIR / args.chunktag / f"c{ci}.parquet")
        ls = pd.read_parquet(CDIR / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist()
                if d in set(ch["docid"]) and d in pm_ok]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        didx = {d: np.where(owner == d)[0] for d in docs}
        E = enc.encode(chunks, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)
        nb = len(chunks)
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {nb} 块 ｜ {time.time() - t0:.0f}s", flush=True)

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            N = len(docs)
            zh, para = str(F2[facet][1]), str(F2[facet][3])
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3] or [para]
            # 三类：每类 = 一组 (kind, rank 数组)
            CLASSES: dict[str, list[np.ndarray]] = {"zh-dense": [], "sq-dense": [], "sq-bm25": []}
            for q in [zh]:
                qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                                convert_to_numpy=True).astype(np.float32)
                s = (qv @ E.T)[0]
                r = np.empty(nb, dtype=np.int64)
                r[np.argsort(-s)] = np.arange(nb)
                CLASSES["zh-dense"].append(r)
            for q in SQ:
                qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                                convert_to_numpy=True).astype(np.float32)
                s = (qv @ E.T)[0]
                r = np.empty(nb, dtype=np.int64)
                r[np.argsort(-s)] = np.arange(nb)
                CLASSES["sq-dense"].append(r)
            for q in SQ:
                s = bm.scores(q).astype(np.float32)
                r = np.empty(nb, dtype=np.int64)
                r[np.argsort(-s)] = np.arange(nb)
                CLASSES["sq-bm25"].append(r)

            # ── ① 逐类 d 曲线：该类各列表 top-d 块的并集能覆盖多少 gold ──
            for cls, rs in CLASSES.items():
                prev = 0.0
                for d in D_CURVE:
                    idxs: set[int] = set()
                    for r in rs:
                        idxs |= set((np.where(r < d)[0] if d else np.arange(nb)).tolist())
                    pool = {owner[i] for i in idxs}
                    cov = len(pool & gold) / len(gold)
                    curve.append(dict(cluster=ci + 1, facet=facet, cls=cls, d=d,
                                      n_chunks=len(idxs), n_docs=len(pool),
                                      cov=cov, marginal=cov - prev))
                    prev = cov

            # ── ② 三维 d × ρ 网格（篇内 max）──
            for dzh in GRID_VALS:
                for dsd in GRID_VALS:
                    for dsb in GRID_VALS:
                        dd = {"zh-dense": dzh, "sq-dense": dsd, "sq-bm25": dsb}
                        for rho in RHO_GRID:
                            sc = np.zeros(nb, dtype=np.float64)
                            for cls, rs in CLASSES.items():
                                # 权重：两类 dense 恒 1.0；bm25 取 ρ
                                # ρ=0 → 纯 dense（bm25 权重 0）；ρ=None → 纯 bm25（dense 权重 0）
                                if rho is None:
                                    w = 1.0 if cls == "sq-bm25" else 0.0
                                else:
                                    w = float(rho) if cls == "sq-bm25" else 1.0
                                if w <= 0:
                                    continue
                                d_ = dd[cls]
                                for r in rs:
                                    m = np.ones(nb, bool) if d_ == 0 else (r < d_)
                                    sc[m] += w / (RRF_C + r[m] + 1)
                            dm = {x: float(sc[didx[x]].max()) for x in docs}
                            o = sorted(docs, key=lambda x: -dm[x])
                            for k in (10, 13, 20):
                                p, r_, f1 = _m.setpf(o, gold, k)
                                grid.append(dict(cluster=ci + 1, facet=facet, k=k,
                                                 dzh=dzh, dsd=dsd, dsb=dsb,
                                                 rho=RHO_LBL[rho], n_gold=len(gold),
                                                 n_docs=N, StRecall=_m.strecall(o, gold, k),
                                                 setP=p, setF1=f1,
                                                 MRecall=_m.mrecall(o, gold, k), random=k / N))

            # ── ③ 最优配置附近复核段级聚合（用网格中位配置 d=50 三位 + ρ=1:4）──
            sc0 = np.zeros(nb, dtype=np.float64)
            for cls, rs in CLASSES.items():
                w = 4.0 if cls == "sq-bm25" else 1.0
                for r in rs:
                    m = r < 50
                    sc0[m] += w / (RRF_C + r[m] + 1)
            for nk in NKEEP:
                for agg in AGGS:
                    dm = {}
                    for x in docs:
                        v = np.sort(sc0[didx[x]])[::-1][:nk]
                        dm[x] = (float(v.max()) if agg == "max"
                                 else float(v.sum()) if agg == "sum" else float(v.mean()))
                    o = sorted(docs, key=lambda x: -dm[x])
                    p, r_, f1 = _m.setpf(o, gold, 13)
                    aggd.append(dict(cluster=ci + 1, facet=facet, nk=nk, aggm=agg,
                                     StRecall=_m.strecall(o, gold, 13), setP=p, setF1=f1))

    CV, GD, AG = pd.DataFrame(curve), pd.DataFrame(grid), pd.DataFrame(aggd)
    for df, nm in ((CV, "CURVE"), (GD, "GRID"), (AG, "AGG")):
        df.to_csv(HERE / "results" / f"R2_RD2_{nm}.csv", index=False, encoding="utf-8-sig")

    # ── 表① 逐类 d 曲线 ──
    print("\n" + "=" * 122)
    print("【表① 逐类 d 曲线】该类各列表 **top-d 块的并集** 覆盖多少 gold（k 无关，是召回上界）")
    print(f"  {'类':<10}{'d':>6}" + "".join(f"{x:>9}" for x in
                                          ("覆盖gold", "边际", "块数", "篇数")))
    for cls in ("zh-dense", "sq-dense", "sq-bm25"):
        t = CV[CV["cls"] == cls]
        for d in D_CURVE:
            s = t[t["d"] == d]
            if not len(s):
                continue
            lbl = "不截" if d == 0 else str(d)
            print(f"  {cls if d == D_CURVE[0] else '':<10}{lbl:>6}"
                  f"{s['cov'].mean():>9.3f}{s['marginal'].mean():>+9.3f}"
                  f"{s['n_chunks'].mean():>9.0f}{s['n_docs'].mean():>9.1f}")
        # 拐点：边际增益掉到**最大边际** 20% 以下的最小 d（即"再往深挖已不划算"）
        tt = t[t["d"] != 0]
        mg = tt.groupby("d")["marginal"].mean().sort_index()
        mxm = float(mg.max()) if len(mg) else 0.0
        knee = [int(d) for d in mg.index if mg[d] >= 0.2 * mxm]
        print(f"  {'':<10}→ 边际拐点 ≈ d={max(knee) if knee else '-'}"
              f"（边际首次跌破最大边际 20% 之前） ｜ 最大边际 {mxm:+.3f}")

    # ── 表② 三维网格 ──
    print("\n" + "=" * 122)
    print("【表② 三维 d × ρ 网格】k=13（按 setF1 降序，top 14）")
    g13 = GD[GD["k"] == 13]
    gg2 = (g13.groupby(["dzh", "dsd", "dsb", "rho"])
           .agg(sr=("StRecall", "mean"), sp=("setP", "mean"), f1=("setF1", "mean"),
                mc=("MRecall", "mean")).reset_index().sort_values("f1", ascending=False))
    print(f"  {'d_zh':>5}{'d_sq_dense':>12}{'d_sq_bm25':>11}{'ρ':>7}"
          f"{'StRecall':>10}{'setP':>8}{'setF1':>9}{'MRecall':>9}")
    for _, r in gg2.head(14).iterrows():
        print(f"  {int(r['dzh']):>5}{int(r['dsd']):>12}{int(r['dsb']):>11}{r['rho']:>7}"
              f"{r['sr']:>10.3f}{r['sp']:>8.3f}{r['f1']:>9.3f}{r['mc']:>9.3f}")
    bt = gg2.iloc[0]
    print(f"\n  ★ 最优：d_zh=**{int(bt['dzh'])}** ｜ d_sq_dense=**{int(bt['dsd'])}**"
          f" ｜ d_sq_bm25=**{int(bt['dsb'])}** ｜ ρ=**{bt['rho']}** → setF1 **{bt['f1']:.3f}**")

    # ── 表③ 与「所有类共用一个 d」对照 ──
    print("\n" + "=" * 122)
    print("【表③ 三 d 分离 vs 单一 d】同 ρ=1:4 下")
    print(f"  {'配置':<34}{'StRecall':>10}{'setF1':>9}")
    same = gg2[(gg2["rho"] == "1:4") & (gg2["dzh"] == gg2["dsd"])
               & (gg2["dsd"] == gg2["dsb"])].sort_values("f1", ascending=False)
    for _, r in same.iterrows():
        print(f"  {'三类同 d=' + str(int(r['dzh'])):<34}{r['sr']:>10.3f}{r['f1']:>9.3f}")
    sep = gg2[gg2["rho"] == "1:4"].iloc[0]
    print(f"  {'三类分离（最优）':<34}{sep['sr']:>10.3f}{sep['f1']:>9.3f}"
          f"   ← d=({int(sep['dzh'])},{int(sep['dsd'])},{int(sep['dsb'])})")

    # ── 表④ 段级聚合复核 ──
    print("\n" + "=" * 122)
    print("【表④ 段级聚合复核】d=(50,50,50), ρ=1:4 ｜ k=13")
    print(f"  {'n_keep':<9}" + "".join(f"{x:>10}" for x in AGGS))
    for nk in NKEEP:
        row = ""
        for a in AGGS:
            t = AG[(AG["nk"] == nk) & (AG["aggm"] == a)]
            row += f"{t['setF1'].mean():>10.3f}" if len(t) else f"{'-':>10}"
        print(f"  {nk:<9}{row}")

    # ── 表⑤ 配对自举 ──
    print("\n" + "=" * 122)
    print("【表⑤ 配对自举（4000 次）】k=13")
    def pick(dzh, dsd, dsb, rho):
        return g13[(g13["dzh"] == dzh) & (g13["dsd"] == dsd)
                   & (g13["dsb"] == dsb) & (g13["rho"] == rho)]
    cand = pick(int(bt["dzh"]), int(bt["dsd"]), int(bt["dsb"]), bt["rho"])
    refs = {"三 d 最优 vs 三类同 d=50（上一版最优）": (50, 50, 50, "1:4"),
            "三 d 最优 vs 三类同 d=20": (20, 20, 20, "1:4"),
            "三 d 最优 vs 三类同 d=100": (100, 100, 100, "1:4"),
            "三 d 最优 vs 纯 dense（d=50）": (50, 50, 50, "dense"),
            "三 d 最优 vs 纯 bm25（d=50）": (50, 50, 50, "bm25")}
    print(f"  {'对比':<42}{'ΔsetF1':>9}{'95% CI':>20}{'p(≤0)':>9}")
    for nm, (a, b2, c2, rr) in refs.items():
        ref = pick(a, b2, c2, rr)
        if not len(ref):
            continue
        r = boot(cand, ref)
        if r:
            print(f"  {nm:<42}{r[0]:>+9.3f}   [{r[1]:+.3f}, {r[2]:+.3f}]{r[3]:>9.3f}")
    print(f"\n  → 已写 results/R2_RD2_{{CURVE,GRID,AGG}}.csv（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
