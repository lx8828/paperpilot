"""**召回设计标定**（50 篇口径，零 LLM）——按用户架构逐步实现

## 用户架构
```
中文题面 → 1 路 dense ┐
n 路英文子查询 → 各 1 路 dense ├→ 每路**先截 top-d** → RRF 融合（权重 ρ）→ 统一候选块 top-K
n 路英文子查询 → 各 1 路 BM25 ┘
   → 段级聚合去重（块→论文，**每篇最多留 n_keep 块**，聚合方式 agg）
   → 交付 top-k 篇 → 独立召回 + 逐条验证
```

## 本脚本的四个标定阶段
| 阶段 | 问题 | 网格 |
|---|---|---|
| **A** | 每路**先截多少** `d`、融合**权重** `ρ` 多少？ | d × ρ × k |
| **B** | 段级聚合怎么算？每篇留几块？ | `n_keep` × `agg` |
| **C** | 统一候选块 `top-K` 截断要不要、截多少？ | K ∈ {k·nk, 2k·nk, 5k·nk, ∞} |
| **D** | 与现状/旧结论的**配对显著性** | 4000 次自举 |
+ **E** 逐 facet（最优配置下）看分型建议是否兑现

## 依据上一轮体检的两个改动
1. **权重区间放宽到 bm25 主导区** —— `sq{1,2,3}-bm25` 是**全 9 路最强**（0.473~0.529），
   而旧结论 ρ=4:1（dense 主导）建立在"BM25 只用弱查询 `para`"之上 → **作废**
2. **对照"去掉 `para`"** —— 它在 dense 路（0.445）与 `zh`（0.448）几乎相同，且是**最弱的 BM25 查询**

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_recall_design.py --corpus 50
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
# ★ `d` = 每路保留 **top-d 个块**（块级深度！不是"篇数"）。
#   库约 1,079 块、约 22 块/篇 → `d=50` ≈ 2.3 篇的块量。`d=0` = **不截**（全库）。
#   ⚠️ 第一版把 `d` 上限设成 50，结果最优落在**边界**上 → 必须往外扩才能定位最优。
D_GRID = [10, 20, 50, 100, 200, 0]
B_D_GRID = [0, 50, 100, 200]                # 表 B/C 在哪些块深度上扫
# ρ = w_bm25 / w_dense；None = 纯 bm25。**对称搜索到 bm25 主导区**
RHO_GRID: list = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, None]
K_GRID = [5, 10, 13, 20]
NKEEP_GRID = [1, 3, 6]
AGGS = ["max", "sum", "mean"]
KCMULT_GRID = [1, 2, 5, 0]                   # 统一候选块截断 = kc × k × n_keep；0 = 不截
RHO_LBL = {0.0: "dense", 0.25: "4:1", 0.5: "2:1", 1.0: "1:1", 2.0: "1:2",
           4.0: "1:4", None: "bm25"}
RNG = np.random.default_rng(20260930)


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def boot(a: pd.DataFrame, b: pd.DataFrame, col: str = "setF1", keys=("cluster", "facet", "k")):
    j = (a.set_index(list(keys))[col].to_frame("a")
         .join(b.set_index(list(keys))[col].to_frame("b"), how="inner"))
    if not len(j):
        return None
    d = (j["a"] - j["b"]).to_numpy()
    m = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float((m <= 0).mean()), len(d))


def agg_doc(sc: np.ndarray, didx: dict, docs: list[str], nk: int, agg: str) -> dict:
    out = {}
    for d in docs:
        v = np.sort(sc[didx[d]])[::-1][:nk]
        out[d] = (float(v.max()) if agg == "max"
                  else float(v.sum()) if agg == "sum" else float(v.mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--paravariant", action="store_true",
                    help="同时跑「含 para」变体（默认只跑「去 para」）")
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
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    print("=" * 124)
    print(f"【召回设计标定】语料 {args.corpus}（{CDIR.name}）｜ 真值 {GF.name}"
          f" ｜ 切块 {KDIR.name}/{args.chunktag} ｜ RRF_C={RRF_C}")
    print(f"  列表：zh-dense ×1 ｜ sq{'{1..n}'}-dense ×n ｜ sq{'{1..n}'}-bm25 ×n"
          f" ｜ **默认去掉 `para`**（它 dense≈zh、bm25 最弱）")
    print(f"  d∈{D_GRID} ｜ ρ∈{[RHO_LBL[r] for r in RHO_GRID]} ｜ k∈{K_GRID}"
          f" ｜ n_keep∈{NKEEP_GRID} ｜ agg∈{AGGS}")

    A_rows, B_rows, C_rows, O_rows = [], [], [], []
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
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {len(chunks)} 块 ｜ {time.time() - t0:.0f}s",
              flush=True)

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            N = len(docs)
            zh, para = str(F2[facet][1]), str(F2[facet][3])
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3] or [para]
            variants = {"nopara": ([zh] + SQ, SQ)}
            if args.paravariant:
                variants["withpara"] = ([zh, para] + SQ, [para] + SQ)
            for vname, (DQ, BQ) in variants.items():
                lists: list[tuple[str, np.ndarray]] = []
                for i, q in enumerate(DQ):
                    qv = enc.encode([q], normalize_embeddings=True,
                                    show_progress_bar=False,
                                    convert_to_numpy=True).astype(np.float32)
                    s = (qv @ E.T)[0]
                    r = np.empty(len(s), dtype=np.int64)
                    r[np.argsort(-s)] = np.arange(len(s))
                    lists.append(("dense", r))
                for q in BQ:
                    s = bm.scores(q).astype(np.float32)
                    r = np.empty(len(s), dtype=np.int64)
                    r[np.argsort(-s)] = np.arange(len(s))
                    lists.append(("bm25", r))
                nb = len(chunks)

                # ── A：d × ρ × k（agg=max, n_keep=1）──
                for d in D_GRID:
                    for rho in RHO_GRID:
                        sc = np.zeros(nb, dtype=np.float64)
                        for kind, r in lists:
                            # ρ = w_bm25/w_dense；rho=None → 纯 bm25
                            w = (1.0 if kind == "bm25" else 0.0) if rho is None \
                                else (1.0 if kind == "dense" else float(rho))
                            if w <= 0:
                                continue
                            m = np.ones(nb, bool) if d == 0 else (r < d)   # ★ 每路只保留 top-d
                            sc[m] += w / (RRF_C + r[m] + 1)
                        dm = {dd: float(sc[didx[dd]].max()) for dd in docs}
                        o = sorted(docs, key=lambda dd: -dm[dd])
                        for k in K_GRID:
                            p, r_, f1 = _m.setpf(o, gold, k)
                            A_rows.append(dict(cluster=ci + 1, facet=facet, variant=vname,
                                               d=d, rho=RHO_LBL[rho], k=k, nk=1, aggm="max",
                                               n_gold=len(gold), n_docs=N,
                                               StRecall=_m.strecall(o, gold, k), setP=p,
                                               setF1=f1, MRecall=_m.mrecall(o, gold, k),
                                               random=k / N))
                # ── B/C：在 **每个 (块深度 d, ρ)** 下都算 ——
                #    ⚠️ 第一版固定 ρ=1:1 / d=不截，与表 A 的最优配置不可比
                for dB in B_D_GRID:
                    for rho in RHO_GRID:
                        sc0 = np.zeros(nb, dtype=np.float64)
                        for kind, r in lists:
                            w = (1.0 if kind == "bm25" else 0.0) if rho is None \
                                else (1.0 if kind == "dense" else float(rho))
                            if w <= 0:
                                continue
                            m = np.ones(nb, bool) if dB == 0 else (r < dB)
                            sc0[m] += w / (RRF_C + r[m] + 1)
                        for nk in NKEEP_GRID:
                            for agg in AGGS:
                                dm = agg_doc(sc0, didx, docs, nk, agg)
                                o = sorted(docs, key=lambda dd: -dm[dd])
                                for k in K_GRID:
                                    p, r_, f1 = _m.setpf(o, gold, k)
                                    # ⚠️ 列名 `aggm` 不能用 `agg`（`DataFrame.agg` 是方法名）
                                    B_rows.append(dict(
                                        cluster=ci + 1, facet=facet, variant=vname, d=dB,
                                        rho=RHO_LBL[rho], nk=nk, aggm=agg, k=k,
                                        n_gold=len(gold), n_docs=N,
                                        StRecall=_m.strecall(o, gold, k), setP=p, setF1=f1,
                                        MRecall=_m.mrecall(o, gold, k)))
                        for kc in KCMULT_GRID:
                            k = 13
                            Kc = nb if kc == 0 else min(int(kc * k * 3), nb)
                            keepb = np.argsort(-sc0)[:Kc]
                            cut = np.zeros(nb, dtype=np.float64)
                            cut[keepb] = sc0[keepb]
                            dm = agg_doc(cut, didx, docs, 3, "mean")
                            o = sorted(docs, key=lambda dd: -dm[dd])
                            p, r_, f1 = _m.setpf(o, gold, k)
                            C_rows.append(dict(cluster=ci + 1, facet=facet, variant=vname,
                                               d=dB, rho=RHO_LBL[rho], kc=kc, K_chunks=Kc,
                                               k=k, nk=3, aggm="mean", n_gold=len(gold),
                                               n_docs=N, StRecall=_m.strecall(o, gold, k),
                                               setP=p, setF1=f1))

                # ── E：两通道**并集上界**（每路 top-d 并集里挑最优 k）──
                for k in (13,):
                    uni = set()
                    for _kind, r in lists:
                        uni |= set(np.where(r < 13)[0].tolist())
                    pool = {owner[i] for i in uni}
                    tp = len(pool & gold)
                    bk = min(tp, k)
                    p = bk / k
                    r_ = bk / len(gold)
                    # ⚠️ 列名用 `pool_cov` 而非 `cov` —— `DataFrame.cov` 是方法名
                    O_rows.append(dict(cluster=ci + 1, facet=facet, variant=vname, k=k,
                                       pool_docs=len(pool), pool_gold=tp,
                                       pool_cov=tp / len(gold),
                                       oracleF1=(2 * p * r_ / (p + r_) if (p + r_) else 0.0)))

    A, B, C, O = (pd.DataFrame(A_rows), pd.DataFrame(B_rows),
                  pd.DataFrame(C_rows), pd.DataFrame(O_rows))
    for df, nm in ((A, "GRID"), (B, "AGG"), (C, "KCUT"), (O, "ORACLE")):
        df.to_csv(HERE / "results" / f"R2_RD_{nm}.csv", index=False, encoding="utf-8-sig")

    # ══ 表 A ══
    BESTLIST: list[tuple] = []
    for vn in A.variant.unique():
        print("\n" + "=" * 124)
        print(f"【表 A｜d × ρ 网格】变体 **{vn}** ｜ k=13 ｜ 篇分=篇内 max（n_keep=1）")
        a = A[(A.variant == vn) & (A.k == 13)]
        piv = a.pivot_table(index="d", columns="rho", values="setF1", aggfunc="mean")
        order = [RHO_LBL[r] for r in RHO_GRID]
        print(f"  {'d(每路块深度)':<14}" + "".join(f"{o:>9}" for o in order if o in piv.columns))
        for d in D_GRID:
            if d not in piv.index:
                continue
            lbl = "0(不截)" if d == 0 else f"{d}"
            print(f"  {lbl:<14}" + "".join(f"{piv.loc[d, o]:>9.3f}"
                                           for o in order if o in piv.columns))
        # ⚠️ 必须**先按配置聚合再取最优** —— 直接 `sort_values().iloc[0]` 会取到
        #    "最好的那一题"（setF1=0.833），不是最优配置（第一版就是这个 bug）
        ag = (a.groupby(["d", "rho"])
              .agg(f1=("setF1", "mean"), sr=("StRecall", "mean"),
                   sp=("setP", "mean"), mc=("MRecall", "mean")).reset_index())
        rb = ag.sort_values("f1", ascending=False).iloc[0]
        print(f"  ★ 最优：d=**{int(rb['d'])}** ｜ ρ=**{rb['rho']}** → setF1 **{rb['f1']:.3f}**"
              f"（StRecall {rb['sr']:.3f}、setP {rb['sp']:.3f}、MRecall {rb['mc']:.3f}）")
        BESTLIST.append((vn, int(rb["d"]), rb["rho"], float(rb["f1"])))
    vn, BD, BRHO, BF1 = max(BESTLIST, key=lambda x: x[3])
    print(f"\n  ★★ 全局最优：变体 **{vn}** ｜ d=**{BD}** ｜ ρ=**{BRHO}** → setF1 **{BF1:.3f}**")

    # ══ 表 B ══
    print("\n" + "=" * 124)
    print(f"【表 B｜段级聚合】块深度 d={BD}{'(不截)' if BD == 0 else ''} ｜ **ρ={BRHO}**"
          f"（= 表 A 最优）｜ k=13（n_keep × 聚合方式）")
    b = B[(B["k"] == 13) & (B["variant"] == vn) & (B["rho"] == BRHO) & (B["d"] == BD)]
    print(f"  {'n_keep':<9}" + "".join(f"{x:>10}" for x in AGGS))
    for nk in NKEEP_GRID:
        row = ""
        for aggm in AGGS:
            t = b[(b["nk"] == nk) & (b["aggm"] == aggm)]
            row += f"{t['setF1'].mean():>10.3f}" if len(t) else f"{'-':>10}"
        print(f"  {nk:<9}{row}")
    bg = b.groupby(["nk", "aggm"]).agg(f1=("setF1", "mean")).reset_index()
    bb = bg.sort_values("f1", ascending=False).iloc[0]
    print(f"  ★ 最优：n_keep=**{int(bb['nk'])}** ｜ 聚合=**{bb['aggm']}** → setF1 **{bb['f1']:.3f}**")
    b11 = B[(B["k"] == 13) & (B["variant"] == vn) & (B["rho"] == BRHO) & (B["d"] == 0)]
    if len(b11):
        print(f"  （对照：同一 ρ 下**不截**的最优 "
              f"{b11.groupby(['nk','aggm']).setF1.mean().max():.3f}）")

    # ══ 表 C ══
    print("\n" + "=" * 124)
    print(f"【表 C｜统一候选块 top-K 截断】块深度 d={BD}{'(不截)' if BD == 0 else ''}, "
          f"**ρ={BRHO}**, n_keep=3, 聚合=mean, k=13")
    print(f"  {'K 倍数':<14}{'块数':>8}{'StRecall':>10}{'setP':>8}{'setF1':>9}")
    c = C[(C["k"] == 13) & (C["variant"] == vn) & (C["rho"] == BRHO) & (C["d"] == BD)]
    for kc in KCMULT_GRID:
        t = c[c["kc"] == kc]
        if not len(t):
            continue
        lbl = "不截（全库）" if kc == 0 else f"{kc}×k×nk"
        print(f"  {lbl:<14}{t['K_chunks'].mean():>8.0f}{t['StRecall'].mean():>10.3f}"
              f"{t['setP'].mean():>8.3f}{t['setF1'].mean():>9.3f}")

    # ══ 表 D ══
    print("\n" + "=" * 124)
    print("【表 D｜配对自举（4000 次，28 题）× 关键对照】k=13")
    print(f"  {'对比':<44}{'ΔsetF1':>9}{'95% CI':>20}{'p(≤0)':>9}")
    cand = A[(A["k"] == 13) & (A["variant"] == vn) & (A["d"] == BD) & (A["rho"] == BRHO)]
    refs = {
        "最优 vs 纯 dense（d=50）": (50, "dense"),
        "最优 vs 旧结论 ρ=4:1（d=50）": (50, "4:1"),
        "最优 vs 纯 bm25（d=50）": (50, "bm25"),
        "最优 vs 每路只截 top-5": (5, BRHO),
        "最优 vs 每路只截 top-10": (10, BRHO),
        "最优 vs 每路只截 top-20": (20, BRHO),
    }
    for nm, (dd_, rr_) in refs.items():
        ref = A[(A["k"] == 13) & (A["variant"] == vn) & (A["d"] == dd_) & (A["rho"] == rr_)]
        if not len(ref):
            continue
        r = boot(cand, ref)
        if r:
            print(f"  {nm:<44}{r[0]:>+9.3f}   [{r[1]:+.3f}, {r[2]:+.3f}]{r[3]:>9.3f}")

    # ══ 表 E ══
    print("\n" + "=" * 124)
    print("【表 E｜并集上界 + 逐 facet 短板】k=13")
    o = O[(O["k"] == 13) & (O["variant"] == vn)]
    ocov, of1 = o["pool_cov"].mean(), o["oracleF1"].mean()
    print(f"  并集上界（每路 top-13 的并集）：池覆盖 gold **{ocov:.1%}**"
          f" ｜ 完美重排 setF1 **{of1:.3f}** ｜ 最优实际 **{BF1:.3f}**"
          f" → 留白 **{of1 - BF1:+.3f}**")
    fa = A[(A["k"] == 13) & (A["d"] == BD) & (A["rho"] == BRHO)
           & (A["variant"] == vn)].copy()
    fa["key"] = fa["cluster"].astype(str) + " " + fa["facet"]
    fa = fa.sort_values("setF1")
    print(f"\n  {'簇 facet':<22}{'gold':>5}{'StRecall':>10}{'setP':>8}{'setF1':>9}")
    for _, r in fa.iterrows():
        print(f"  {r['key']:<22}{int(r['n_gold']):>5}{r['StRecall']:>10.3f}"
              f"{r['setP']:>8.3f}{r['setF1']:>9.3f}")
    print(f"\n  → 已写 results/R2_RD_{{GRID,AGG,KCUT,ORACLE}}.csv（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
