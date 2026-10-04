r"""**最终召回流水线**：融合 → 重排 → 截断 k（50 篇口径）

## 用户定稿的架构
```
中文题面 → 1 路 dense ┐
n 路英文子查询 → 各 1 路 dense ├→ 每路先截 top-d → RRF 融合（ρ=1:4）→ 候选块池 C
n 路英文子查询 → 各 1 路 BM25 ┘      d = (zh:100, sq-dense:50, sq-bm25:50)
   → ★ 重排（cross-encoder，bge-reranker-v2-m3，本地已有）
   → 段级聚合（篇内 max）
   → 截断 top-k 篇  ← ★ k **重测**（本篇核心）
   → 交付 → 【C3 独立召回 + 逐条验证】（**不在本脚本内**，见下）
```

## ⚠️ 两阶段（必须）：6GB 卡上 bge-m3 与 reranker **不能同驻**
第一版把两个模型同时加载 → 显存 5.7/6.0 GiB、仅剩 ~200 MiB →
**Windows 驱动开始把显存换页到主机内存**（GPU 100% 占用但几乎无进展，看起来像"卡死"）。
→ 改为 **阶段 A 只载 bge-m3（算完融合池）→ 卸载 → 阶段 B 只载 reranker（批量重排）**。

## 本脚本三件事
| # | 做什么 | 为什么 |
|---|---|---|
| **1** | **体积比体检** | `zh-dense` 只 **1 路**而 sq 类各 **3 路** → 同一 `d` 下体积天然 1:3:3。报告三类块数、池/全库，并给出"三类**等体积**时 d 应取多少" |
| **2** | **加重排** | 融合（RRF，只看名次、无精排能力）→ cross-encoder 精排；对照 **有/无重排** |
| **3** | **重测 k** | 下游是 **C3 独立召回 + 逐条验证**（高召回优先）→ **k 判据不该是 setP，而是覆盖率**；k 扫 3~50，同时报 StRecall/setP/setF1/MRecall/α-nDCG |

## ★「独立召回 + 逐条验证」加在哪（文档已定，非本脚本内）
`R2_CHUNK_SWITCH_TODO_20260929.md` 的 **C3**：*"answer recaller 逐段落独立预测，优先高召回 →
逐候选独立验证"*，即 **ALCE 的 map-reduce + verify**，位于 **"交付 top-k 篇"之后**：

```
融合 → 重排 → 截断 top-k 篇 ──交付──▶ C3：对每个交付篇的**每个段落**独立问"支持该 facet 吗"
                                        （map，不比较段落间优劣）→ 逐条 verify（reduce）
```
→ 它**不是**排序链里的一个环节，而是**下游消费者**；本脚本负责导出它的**输入**（交付包）。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_final_pipeline.py --corpus 50
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_final_pipeline.py --corpus 50 --no-rerank
"""
from __future__ import annotations

import argparse
import gc
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
D_ZH, D_SD, D_SB = 100, 50, 50          # ★ 用户定稿
RHO = 4.0                                # bm25 权重（两类 dense 恒 1.0）
K_LIST = [3, 5, 8, 10, 13, 20, 30, 50]
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
    j = (a.set_index(keys)[col].to_frame("a")
         .join(b.set_index(keys)[col].to_frame("b"), how="inner"))
    d = (j["a"] - j["b"]).to_numpy()
    if not len(d):
        return None
    m = d[RNG.integers(0, len(d), size=(4000, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float((m <= 0).mean()), len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--subqueries", default="subqueries_v2.json")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rr-q", default="both", choices=["zh", "both", "sub"],
                    help="重排 query：zh = 仅中文题面；both = 题面 + 子查询拼接；sub = 仅子查询")
    ap.add_argument("--rr-batch", type=int, default=16)
    ap.add_argument("--enc-batch", type=int, default=8)
    args = ap.parse_args()
    if args.corpus == "50":
        CDIR, KDIR = DEV / "corpus50", DEV / "prodchunk50"
        GF, pmap_f = DEV / "gold_final3.csv", DEV / "corpus50" / "pdf_map_all.json"
    else:
        CDIR, KDIR = DEV / "clusters", DEV / "prodchunk"
        GF, pmap_f = DEV / "gold_final2.csv", DEV / "pdf_map.json"
    pm_ok = {d for d, v in json.loads(pmap_f.read_text(encoding="utf-8")).items() if v.get("ok")}
    gg = pd.read_csv(GF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    fac2cl = {f: c for (c, f) in combos}
    subq = json.loads((DEV / args.subqueries).read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 124)
    print(f"【最终流水线】语料 {args.corpus} ｜ 真值 {GF.name} ｜ 查询 {args.subqueries}")
    print(f"  融合 RRF(C={RRF_C}) ｜ d=(zh {D_ZH}, sq-dense {D_SD}, sq-bm25 {D_SB}) ｜ ρ={RHO:.0f}（bm25 加权）")
    print(f"  重排 {'❌ 关' if args.no_rerank else '✅ ' + args.rerank_model} ｜ query={args.rr_q}")
    print(f"  k 扫描 {K_LIST} ｜ 两阶段（模型不共驻）")
    t0 = time.time()

    # ══════════════ 阶段 A：只载 bge-m3 → 每簇的融合池 ══════════════
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    clusters: list[dict] = []
    for ci in range(3):
        ch = pd.read_parquet(KDIR / args.chunktag / f"c{ci}.parquet")
        ls = pd.read_parquet(CDIR / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"]) and d in pm_ok]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        didx = {d: np.where(owner == d)[0] for d in docs}
        nb = len(chunks)
        E = enc.encode(chunks, batch_size=args.enc_batch, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)

        def rank_dense(q: str) -> np.ndarray:
            qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                            convert_to_numpy=True).astype(np.float32)
            r = np.empty(nb, dtype=np.int64)
            r[np.argsort(-(qv @ E.T)[0])] = np.arange(nb)
            return r

        def rank_bm25(q: str) -> np.ndarray:
            r = np.empty(nb, dtype=np.int64)
            r[np.argsort(-bm.scores(q).astype(np.float32))] = np.arange(nb)
            return r

        packs = []
        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            zh = str(F2[facet][1])
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3] or [str(F2[facet][3])]
            cls: dict[str, list[np.ndarray]] = {
                "zh-dense": [rank_dense(zh)],
                "sq-dense": [rank_dense(q) for q in SQ],
                "sq-bm25": [rank_bm25(q) for q in SQ]}
            dd = {"zh-dense": D_ZH, "sq-dense": D_SD, "sq-bm25": D_SB}
            sc = np.zeros(nb, dtype=np.float64)
            sets: dict[str, set] = {}
            for c, rs in cls.items():
                w = RHO if c == "sq-bm25" else 1.0
                u: set[int] = set()
                for r in rs:
                    m = r < dd[c]
                    sc[m] += w / (RRF_C + r[m] + 1)
                    u |= set(np.where(m)[0].tolist())
                sets[c] = u
            cand = sorted(sets["zh-dense"] | sets["sq-dense"] | sets["sq-bm25"])
            packs.append(dict(facet=facet, gold=gold, docs=docs, cand=cand, sc=sc,
                              owner=owner, didx=didx, nb=nb,
                              vs={"zh": len(sets["zh-dense"]), "sd": len(sets["sq-dense"]),
                                  "sb": len(sets["sq-bm25"])},
                              qr=(zh if args.rr_q == "zh" else
                                  (" ".join([zh] + SQ) if args.rr_q == "both"
                                   else " ".join(SQ)))))
        clusters.append(dict(ci=ci, nb=nb, chunks=chunks, packs=packs, docs=docs))
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {nb} 块 ｜ 候选池均 "
              f"{np.mean([len(p['cand']) for p in packs]):.0f} 块 ｜ {time.time() - t0:.0f}s",
              flush=True)
    del enc
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"  [阶段 A 完成] bge-m3 已卸载 {time.time() - t0:.0f}s", flush=True)

    # ══════════════ 阶段 B：只载 reranker → 逐簇重排 ══════════════
    if not args.no_rerank:
        rr = CrossEncoder(args.rerank_model, device=dev, max_length=512)
        if dev == "cuda":
            rr.model.half()
        for cd in clusters:
            pairs, spans = [], []
            for p in cd["packs"]:
                spans.append((len(pairs), len(pairs) + len(p["cand"])))
                pairs += [(p["qr"], cd["chunks"][j]) for j in p["cand"]]
            lg = np.asarray(rr.predict(pairs, batch_size=args.rr_batch,
                                       show_progress_bar=False), dtype=np.float64).reshape(-1)
            sig = 1.0 / (1.0 + np.exp(-lg))
            for p, (a, b2) in zip(cd["packs"], spans):
                p["rr"] = sig[a:b2]
            print(f"  簇{cd['ci'] + 1} 重排 {len(pairs)} 对 ｜ {time.time() - t0:.0f}s", flush=True)
        del rr
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ══════════════ 阶段 C：指标（纯 CPU）══════════════
    rows = []
    for cd in clusters:
        for p in cd["packs"]:
            owner, didx, docs, gold, cand = p["owner"], p["didx"], p["docs"], p["gold"], p["cand"]
            for arm, key in (("rrf", "sc"), ("rerank", "rr")):
                if key == "rr" and "rr" not in p:
                    continue
                s = p[key]
                if arm == "rrf":
                    dm = {x: float(s[didx[x]].max()) for x in docs}
                else:
                    best: dict[str, float] = {}
                    for k2, j in enumerate(cand):
                        x = owner[j]
                        v = float(s[k2])
                        if v > best.get(x, -9.9):
                            best[x] = v
                    dm = {x: best.get(x, 0.0) for x in docs}
                o = sorted(docs, key=lambda x: -dm[x])
                for k in K_LIST:
                    sp, _, f1 = _m.setpf(o, gold, k)
                    rows.append(dict(cluster=fac2cl[p["facet"]], facet=p["facet"], arm=arm, k=k,
                                     n_gold=len(gold), n_docs=len(docs),
                                     StRecall=_m.strecall(o, gold, k), setP=sp, setF1=f1,
                                     MRecall=_m.mrecall(o, gold, k),
                                     andcg=_m.alpha_ndcg(o, gold, k), random=k / len(docs)))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / "R2_FINAL_ROWS.csv", index=False, encoding="utf-8-sig")

    # ── 表① 体积比 ──
    print("\n" + "=" * 124)
    print("【表① 体积比体检】`zh-dense` 只 1 路，sq 类各 3 路 → 同 d 下体积天然 1:3:3")
    print(f"  {'簇':>3}{'zh 块':>8}{'sq-dense':>10}{'sq-bm25':>9}{'池(并集)':>10}"
          f"{'全库块':>9}{'池/全库':>9}{'zh 占比':>9}")
    VOL = []
    for cd in clusters:
        z = np.mean([p["vs"]["zh"] for p in cd["packs"]])
        sd = np.mean([p["vs"]["sd"] for p in cd["packs"]])
        sb = np.mean([p["vs"]["sb"] for p in cd["packs"]])
        pl = np.mean([len(p["cand"]) for p in cd["packs"]])
        VOL.append((z, sd, sb, pl, cd["nb"]))
        print(f"  {cd['ci'] + 1:>3}{z:>8.0f}{sd:>10.0f}{sb:>9.0f}{pl:>10.0f}{cd['nb']:>9.0f}"
              f"{pl / cd['nb']:>9.1%}{z / (z + sd + sb):>9.1%}")
    Z, SD, SB = (np.mean([v[0] for v in VOL]), np.mean([v[1] for v in VOL]),
                 np.mean([v[2] for v in VOL]))
    POOL, NB = np.mean([v[3] for v in VOL]), np.mean([v[4] for v in VOL])
    print(f"\n  合计均：zh {Z:.0f} ｜ sq-dense {SD:.0f} ｜ sq-bm25 {SB:.0f}"
          f" ｜ 池 {POOL:.0f} ｜ 池/全库 {POOL / NB:.1%}")
    print(f"  ★ 体积比（zh : sd : sb）= 1 : {SD / Z:.2f} : {SB / Z:.2f}（**路数比 = 1 : 3 : 3**）")
    print(f"  → 若要**三类等体积**（各 ≈{POOL / 3:.0f} 块）："
          f"d_zh≈{D_ZH * (POOL / 3) / Z:.0f} ｜ d_sd≈{D_SD * (POOL / 3) / SD:.0f}"
          f" ｜ d_sb≈{D_SB * (POOL / 3) / SB:.0f}")

    # ── 表② k 扫描 ──
    print("\n" + "=" * 124)
    print("【表② ★ k 重测】下游 =「独立召回 + 逐条验证」（高召回优先）→ **同时看覆盖率与精确率**")
    print(f"  {'k':>4}{'StRecall':>10}{'setP':>8}{'setF1':>9}{'MRecall':>9}{'α-nDCG':>9}"
          f"{'随机岭':>9}   臂")
    for arm in ("rrf", "rerank"):
        t = R[R["arm"] == arm]
        if not len(t):
            continue
        for k in K_LIST:
            s = t[t["k"] == k]
            print(f"  {k:>4}{s['StRecall'].mean():>10.3f}{s['setP'].mean():>8.3f}"
                  f"{s['setF1'].mean():>9.3f}{s['MRecall'].mean():>9.3f}"
                  f"{s['andcg'].mean():>9.3f}{s['random'].mean():>9.3f}   {arm}")
        print()

    # ── 表③ 重排收益 ──
    if not args.no_rerank:
        print("=" * 124)
        print("【表③ 重排收益】配对自举 4000 次（rerank − rrf，同 k）")
        print(f"  {'k':>4}{'ΔStRecall':>12}{'ΔsetF1':>10}{'Δα-nDCG':>10}{'95% CI(ΔsetF1)':>22}{'p(≤0)':>9}")
        for k in K_LIST:
            a = R[(R["arm"] == "rerank") & (R["k"] == k)]
            b = R[(R["arm"] == "rrf") & (R["k"] == k)]
            r1, r2_, r3 = boot(a, b, "setF1"), boot(a, b, "StRecall"), boot(a, b, "andcg")
            if r1 and r2_ and r3:
                print(f"  {k:>4}{r2_[0]:>+12.3f}{r1[0]:>+10.3f}{r3[0]:>+10.3f}"
                      f"   [{r1[1]:+.3f}, {r1[2]:+.3f}]{r1[3]:>9.3f}")

    # ── 表④ 逐簇 ──
    print("\n" + "=" * 124)
    k_show = 13
    print(f"【表④ 逐簇（k={k_show}，两臂）】")
    print(f"  {'簇':>3}{'arm':>9}{'题数':>6}{'StRecall':>10}{'setP':>8}{'setF1':>9}{'MRecall':>9}{'α-nDCG':>9}")
    for c in (1, 2, 3):
        for arm in ("rrf", "rerank"):
            t = R[(R["cluster"] == c) & (R["arm"] == arm) & (R["k"] == k_show)]
            if not len(t):
                continue
            print(f"  {c:>3}{arm:>9}{len(t):>6}{t['StRecall'].mean():>10.3f}{t['setP'].mean():>8.3f}"
                  f"{t['setF1'].mean():>9.3f}{t['MRecall'].mean():>9.3f}{t['andcg'].mean():>9.3f}")
    print(f"\n  → 已写 results/R2_FINAL_ROWS.csv（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
