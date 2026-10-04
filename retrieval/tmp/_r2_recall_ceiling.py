"""**多通道召回：现状 + 上限**（S2 的核心诊断，零 LLM）

## 三个要回答的问题
1. **现在只有 dense，效果如何？** → 篇级 StRecall/MRecall/集合P-F1 + 证据召回
2. **多通道召回的上限有多少？** → **上限 = 各通道 top-k 的并集覆盖率**
   （任何融合方式的召回都不可能超过它；与 **RRF 实际融合** 的差距 = **融合损失**）
3. **每条通道的边际贡献？** → **独占 gold**（只有该通道召回到的 gold 篇）

## 通道与查询集（口径必须分开，见 `R2_C2A_AND_PIVOT_20260930.md` §1.4/§3.5）
| 通道 | 查询集 |
|---|---|
| `dense`（bge-m3 语义） | 中文题面 + 英文检索式（多查询取 max） |
| `sparse`（bge-m3 学习型词法） | **必须英文**（纯中文恒 0，见 S1 验收 2） |
| `bm25`（字面词法，ASCII） | 同上 |

| 查询口径 | 内容 | 对应 |
|---|---|---|
| **Q_dev** | 中文题面 + **3 条英文子查询** | 我们评测一直在用的（偏富） |
| **Q_prod** | 中文题面 + **1 条英文检索式**（用 F2 的 `para` 近似 L6） | **贴近生产** |

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_recall_ceiling.py
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
DEV = HERE / "data" / "r2dev"
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
mrecall, strecall, setpf = _m.mrecall, _m.strecall, _m.setpf

f2spec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2 = _iu.module_from_spec(f2spec)
sys.modules["facets2"] = _f2
f2spec.loader.exec_module(_f2)
F2 = _f2.F2

bspec = _iu.spec_from_file_location("bgem3", HERE / "tmp" / "_bge_m3.py")
_b3 = _iu.module_from_spec(bspec)
sys.modules["bgem3"] = _b3
bspec.loader.exec_module(_b3)

CHAR_BUDGET = 8000          # 证据召回口径：每篇字符预算（A3'）
RRF_C = 60
# ⚠️ `k` 必须跟着**库大小**走：小库（13~18 篇）k≥14 就退化成"交全库"；
#    50 篇库（每簇 50）→ k=20 只占 40%，k=30 占 60%，才第一次有区分度。
KS_SMALL = [5, 10, 14, 20]
KS_BIG = [5, 10, 20, 30, 50]


def _rrf(orders: list[list[str]]) -> list[str]:
    """RRF 融合（等权），返回合并后的排序。"""
    sc: dict[str, float] = {}
    for o in orders:
        for i, d in enumerate(o):
            sc[d] = sc.get(d, 0.0) + 1.0 / (RRF_C + i + 1)
    return sorted(sc, key=lambda d: -sc[d])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--corpus", default="20", choices=["20", "50"],
                    help="语料规模：20 = 原 3×20 篇（clusters/ + gold_final2）；"
                         "50 = 扩语料 3×50 篇（corpus50/ + prodchunk50 + gold_final3）")
    args = ap.parse_args()

    if args.corpus == "50":
        CDIR, KDIR = DEV / "corpus50", DEV / "prodchunk50"
        GF = DEV / "gold_final3.csv"
        pmap_f = DEV / "corpus50" / "pdf_map_all.json"
        KS = KS_BIG
    else:
        CDIR, KDIR = DEV / "clusters", DEV / "prodchunk"
        GF = DEV / "gold_final2.csv"
        pmap_f = DEV / "pdf_map.json"
        KS = KS_SMALL
    pm_ok = {d for d, v in json.loads(pmap_f.read_text(encoding="utf-8")).items()
             if v.get("ok")}

    meta = json.loads((DEV / "clusters" / "meta.json").read_text(encoding="utf-8"))
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    gg = pd.read_csv(GF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json").read_text(encoding="utf-8"))["per_combo"]}

    t0 = time.time()
    m = _b3.BGEM3()
    print("=" * 122)
    print(f"【多通道召回：现状 + 上限】语料规模 **{args.corpus}**（{CDIR.name}）｜ 真值 {GF.name}"
          f" ｜ 切块 {KDIR.name}/{args.chunktag} ｜ k∈{KS}"
          f" ｜ 证据召回@每篇 {CHAR_BUDGET} 字符 ｜ 加载 {time.time() - t0:.0f}s")

    rows, excl, ev_rows = [], [], []
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
        lens = np.array([len(c) for c in chunks], dtype=np.int64)
        idx = {d: np.where(owner == d)[0] for d in docs}
        t1 = time.time()
        E = m.encode(chunks, max_len=args.max_len, batch=4)
        bm = _m.BM25(chunks, tok=_m._tok)          # 字面 BM25（ASCII；英文查询可用）
        print(f"  【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块 ｜ 三路编码 {time.time() - t1:.0f}s"
              f" ｜ sparse 非零 token 中位 {int(np.median([len(x) for x in E['sparse']]))}")

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            _, zh, anchor, para = F2[facet]
            SQ = list(subq.get(facet, []))
            QS = {"Q_dev": [zh] + SQ, "Q_prod": [zh] + [para]}
            for qn, qlist in QS.items():
                # dense：多查询取 max（中文走 dense 才有语义）
                dq = [q for q in qlist if q.strip()]
                QE = m.encode(dq, max_len=args.max_len, batch=4)
                S_d = (QE["dense"] @ E["dense"].T).max(axis=0)
                # sparse / bm25：**只用英文**查询（中文恒 0，见 S1）
                eq = [q for q in dq if not any("\u4e00" <= c <= "\u9fff" for c in q)] or dq
                S_s = np.zeros(len(chunks), dtype=np.float32)
                for q in eq:
                    qo = m.encode([q], max_len=args.max_len)["sparse"][0]
                    S_s = np.maximum(S_s, np.array([m.sparse_score(qo, x) for x in E["sparse"]],
                                                   dtype=np.float32))
                S_b = np.max([bm.scores(q) for q in eq], axis=0)

                chans = {"dense": S_d, "sparse": S_s, "bm25": S_b}
                # 篇级排序（篇内 max）
                pm = {n: {d: float(s[idx[d]].max()) for d in docs} for n, s in chans.items()}
                orders = {n: sorted(docs, key=lambda d: -v[d]) for n, v in pm.items()}
                # 证据召回（块级，等字符预算；目标集 = **锚点命中块**代理）
                import re as _re
                hit = np.array([bool(_re.search(F2[facet][0], c, _re.I)) for c in chunks])
                n_hit = int(hit.sum())

                def ev(sel: list[int]) -> float:
                    return float(hit[sel].sum() / n_hit) if n_hit else float("nan")

                def topc(s: np.ndarray, per_doc: int) -> list[int]:
                    order = np.argsort(-s)
                    used: dict[str, int] = {}
                    out: list[int] = []
                    for i in order:
                        d = owner[i]
                        if used.get(d, 0) + lens[i] > per_doc:
                            continue
                        out.append(int(i))
                        used[d] = used.get(d, 0) + int(lens[i])
                    return out

                # ── 逐通道 + 并集上限 + RRF ──
                for k in KS:
                    # ★ 随机交付 k 篇的期望 StRecall = k/N —— 必须拿它作为**下界**校正
                    #   （小库时 k/N 很高 → 召回数字会被"库小"撑起来，见 R2_RECALL_CEILING §0）
                    rec = dict(cluster=ci + 1, facet=facet, qset=qn, k=k,
                               n_gold=len(gold), n_docs=len(docs),
                               k_ratio=k / len(docs), random_StRecall=k / len(docs))
                    for n, o in orders.items():
                        p, r, f1 = setpf(o, gold, k)
                        rec[f"{n}_MRecall"] = mrecall(o, gold, k)
                        rec[f"{n}_StRecall"] = strecall(o, gold, k)
                        rec[f"{n}_P"], rec[f"{n}_F1"] = p, f1
                    # ⚠️ 并集**不能截回 k**（截了就等于 dense 单独，因为 dense 排在最前）——
                    #    并集的价值正在于"可以多于 k 项"
                    uni = list(dict.fromkeys([x for o in orders.values() for x in o[:k]]))
                    tp = len(set(uni) & gold)
                    bk = min(tp, k)                      # k 限交付下的最优子集
                    pb = bk / k if k else 0.0
                    rb = bk / len(gold) if gold else 0.0
                    rec.update(union_n=len(uni), union_tp=tp,
                               union_cov=tp / len(gold) if gold else float("nan"),
                               boracle_StRecall=rb,
                               boracle_P=pb,
                               boracle_F1=(2 * pb * rb / (pb + rb) if (pb + rb) else 0.0),
                               boracle_MRecall=float(tp >= min(len(gold), k)))
                    rr = _rrf(list(orders.values()))
                    pr, rr_, f1r = setpf(rr, gold, k)
                    rec.update(rrf_MRecall=mrecall(rr, gold, k),
                               rrf_StRecall=strecall(rr, gold, k),
                               rrf_P=pr, rrf_F1=f1r)
                    rows.append(rec)

                # ── 独占 gold（该 k=10 下只有某个通道召回到的 gold）──
                top10 = {n: set(o[:10]) & gold for n, o in orders.items()}
                for n in orders:
                    others = set().union(*[v for m2, v in top10.items() if m2 != n]) if len(top10) > 1 else set()
                    e = top10[n] - others
                    if e:
                        excl.append(dict(cluster=ci + 1, facet=facet, qset=qn, channel=n,
                                         n_exclusive=len(e), gold=sorted(e),
                                         n_gold=len(gold)))
                # ── 证据召回（等字符预算）──
                for n, s in chans.items():
                    sel = topc(s, CHAR_BUDGET)
                    ev_rows.append(dict(cluster=ci + 1, facet=facet, qset=qn, channel=n,
                                        n_chunks=len(sel),
                                        chars=int(lens[np.array(sel)].sum()) if sel else 0,
                                        chars_per_doc=(lens[np.array(sel)].sum() / len(docs)) if sel else 0.0,
                                        ev_recall=ev(sel)))
                # 并集证据池（三通道合并后同样预算）
                S_u = (S_d / max(S_d.max(), 1e-9) + S_s / max(S_s.max(), 1e-9)
                       + S_b / max(S_b.max(), 1e-9))
                sel = topc(S_u, CHAR_BUDGET)
                ev_rows.append(dict(cluster=ci + 1, facet=facet, qset=qn, channel="union3",
                                    n_chunks=len(sel),
                                    chars=int(lens[np.array(sel)].sum()) if sel else 0,
                                    chars_per_doc=(lens[np.array(sel)].sum() / len(docs)) if sel else 0.0,
                                    ev_recall=ev(sel)))

    d = pd.DataFrame(rows)
    e = pd.DataFrame(excl)
    v = pd.DataFrame(ev_rows)
    suffix = "" if args.corpus == "20" else f"_n{args.corpus}"
    for df, nm in ((d, "CEIL"), (e, "EXCL"), (v, "EV")):
        df.to_csv(HERE / "results" / f"R2_RECALL{suffix}_{nm}.csv",
                  index=False, encoding="utf-8-sig")

    # ── 表 1：现状（dense 单独）vs 上限（并集）vs 实际（RRF）──
    for qn in ("Q_dev", "Q_prod"):
        q = d[d.qset == qn]
        print("\n" + "=" * 122)
        print(f"【表 1｜口径 {qn}】"
              f"{'（中文题面 + 3 条英文子查询）' if qn == 'Q_dev' else '（中文题面 + 1 条英文检索式 ≈ 生产 L6）'}"
              f" ｜ {q.groupby(['cluster', 'facet']).ngroups} 题 ｜ 均 gold {q.n_gold.mean():.2f}")
        print(f"  {'k':>3}{'k/库':>7}{'dense':>8}{'sparse':>8}{'bm25':>8}"
              f" ｜{'随机':>7}{'dense超随机':>12}"
              f" ｜{'并集覆盖':>10}{'k限上限':>9}{'RRF实际':>9}{'差':>8}"
              f" ｜{'dense篇F1':>11}{'上限篇F1':>10}{'并集项数':>9}")
        for k in KS:
            t = q[q.k == k]
            rnd = t.random_StRecall.mean()
            dns = t.dense_StRecall.mean()
            print(f"  {k:>3}{t.k_ratio.mean():>7.2f}"
                  f"{dns:>8.3f}{t.sparse_StRecall.mean():>8.3f}"
                  f"{t.bm25_StRecall.mean():>8.3f} ｜{rnd:>7.3f}{dns - rnd:>+12.3f}"
                  f" ｜{t.union_cov.mean():>10.3f}"
                  f"{t.boracle_StRecall.mean():>9.3f}{t.rrf_StRecall.mean():>9.3f}"
                  f"{(t.boracle_StRecall - t.rrf_StRecall).mean():>+8.3f}"
                  f" ｜{t.dense_F1.mean():>11.3f}{t.boracle_F1.mean():>10.3f}"
                  f"{t.union_n.mean():>9.1f}")
        kr = 14 if 14 in KS else KS[-2]
        print(f"  【MRecall（全有或全无）】k={kr}：dense {q[q.k == kr].dense_MRecall.mean():.3f}"
              f" ｜ k限上限 {q[q.k == kr].boracle_MRecall.mean():.3f}"
              f" ｜ RRF {q[q.k == kr].rrf_MRecall.mean():.3f}")
        print(f"  【读法】3 个单通道列 = **各自独立 top-k 的 StRecall**；"
              f"`随机` = 随机交付 k 篇的期望召回（= k/库，**下界**）；"
              f"`dense超随机` = dense − 随机 = **真实能力**；"
              f"`并集覆盖` = 三通道 top-k 并集（不截断）覆盖的 gold；"
              f"`k限上限` = 并集里挑最优 k 个（**任何 k 限交付的召回上界**）；`差` = 上限 − RRF = **融合损失**")

    # ── 表 2：独占 gold ──
    print("\n" + "=" * 122)
    print("【表 2｜各通道的**边际贡献**（k=10，只有该通道召回到的 gold 篇）】")
    print(f"  {'口径':<8}{'通道':<9}{'独占 gold 合计':>15}{'涉及题数':>9}{'占总 gold':>10}")
    tot = d[d.k == 10].groupby("qset").n_gold.sum()
    for qn in ("Q_dev", "Q_prod"):
        q = e[e.qset == qn]
        for n in ("dense", "sparse", "bm25"):
            s = q[q.channel == n]
            print(f"  {qn:<8}{n:<9}{int(s.n_exclusive.sum()):>15}{len(s):>9}"
                  f"{s.n_exclusive.sum() / max(tot.get(qn, 1), 1):>10.1%}")
    print(f"  说明：`sparse`/`bm25` 的独占数为 0 时，说明**它们的召回到的 gold 都被 dense 覆盖了** "
          f"→ 该通道对召回无边际贡献（对**排序**仍可能有）。")

    # ── 表 3：证据召回（等字符预算）──
    print("\n" + "=" * 122)
    print(f"【表 3｜证据召回】（每篇 {CHAR_BUDGET} 字符预算；目标集 = **锚点命中块**代理，继承锚点偏差）")
    print(f"  {'口径':<8}{'通道':<9}{'块数':>7}{'实选字符/篇':>12}{'证据召回':>10}")
    for qn in ("Q_dev", "Q_prod"):
        q = v[v.qset == qn]
        for n in ("dense", "sparse", "bm25", "union3"):
            s = q[q.channel == n]
            print(f"  {qn:<8}{n:<9}{s.n_chunks.mean():>7.0f}{s.chars_per_doc.mean():>12,.0f}"
                  f"{s.ev_recall.mean():>10.3f}")
    print(f"\n  → 已写 results/R2_RECALL{suffix}_{{CEIL,EXCL,EV}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
