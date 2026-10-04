"""**标准指标 v2（新口径）** —— 阶段 1 的 **A3'（等字符预算回灌）+ A4（交付 k 曲线）**

## 为什么不直接改 `_r2_std_metrics.py`
该文件被 5 个脚本 `import` 当**库**用（`F` / `mrecall` / `strecall` / `setpf` / `BM25` / `chunks_of`）：
`_r2_retr_eval` / `_r2_reader` / `_r2_gold_recalib` / `_r2_facet_audit` / `_r2_cr_diag`。
在其上动刀会波及全部 → **新建 v2 承接新口径，库保持不动**（v2 从 `stdm` 复用指标函数与 BM25）。

## 新口径 vs 旧口径
| | 旧（`_r2_std_metrics.py`） | **新（本脚本默认）** |
|---|---|---|
| 切块 | 固定窗 1000/900 over **LitSearch 全文** | **生产章节段落（MinerU）** |
| 真值 | `F[facet][0]` 词面锚点 | **三判官多数票（`gold_final2.csv`）** |
| 题集 | `meta.usable`（35） | **`facets_v2_selected.json`（28）** |
| 题面/锚点 | `F`（v1 简述） | **`F2`（操作化）** |
| 编码 | `max_seq_length=512`（截断坑） | **8192** |
| 预算口径 | **块数**（b × 篇数） | **块数 **∪** 字符**（A3'） |

## A3'：等字符预算
同一"块预算 b"在新旧块下**不等价**（块长 ×1.8）→ 本次把 L2/L3 全部改为**双口径**：
`BS`（每篇块数）与 `CHAR_BS`（每篇字符数），后者跨切块公平。
两类策略 × 两种口径都给：`global_topB` / `round_robin`(每篇等块) / `global_topC` / `per_doc_topC`。

## A4：交付 k 曲线
`--k` 默认 `1,3,6,10,14,20` → 看 MRecall / StRecall / 集合P / 集合F1 / α-nDCG 随 k 的变化，定甜点。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics_v2.py
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics_v2.py --legacy     # 旧口径对照
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics_v2.py --k 1,3,6,10,14,20
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

DEV = HERE / "data" / "r2dev"
CACHE = DEV / "clusters"
GOLD2 = DEV / "gold_final2.csv"
SEL2 = DEV / "facets_v2_selected.json"
CHUNKDIR = DEV / "prodchunk" / "mineru"
CHAR_BS = [1600, 8000, 16000]          # 每篇**字符**预算（A3'）；全局口径 = × 篇数

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf, alpha_ndcg, BM25, chunks_of = (
    _m.mrecall, _m.strecall, _m.setpf, _m.alpha_ndcg, _m.BM25, _m.chunks_of)

f2spec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2 = _iu.module_from_spec(f2spec)
sys.modules["facets2"] = _f2
f2spec.loader.exec_module(_f2)
F2 = _f2.F2

SCORERS = {                     # 名 → (dense 查询字段 | None=bm25, 说明)
    "dense_zh": ("zh", "dense·中文题面"),
    "dense_zh_en": ("zh_en", "dense·中文题面+英文词"),
    "dense_en": ("en", "dense·纯英文（同源）"),
    "bm25_para": (None, "BM25·改写句（非同源）"),
    "bm25_anchor": (None, "BM25·锚点原词（同源⚠️）"),
}


def sel_by_char(order: list[int], lens: np.ndarray, budget: int,
                owner=None, per_doc: bool = False, used: dict | None = None) -> list[int]:
    """按分数序取块，累计字符 ≤ budget（**跳过**装不下的大块，继续试后面的小块）。"""
    out: list[int] = []
    acc = 0
    u = used if used is not None else {}
    for i in order:
        cap = (u.get(owner[i], 0) + lens[i]) if per_doc else (acc + lens[i])
        if cap > budget:
            continue
        out.append(int(i))
        if per_doc:
            u[owner[i]] = u.get(owner[i], 0) + int(lens[i])
        else:
            acc += int(lens[i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true",
                    help="旧口径（锚点真值 + meta.usable + v1 题面 + 固定窗 + 512）")
    ap.add_argument("--k", default="1,3,6,10,14,20", help="A4：交付 k 曲线")
    ap.add_argument("--bs", default="1,3,6,12", help="每篇**块**预算")
    ap.add_argument("--char-bs", default=",".join(str(x) for x in CHAR_BS),
                    help="A3'：每篇**字符**预算")
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--workers", type=int, default=0, help="保留位（当前单线程）")
    args = ap.parse_args()

    KS = [int(x) for x in args.k.split(",") if x]
    BS = [int(x) for x in args.bs.split(",") if x]
    CBS = [int(x) for x in args.char_bs.split(",") if x]
    use_prod = not args.legacy
    F_use = F2 if use_prod else F
    CDIR = DEV / "prodchunk" / args.chunktag

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    if use_prod:
        gg = pd.read_csv(GOLD2)
        gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
        NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
              for (c, f), x in gg.groupby(["cluster", "facet"])}
        combos = {(int(kk.split("|")[0]), kk.split("|")[1])
                  for kk in json.loads(SEL2.read_text(encoding="utf-8"))["per_combo"]}
    else:
        NG, combos = None, None

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192 if use_prod else 512
    if dev == "cuda":
        enc.half()
    print("=" * 122)
    print(f"【标准指标 v2】真值 **{'三判官多数票' if use_prod else '词面锚点（旧）'}**"
          f" ｜ 切块 **{'生产章节段落(MinerU)' if use_prod else '固定窗 1000/900'}**"
          f" ｜ max_seq_length {enc.max_seq_length} ｜ k={KS}")
    print(f"  A3' 每篇字符预算 {CBS} ｜ 每篇块预算 {BS} ｜ 设备 {dev}\n")

    lvl1, lvl2, lvl3 = [], [], []
    for ci in range(len(meta)):
        ls = pd.read_parquet(CACHE / f"c{ci}.parquet")
        if use_prod:
            ch = pd.read_parquet(CDIR / f"c{ci}.parquet")
            docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"])]
            chunks, owner = [], []
            for d in docs:
                cs = ch[ch.docid == d]["text"].astype(str).tolist()
                chunks += cs
                owner += [d] * len(cs)
        else:
            docs = ls["docid"].tolist()
            chunks, owner = [], []
            for i, t in enumerate(ls["full_paper"].tolist()):
                cs = chunks_of(t)
                chunks += cs
                owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        lens = np.array([len(c) for c in chunks], dtype=np.int64)
        full_txt = {str(r["docid"]): str(r["full_paper"]) for _, r in ls.iterrows()}
        C = enc.encode(chunks, batch_size=16 if use_prod else 32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = BM25(chunks)
        low = [c.lower() for c in chunks]
        idx = {d: np.where(owner == d)[0] for d in docs}
        facets_here = (sorted(x for (c2, x) in combos if c2 == ci + 1) if combos
                       else meta[ci]["usable"])
        print(f"  【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块 / {len(facets_here)} facet"
              f" ｜ 块长中位 {int(np.median(lens))} 字符 / 篇块数中位 "
              f"{int(np.median([len(idx[d]) for d in docs]))}", flush=True)

        for facet in facets_here:
            if use_prod:
                gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            else:
                pat = re.compile(F[facet][0], re.I)
                gold = {d for d in docs if pat.search(full_txt.get(d, ""))}
            if not gold or len(gold) == len(docs):
                continue
            _, zh, anchor, para = F_use[facet]
            qv = {nm: enc.encode([q], normalize_embeddings=True, convert_to_numpy=True
                                 )[0].astype(np.float32)
                  for nm, q in (("zh", zh), ("zh_en", f"{zh} {anchor}"), ("en", anchor))}
            dsc = {nm: C @ v for nm, v in qv.items()}
            bsc = {"bm25_para": bm.scores(para), "bm25_anchor": bm.scores(anchor)}

            # ⚠️ `hit` = **锚点命中块**，只作"证据块"的代理参照（真值已不用锚点）
            apat = re.compile(F_use[facet][0], re.I)
            hit = np.array([bool(apat.search(c)) for c in low])
            n_hit = int(hit.sum())

            def ev(sel) -> float:
                return float(hit[sel].sum() / n_hit) if n_hit else float("nan")

            # ── L1：篇级交付 k 篇（A4 的 k 曲线）──
            def add_l1(nm: str, ranked: list[str], note: str = "") -> None:
                for k in KS:
                    p, r, f1 = setpf(ranked, gold, k)
                    lvl1.append(dict(cluster=ci + 1, facet=facet, scorer=nm, k=k,
                                     n_gold=len(gold), MRecall=mrecall(ranked, gold, k),
                                     StRecall=strecall(ranked, gold, k), setP=p, setR=r,
                                     setF1=f1, aNDCG=alpha_ndcg(ranked, gold, k), note=note))

            for nm, (which, _d) in SCORERS.items():
                s = dsc[which] if which else bsc[nm]
                pm = {d: float(s[idx[d]].max()) for d in docs}
                add_l1(nm, sorted(docs, key=lambda x: -pm[x]))
            rng = np.random.default_rng(0)
            perms = [rng.permutation(docs).tolist() for _ in range(200)]
            for k in KS:
                lvl1.append(dict(cluster=ci + 1, facet=facet, scorer="random", k=k,
                                 n_gold=len(gold),
                                 MRecall=float(np.mean([mrecall(p_, gold, k) for p_ in perms])),
                                 StRecall=float(np.mean([strecall(p_, gold, k) for p_ in perms])),
                                 setP=float(np.mean([setpf(p_, gold, k)[0] for p_ in perms])),
                                 setR=float(np.mean([setpf(p_, gold, k)[1] for p_ in perms])),
                                 setF1=float(np.mean([setpf(p_, gold, k)[2] for p_ in perms])),
                                 aNDCG=float(np.mean([alpha_ndcg(p_, gold, k) for p_ in perms])),
                                 note="200 次随机排列"))
            add_l1("oracle", sorted(docs, key=lambda x: (x not in gold, x)), "完美排序（上界）")

            # ── L2：块预算（等块 vs 等字符）×（全局 vs 每篇）──  A3'
            s2 = dsc["zh_en"]
            gorder = [int(i) for i in np.argsort(-s2)]
            cases: list[tuple[str, int, str, list[int]]] = []
            for b in BS:
                B = min(b * len(docs), len(chunks))
                cases.append(("块/篇", b, "global_topB", gorder[:B]))
                per_d = {d: list(idx[d][np.argsort(-s2[idx[d]])]) for d in docs}
                rr = [int(per_d[d][r]) for r in range(b) for d in docs if r < len(per_d[d])]
                cases.append(("块/篇", b, "round_robin", rr[:B]))
            for cb in CBS:
                cases.append(("字符/篇", cb, "global_topC",
                              sel_by_char(gorder, lens, cb * len(docs))))
                cases.append(("字符/篇", cb, "per_doc_topC",
                              sel_by_char(gorder, lens, cb, owner=owner, per_doc=True, used={})))
            for unit, bud, strat, sel in cases:
                if not sel:
                    continue
                used = list(dict.fromkeys(owner[np.array(sel)].tolist()))
                p, r, f1 = setpf(used, gold, len(used)) if used else (0.0, 0.0, 0.0)
                lvl2.append(dict(cluster=ci + 1, facet=facet, unit=unit, budget=bud,
                                 strategy=strat, n_chunks=len(sel),
                                 chars_total=int(lens[np.array(sel)].sum()),
                                 chars_per_doc=float(lens[np.array(sel)].sum() / len(docs)),
                                 n_used=len(used), covered=len(set(used) & gold),
                                 cover_rate=len(set(used) & gold) / len(gold),
                                 setP=p, setR=r, setF1=f1,
                                 aNDCG=alpha_ndcg(used, gold, len(used)), ev_recall=ev(sel)))

            # ── L3：ALCE 式引用指标（双口径 per-claim 预算）──
            s3 = dsc["zh_en"]
            pm3 = {d: float(s3[idx[d]].max()) for d in docs}
            thr = float(np.quantile([pm3[d] for d in docs], 1 - len(gold) / len(docs)))
            claims = [d for d in docs if pm3[d] >= thr]
            for unit, bud in ([("块/篇", b) for b in BS] + [("字符/篇", c) for c in CBS]):
                cr, cp = [], []
                for d in claims:
                    order = list(idx[d][np.argsort(-s3[idx[d]])])
                    if unit == "块/篇":
                        ii = np.array(order[:bud], dtype=int)
                    else:
                        ii = np.array(sel_by_char(order, lens, bud), dtype=int)
                    if not ii.size:
                        continue
                    sup = hit[ii]
                    cr.append(1.0 if sup.any() else 0.0)
                    cp.append(float(sup.mean()))
                lvl3.append(dict(cluster=ci + 1, facet=facet, unit=unit, budget=bud,
                                 n_claim=len(claims), n_gold=len(gold),
                                 claim_in_gold=len(set(claims) & gold) / max(1, len(claims)),
                                 cite_recall=float(np.mean(cr)) if cr else float("nan"),
                                 cite_precision=float(np.mean(cp)) if cp else float("nan")))

    tag = "anchor" if args.legacy else "new"
    d1, d2, d3 = pd.DataFrame(lvl1), pd.DataFrame(lvl2), pd.DataFrame(lvl3)
    for df, nm in ((d1, "L1_paper"), (d2, "L2_budget"), (d3, "L3_citation")):
        df.to_csv(HERE / "results" / f"R2_STD2_{tag}_{nm}.csv", index=False, encoding="utf-8-sig")

    nq = d1.groupby(["cluster", "facet"]).ngroups
    print(f"\n{'=' * 122}\n【L1｜篇级交付 k 篇】**A4：k 曲线**（{nq} 个真值 × {len(SCORERS)} 打分 × "
          f"{len(KS)} 个 k ｜ 均 gold {d1.n_gold.mean():.2f}）")
    print(f"  {'打分':<13}{'k':>3}{'MRecall':>9}{'StRecall':>10}{'集合P':>8}{'集合R':>8}"
          f"{'集合F1':>8}{'α-nDCG':>9}")
    for nm in list(SCORERS) + ["random", "oracle"]:
        sub = d1[d1.scorer == nm]
        for k in KS:
            t = sub[sub.k == k]
            if not len(t):
                continue
            print(f"  {nm:<13}{k:>3}{t.MRecall.mean():>9.3f}{t.StRecall.mean():>10.3f}"
                  f"{t.setP.mean():>8.3f}{t.setR.mean():>8.3f}{t.setF1.mean():>8.3f}"
                  f"{t.aNDCG.mean():>9.3f}")

    print(f"\n  【★ A4 判读：k 甜点】以 dense+中文题面+英文词（{SCORERS['dense_zh_en'][1]}）为例")
    print(f"  {'k':>3}{'k/gold':>8}{'MRecall':>9}{'StRecall':>10}{'集合P':>8}{'集合F1':>8}{'α-nDCG':>9}")
    for k in KS:
        t = d1[(d1.scorer == "dense_zh_en") & (d1.k == k)]
        print(f"  {k:>3}{k / d1.n_gold.mean():>8.2f}{t.MRecall.mean():>9.3f}"
              f"{t.StRecall.mean():>10.3f}{t.setP.mean():>8.3f}{t.setF1.mean():>8.3f}"
              f"{t.aNDCG.mean():>9.3f}")
    f1s = {k: d1[(d1.scorer == "dense_zh_en") & (d1.k == k)].setF1.mean() for k in KS}
    best_k = max(f1s, key=lambda x: f1s[x])
    print(f"  → **集合F1 峰值 k={best_k}（{f1s[best_k]:.3f}）** ｜ "
          f"对照 gold 均值 {d1.n_gold.mean():.2f} 篇")

    print(f"\n{'=' * 122}\n【L2｜块预算】**A3'：等块 vs 等字符**（打分 = dense zh_en）")
    print(f"  {'口径':<8}{'预算':>7}{'策略':<14}{'块数':>6}{'字符/篇':>9}{'涉及篇':>7}"
          f"{'覆盖篇':>7}{'覆盖率':>8}{'集合P':>8}{'集合F1':>8}{'α-nDCG':>9}{'证据召回':>9}")
    for unit in ("块/篇", "字符/篇"):
        buds = BS if unit == "块/篇" else CBS
        for bud in buds:
            for strat in (("global_topB", "round_robin") if unit == "块/篇"
                          else ("global_topC", "per_doc_topC")):
                t = d2[(d2.unit == unit) & (d2.budget == bud) & (d2.strategy == strat)]
                if not len(t):
                    continue
                print(f"  {unit:<8}{bud:>7}{strat:<14}{t.n_chunks.mean():>6.0f}"
                      f"{t.chars_per_doc.mean():>9,.0f}{t.n_used.mean():>7.1f}"
                      f"{t.covered.mean():>7.2f}{t.cover_rate.mean():>8.3f}"
                      f"{t.setP.mean():>8.3f}{t.setF1.mean():>8.3f}"
                      f"{t.aNDCG.mean():>9.3f}{t.ev_recall.mean():>9.3f}")

    print(f"\n  【★ A3' 判读：同一预算下表头互换的偏差】"
          f"（旧口径只报'块数'，但块长已 ×1.8 → 同一 b 的字符量完全不同）")
    for unit, bud in (("块/篇", 6), ("字符/篇", 8000)):
        t = d2[(d2.unit == unit) & (d2.budget == bud)]
        if len(t):
            print(f"    {unit} {bud}：实际字符/篇 "
                  f"{' / '.join(f'{s}={t[t.strategy == s].chars_per_doc.mean():,.0f}' for s in t.strategy.unique())}"
                  f" ｜ 集合F1 {' / '.join(f'{s}={t[t.strategy == s].setF1.mean():.3f}' for s in t.strategy.unique())}")

    print(f"\n{'=' * 122}\n【L3｜ALCE 式引用指标（检索侧代理：词面锚点代 NLI）】")
    print(f"  {'口径':<8}{'预算':>7}{'claim数':>9}{'claim在金标':>11}"
          f"{'cite_recall':>12}{'cite_precision':>15}")
    for unit in ("块/篇", "字符/篇"):
        buds = BS if unit == "块/篇" else CBS
        for bud in buds:
            t = d3[(d3.unit == unit) & (d3.budget == bud)]
            if len(t):
                print(f"  {unit:<8}{bud:>7}{t.n_claim.mean():>9.1f}{t.claim_in_gold.mean():>11.3f}"
                      f"{t.cite_recall.mean():>12.3f}{t.cite_precision.mean():>15.3f}")

    print(f"\n  → 已写 results/R2_STD2_{tag}_L{{1_paper,2_budget,3_citation}}.csv")
    print("""
读法：
  · MRecall@k 全有或全无（JPR）；StRecall@k 覆盖率（CoverageBench）；两者差 = 差几篇
  · **A3'**：看 `字符/篇` 列 —— 同一个"块预算"在新旧块下字符量不同，报数字必须带这个列
  · **A4**：看 L1 的 k 曲线；`k/gold` 接近 1 但 StRecall 未饱和 → 还在漏；P 开始掉 → 过交付
  · L2 的 `证据召回` 用**锚点命中块**作代理 → 继承锚点偏差，只宜作臂间相对比较
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
