"""**facet / 子查询体检**（50 篇口径）——把"查询质量"变成可量化的一等变量

## 为什么先做这个
召回网格（每路截断深度 d × 权重 ρ × 段级聚合）是在"查询好不好"的基础上调参。
若不先知道 **哪条子查询有用 / 哪条冗余 / 哪条贴该簇 / 哪条泄露真值**，
网格出来的最优权重就是在拟合噪声。

## 三段体检
### ① 子查询**静态**质量（零模型，纯文本）
· 覆盖：哪些 facet 没子查询
· 语言纯度：是否混入中文（BM25 会恒 0）
· **互补性**：3 条之间的两两 Jaccard —— 是否真"措辞不同"
· **泄露指标**：子查询 ↔ `anchor` / `pat`（gold 正则）的词汇重合率
  （参考：第一版用 `anchor` 做查询时重合 **63%** → 判泄露；这里量化"子查询是否干净"）

### ② 子查询的**检索贡献**（逐条）
· 每条子查询**单独**走 dense / BM25 的 StRecall@k
· **独占 gold**：只有这条子查询召回到的 gold 数 → 冗余 / 主力一眼可见
· 与 `zh-dense` 的互补：相对中文题面额外贡献了多少

### ③ 逐 facet × 逐路诊断 + **自动分型**
· 每 facet：各路的 StRecall、`bm25最强路 vs dense最强路` 谁赢、锚点命中率
· **自动分型**：`词面型`（BM25 ≥ dense，或锚点命中率高）/ `语义型`（dense ≫ bm25）
  → 为"分型权重"提供依据（而不是全局一个 ρ）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_subq_diag.py --corpus 50
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import re
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
KS = [5, 10, 13, 20]
STOP = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "with", "via", "using",
        "towards", "toward", "from", "by", "at", "as", "is", "are", "be", "we", "our",
        "this", "that", "it", "can", "not", "but", "or", "its", "their", "than", "then"}


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{1,}", str(s).lower()) if t not in STOP}


def jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50")
    ap.add_argument("--chunktag", default="mineru")
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
    in_set = sorted({f for _, f in combos})          # 题集里出现的 facet

    # ══════════ ① 子查询静态质量（零模型）══════════
    print("=" * 126)
    print(f"【① 子查询静态质量】`subqueries.json` 覆盖 {len(subq)} 个 facet ｜ "
          f"题集涉及 {len(in_set)} 个")
    miss = [f for f in in_set if f not in subq]
    print(f"  ⚠️ 题集里**没有子查询**的 facet（{len(miss)}）：{miss}")
    rows = []
    for f in in_set:
        subs = subq.get(f, [])
        pat = str(F2[f][0])
        anc = str(F2[f][2])
        pt, at, zt = toks(pat), toks(anc), toks(str(F2[f][1]))
        for i, s in enumerate(subs):
            st = toks(s)
            rows.append(dict(facet=f, idx=i + 1, subq=s, n_tok=len(st),
                             has_cjk=int(any("\u4e00" <= ch <= "\u9fff" for ch in s)),
                             jac_pat=jac(st, pt), jac_anchor=jac(st, at),
                             overlap_pat=len(st & pt) / max(len(pt), 1),
                             n_share_pat=len(st & pt), n_pat=len(pt)))
    S = pd.DataFrame(rows)
    print(f"\n  {'facet':<18}{'条数':>5}{'含中文':>7}{'token中位':>10}"
          f"{'↔anchor Jaccard':>17}{'↔gold正则 重合率':>17}{'共享词数':>9}")
    for f in in_set:
        t = S[S.facet == f]
        if not len(t):
            print(f"  {f:<18}{'0':>5}{'—':>7}{'—':>10}{'—':>17}{'—':>17}{'—':>9}")
            continue
        print(f"  {f:<18}{len(t):>5}{int(t.has_cjk.sum()):>7}{int(t.n_tok.median()):>10}"
              f"{t.jac_anchor.mean():>17.3f}{t.overlap_pat.mean():>17.1%}"
              f"{t.n_share_pat.mean():>9.1f}")
    print(f"\n  【读法】`↔gold正则 重合率` 是**泄露指标**：子查询里有多少比例的词直接来自真值正则。"
          f"\n          参考标定：第一版用 `anchor` 做查询时该值为 **63%**（判泄露）；"
          f"子查询应显著低于它（本来就是在给「消融实验→ablation」这种语言必然的重合）。")
    # 互补性（同 facet 内两两 Jaccard）
    print(f"\n  【互补性】同一 facet 的 3 条子查询两两 Jaccard（越低越「措辞不同」）")
    comp = []
    for f in in_set:
        ss = [toks(x) for x in subq.get(f, [])]
        if len(ss) < 2:
            continue
        vs = [jac(ss[i], ss[j]) for i in range(len(ss)) for j in range(i + 1, len(ss))]
        comp.append(dict(facet=f, n=len(ss), jac_med=float(np.median(vs)),
                         jac_max=float(np.max(vs))))
    C = pd.DataFrame(comp)
    print(f"    全体：中位 {C.jac_med.mean():.3f} ｜ 最大 {C.jac_max.max():.3f}"
          f" ｜ 最高者：{C.loc[C.jac_max.idxmax(), 'facet']}")
    for _, r in C.sort_values("jac_max", ascending=False).head(6).iterrows():
        print(f"      {r['facet']:<18}中位 {r.jac_med:.3f} ｜ 最大 {r.jac_max:.3f}"
              + ("   ⚠️ 高度重合" if r.jac_max > 0.4 else ""))

    # ══════════ ②③ 逐条贡献 + 逐 facet 分型（需要编码）══════════
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    contrib, facet_rows = [], []
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
        print(f"\n  簇{ci + 1}：{len(docs)} 篇 / {len(chunks)} 块 ｜ {time.time() - t0:.0f}s",
              flush=True)
        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            N = len(docs)
            zh, para = str(F2[facet][1]), str(F2[facet][3])
            SQ = [str(q) for q in subq.get(facet, []) if str(q).strip()][:3]
            paths = [("zh-dense", zh, "dense"), ("para-dense", para, "dense"),
                     ("para-bm25", para, "bm25")]
            paths += [(f"sq{i + 1}-dense", q, "dense") for i, q in enumerate(SQ)]
            paths += [(f"sq{i + 1}-bm25", q, "bm25") for i, q in enumerate(SQ)]
            paths_kind = {nm: kind for nm, _q, kind in paths}
            tops: dict[str, set[str]] = {}
            for nm, q, kind in paths:
                if kind == "dense":
                    qv = enc.encode([q], normalize_embeddings=True,
                                    show_progress_bar=False,
                                    convert_to_numpy=True).astype(np.float32)
                    s = (qv @ E.T)[0].astype(np.float32)
                else:
                    s = bm.scores(q).astype(np.float32)
                dm = {d: float(s[didx[d]].max()) for d in docs}
                o = sorted(docs, key=lambda d: -dm[d])
                tops[nm] = set(o[:13])
            # 逐路**独占 gold**：`excl_all` = 本路 top-13 里、**其余所有路**都没召回到的 gold
            alln = list(tops)
            for nm in alln:
                others = set().union(*[tops[o] for o in alln if o != nm])
                tp = len(tops[nm] & gold)
                contrib.append(dict(
                    cluster=ci + 1, facet=facet, path=nm,
                    kind=dict(paths_kind).get(nm, "dense"),
                    n_gold=len(gold), n_docs=N, k=13,
                    StRecall=tp / len(gold), purity=tp / 13, random=13 / N,
                    excl_all=len(tops[nm] & (gold - others))))
            sqd = [f"sq{i + 1}-dense" for i in range(len(SQ))]
            sqb = [f"sq{i + 1}-bm25" for i in range(len(SQ))]
            for nm in sqd + sqb:
                others = set().union(*[tops[o] for o in sqd + sqb if o != nm]) if tops else set()
                tp = len(tops.get(nm, set()) & gold)
                facet_rows.append(dict(
                    cluster=ci + 1, facet=facet, path=nm,
                    win13=len(tops.get(nm, set()) & (gold - others)),
                    StRecall=tp / len(gold),
                    bm_best=max(len(tops[o] & gold) for o in sqb) / len(gold) if sqb else 0,
                    dense_best=max(len(tops[o] & gold) for o in sqd) / len(gold) if sqd else 0,
                    zh=len(tops["zh-dense"] & gold) / len(gold)))
    CT, FR = pd.DataFrame(contrib), pd.DataFrame(facet_rows)
    CT.to_csv(HERE / "results" / "R2_SUBQ_CONTRIB.csv", index=False, encoding="utf-8-sig")
    FR.to_csv(HERE / "results" / "R2_SUBQ_FACET.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 126)
    print("【② 逐条路的贡献】k=13（篇分 = 篇内 max）｜ `独占gold` = 本路 top-13 里其余**所有路**都没召回到的")
    print(f"  {'路':<14}{'种类':<7}{'StRecall':>10}{'纯度':>8}{'超随机':>9}{'独占gold合计':>13}{'占总gold':>10}")
    totg = CT[CT.path == CT.path.iloc[0]].n_gold.sum()
    for nm in CT.path.unique():
        t = CT[CT.path == nm]
        print(f"  {nm:<14}{t.kind.iloc[0]:<7}{t.StRecall.mean():>10.3f}"
              f"{t.purity.mean():>8.3f}{t.StRecall.mean() - t.random.mean():>+9.3f}"
              f"{int(t.excl_all.sum()):>13}{t.excl_all.sum() / max(totg, 1):>10.1%}")
    # 子查询的独占贡献（按 facet 汇总）
    sub = FR[FR.path.str.startswith("sq")]
    print(f"\n  【子查询的独占贡献】（相对其他子查询路；k=13）")
    print(f"    {'路':<14}{'独占 gold 合计':>15}{'有贡献的题数':>13}{'均 StRecall':>12}")
    for nm in sorted(sub.path.unique()):
        t = sub[sub.path == nm]
        print(f"    {nm:<14}{int(t.win13.sum()):>15}{int((t.win13 > 0).sum()):>13}"
              f"{t.StRecall.mean():>12.3f}")

    print("\n" + "=" * 126)
    print("【③ 逐 facet 分型】按「BM25 最强路 vs dense 最强路」与「锚点命中」自动分型")
    print(f"  {'簇':>3} {'facet':<18}{'dense最强':>10}{'bm25最强':>10}{'zh-dense':>10}"
          f"{'谁赢':>7}{'建议权重':>10}")
    F = FR[FR.k.isna()] if False else FR
    typ = []
    for (c, fa), g in F.groupby(["cluster", "facet"]):
        db = g.dense_best.mean()
        bb = g.bm_best.mean()
        zz = g.zh.mean()
        win = "dense" if db > bb + 0.02 else ("bm25" if bb > db + 0.02 else "平")
        sug = "ρ=0.25" if win == "dense" else ("ρ=1.0" if win == "bm25" else "ρ=0.5")
        typ.append(dict(cluster=c, facet=fa, dense_best=db, bm_best=bb, zh=zz,
                        win=win, sug=sug))
        print(f"  {int(c):>3} {fa:<18}{db:>10.3f}{bb:>10.3f}{zz:>10.3f}{win:>7}{sug:>10}")
    T = pd.DataFrame(typ)
    T.to_csv(HERE / "results" / "R2_FACET_TYPE.csv", index=False, encoding="utf-8-sig")
    print(f"\n  分型汇总：{T.win.value_counts().to_dict()}")
    print(f"\n  → 已写 results/R2_SUBQ_CONTRIB.csv ｜ R2_SUBQ_FACET.csv ｜ R2_FACET_TYPE.csv"
          f"（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
