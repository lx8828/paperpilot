"""**Step 3｜换切块后的检索评测**：三臂同题对照（隔离"文本源"与"切块"两个变量）

| 臂 | 文本源 | 切块 | 目的 |
|---|---|---|---|
| `A_ls_fix` | LitSearch 纯文本 | 固定窗 1000/900 | 旧基线（在**同一批 47 篇**上重算） |
| `B_pdf_fix` | **PDF(pymupdf)** | 固定窗 1000/900 | 隔离"文本源"的影响 |
| `C_pdf_prod` | **PDF(pymupdf)** | **生产章节段落切块** | 隔离"切块"的影响（本次要看的） |

## 口径（与历史可比 + 修掉两处已知问题）
· 查询集 = **干净**（`[中文题面] + LLM 子查询`，`subqueries.json`）；排序臂 = `A base`（单查询）/ `B mq_max`（多查询取 max）
· 打分 = dense（bge-m3），**`max_seq_length=8192`**（生产值；旧脚手架用 512 是截断坑）
· 篇级分 = 篇内 **max**
· **真值统一用 PDF 全文的 facet 正则命中**（不随切块变）→ 三臂可比
· 指标：`MRecall@k`(JPR) / `StRecall@k`(Zhai) / 集合 `P` / 集合 `F1` / `α-nDCG`(Clarke) / **`ev_recall`（证据召回）**
  `ev_recall@b` = 选中的 top-(b×篇数) 块里覆盖了多少"词面命中块"

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_retr_eval.py --limit 4   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_retr_eval.py
"""
from __future__ import annotations

import argparse
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
GOLD2 = DEV / "gold_final2.csv"                 # ★ 新真值（三判官多数票）
SEL2 = DEV / "facets_v2_selected.json"          # ★ 新题集
KS = [5, 10, 20]
BS = [6, 12]                      # 每篇块预算（证据召回·块数口径）
CHAR_BS = [6000, 12000]           # 每篇**字符**预算（证据召回·跨切块公平口径）
CHUNK, OVERLAP = 1000, 100
ALPHA = 0.9

import importlib.util as _iu  # noqa: E402

fspec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2 = _iu.module_from_spec(fspec)
sys.modules["facets2"] = _f2
fspec.loader.exec_module(_f2)
F2 = _f2.F2

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F


def fixed_chunks(t: str) -> list[str]:
    """旧脚手架的固定窗切块（1000 字符 / 步长 900）。"""
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def mrecall(ranked, gold, k):
    m = len(gold)
    if m == 0:
        return float("nan")
    c = len(set(ranked[:k]) & gold)
    return 1.0 if (c == m if m <= k else c >= k) else 0.0


def strecall(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / len(gold) if gold else float("nan")


def setpf(ranked, gold, k):
    topk = set(ranked[:k])
    tp = len(topk & gold)
    p = tp / len(topk) if topk else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def andcg(ranked, gold, k, alpha=ALPHA):
    dcg, pfx = 0.0, 1.0
    for i, d in enumerate(ranked[:k], 1):
        g = 1.0 if d in gold else 0.0
        dcg += g / np.log2(i + 1) * pfx
        pfx *= (1 - alpha * g)
    idcg, pfx = 0.0, 1.0
    for i in range(1, min(k, len(gold)) + 1):
        idcg += 1.0 / np.log2(i + 1) * pfx
        pfx *= (1 - alpha)
    return dcg / idcg if idcg else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="A_ls_fix,B_pdf_fix,C_pdf_prod")
    ap.add_argument("--chunktag", default="mineru",
                    help="生产切块取自 `prodchunk/<tag>/`（mineru = 生产口径）")
    ap.add_argument("--gold", default="new", choices=["new", "anchor"],
                    help="真值口径：new = 三判官多数票 `gold_final2.csv`；"
                         "anchor = 旧的 `F[f][0]` 正则命中（副产物：真值来源）")
    ap.add_argument("--combos", default="auto", choices=["auto", "new", "anchor"],
                    help="题集与题面来源：new = `facets_v2_selected.json` 的 28 组合 + v2 题面；"
                         "anchor = `meta.usable` + v1 题面；auto = 跟随 `--gold`。"
                         "→ `--gold anchor --combos new` 可**只换真值**（隔离真值效应）")
    ap.add_argument("--corpus", default="20", choices=["20", "50"],
                    help="语料规模：20 = 原 3×20 篇（clusters/）；"
                         "50 = 扩语料 3×50 篇（corpus50/，47 原有 + 103 同领域干扰项）")
    ap.add_argument("--gold-file", default="",
                    help="真值 CSV（默认：20 → gold_final2.csv；50 → gold_final3.csv）")
    args = ap.parse_args()

    if args.corpus == "50":
        CDIR, TDIR, KDIR = DEV / "corpus50", DEV / "pdftext50", DEV / "prodchunk50"
        pmap = json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
    else:
        CDIR, TDIR, KDIR = DEV / "clusters", DEV / "pdftext", DEV / "prodchunk"
        pmap = json.loads((DEV / "pdf_map.json").read_text(encoding="utf-8"))
    GOLDF = DEV / (args.gold_file or
                   ("gold_final3.csv" if args.corpus == "50" else "gold_final2.csv"))

    arx = json.loads((DEV / "arxiv_map.json").read_text(encoding="utf-8"))
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    meta = json.loads((DEV / "clusters" / "meta.json").read_text(encoding="utf-8"))
    keep = {d for d, v in pmap.items() if v.get("ok")}
    arms = [a for a in args.arms.split(",") if a]
    gg = pd.read_csv(GOLDF)
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG_ALL = {(int(c), str(f)): set(x[x["gold"]]["docid"])
              for (c, f), x in gg.groupby(["cluster", "facet"])}
    NEW_COMBOS = {(int(k.split("|")[0]), k.split("|")[1])
                  for k in json.loads(SEL2.read_text(encoding="utf-8"))["per_combo"]}
    cbs = args.combos if args.combos != "auto" else args.gold
    if cbs == "new":
        combos, F_use = NEW_COMBOS, F2
    else:
        combos, F_use = None, F
    NG = NG_ALL if args.gold == "new" else None

    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192                     # ⚠️ 生产值（512 是截断坑）
    if dev == "cuda":
        enc.half()
    print("=" * 116)
    print(f"【R2 检索评测】设备 {dev} ｜ max_seq_length=8192 ｜ 真值口径 "
          f"**{args.gold}**{'（三判官多数票）' if args.gold == 'new' else '（facet 正则命中·旧）'}"
          f" ｜ 臂：{arms}")
    # ⚠️ `corpus50/pdf_map_all.json` 自带 `cluster`；旧的 `pdf_map.json` 没有 → 回落到 `arxiv_map`
    _cl_of = (lambda d: int(pmap[d]["cluster"])) if "cluster" in next(iter(pmap.values())) \
        else (lambda d: int(arx[d]["cluster"]))
    _cnt = [sum(1 for d in keep if _cl_of(d) == c) for c in (1, 2, 3)]
    print(f"  语料规模 **{args.corpus}**：{len(keep)} 篇（有 PDF）｜ 簇 {_cnt[0]}/{_cnt[1]}/{_cnt[2]} 篇")
    print(f"  切块 {KDIR.name}/{args.chunktag} ｜ 文本 {TDIR.name} ｜ 真值 {GOLDF.name}")
    if args.corpus == "50" and "A_ls_fix" in arms:
        print("  ⚠️ 50 篇语料里**干扰项没有 LitSearch 全文**（它们是 arxiv 新篇）→ "
              "`A_ls_fix` 对干扰项为空文本，**该臂不成立**，请看 PDF 臂。\n")
    else:
        print()

    rows = []
    for ci in range(len(meta)):
        # —— 文本源（两种）——
        txt_prod = pd.read_parquet(TDIR / f"c{ci}.parquet")
        chk_prod = pd.read_parquet(KDIR / args.chunktag / f"c{ci}.parquet")
        ls = pd.read_parquet(CDIR / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in keep]
        ls_txt = ({str(r["docid"]): str(r["full_paper"]) for _, r in ls.iterrows()}
                  if "full_paper" in ls.columns else {})
        pdf_txt = {str(r["docid"]): str(r["text"]) for _, r in txt_prod.iterrows()}

        # —— 真值 ——
        # `anchor`：PDF 全文上的 facet 正则命中（**不随切块变**，可做切块对照）
        # `new`  ★：三判官多数票（`gold_final2.csv`）—— 与锚点无关，是**独立真值**
        facets_here = (sorted(x for (c2, x) in combos if c2 == ci + 1) if combos
                       else meta[ci]["usable"])
        n_facet_all = len(facets_here)
        gold = {}
        if args.gold == "new":
            for f in facets_here:
                g2 = (NG.get((ci + 1, f)) or set()) & set(docs)
                if g2 and len(g2) < len(docs):
                    gold[f] = g2
        else:
            for f in facets_here:
                pat = re.compile(F_use[f][0], re.I)
                g = {d for d in docs if d in pdf_txt and pat.search(pdf_txt[d])}
                if g and len(g) < len(docs):
                    gold[f] = g
        print(f"    [语料] 簇{ci + 1}: {len(docs)} 篇有 PDF / {len(ls_txt)} 篇原语料"
              f" ｜ 生产块 {len(chk_prod)} ｜ facet {n_facet_all} → 真值 {len(gold)}"
              f" ｜ 真值 **{args.gold}** ｜ 题集/题面 **{args.combos if args.combos != 'auto' else args.gold}**"
              f" ｜ 均 gold {np.mean([len(x) for x in gold.values()]) if gold else 0:.2f}", flush=True)
        if not gold:
            continue

        # —— 三臂的块集合（(doc, chunk_text) 序列 + owner）——
        def build(arm: str):
            texts, owner, gchunk = [], [], {}
            if arm == "A_ls_fix":
                src = ls_txt
            else:
                src = pdf_txt
            for d in docs:
                t = src.get(d, "")
                cs = (fixed_chunks(t) if arm in ("A_ls_fix", "B_pdf_fix")
                      else None)
                if cs is None:
                    g = chk_prod[chk_prod.docid == d]
                    cs = g["text"].astype(str).tolist()
                texts += cs
                owner += [d] * len(cs)
            owner = np.array(owner)
            return texts, owner

        for arm in arms:
            texts, owner = build(arm)
            if not texts:
                continue
            t0 = time.time()
            C = enc.encode(texts, batch_size=16, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            low = [t.lower() for t in texts]
            idx = {d: np.where(owner == d)[0] for d in docs}

            for f, g in gold.items():
                qs = [F_use[f][1]] + list(subq.get(f, []))
                Q = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True
                               ).astype(np.float32)
                S = Q @ C.T
                Smax = S.max(axis=0)
                # ⚠️ `hit` 是**锚点正则的命中块**，只作"证据块"的代理参照（真值已不用锚点）
                hit = np.array([bool(re.search(F_use[f][0], c, re.I)) for c in low])
                # 排序臂
                pmA = {d: float(S[0][idx[d]].max()) for d in docs}
                pmB = {d: float(Smax[idx[d]].max()) for d in docs}
                for nm, pm in (("A base", pmA), ("B mq_max", pmB)):
                    ranked = sorted(docs, key=lambda d: -pm[d])
                    rec = dict(cluster=ci + 1, facet=f, arm=arm, sorter=nm,
                               n_gold=len(g), n_chunks=len(texts), n_docs=len(docs),
                               sec=round(time.time() - t0, 1))
                    for k in KS:
                        p, r, f1 = setpf(ranked, g, k)
                        rec.update({f"MRecall@{k}": mrecall(ranked, g, k),
                                    f"StRecall@{k}": strecall(ranked, g, k),
                                    f"setP@{k}": p, f"setF1@{k}": f1,
                                    f"aNDCG@{k}": andcg(ranked, g, k)})
                    # 证据召回①：全局 top-(b×篇数) 块 —— ⚠️ **块数预算**，块越大越占便宜
                    for b in BS:
                        B = min(b * len(docs), len(texts))
                        sel = np.argsort(-Smax)[:B]
                        rec[f"ev_recall@{b}"] = (float(hit[sel].sum() / max(hit.sum(), 1))
                                                 if hit.any() else float("nan"))
                    # 证据召回②：**等字符预算**（每篇累计到 C 字符）—— 跨切块公平
                    lens = np.array([len(t) for t in texts])
                    order_all = np.argsort(-Smax)
                    for CB in CHAR_BS:
                        seli: list[int] = []          # ⚠️ 不要叫 `keep`：会覆盖外层的 docid 集合
                        used = {d: 0 for d in docs}
                        for i in order_all:
                            d = owner[i]
                            if used[d] >= CB:
                                continue
                            seli.append(int(i))
                            used[d] += lens[i]
                        selc = np.array(seli)
                        rec[f"ev_recall_c{CB // 1000}k"] = (
                            float(hit[selc].sum() / max(hit.sum(), 1)) if hit.any()
                            else float("nan"))
                        rec[f"chars_c{CB // 1000}k"] = float(lens[selc].sum() / len(docs))
                    rows.append(rec)
            print(f"  簇{ci + 1} {arm:<11}{len(texts):>7} 块 / {len(docs):>3} 篇"
                  f" ｜ {len(gold)} facet ｜ {time.time() - t0:>5.0f}s", flush=True)

    d = pd.DataFrame(rows)
    outp = (HERE / "results" /
            f"R2_RETR_n{args.corpus}_g{args.gold}"
            f"_c{args.combos if args.combos != 'auto' else args.gold}"
            f"_{len(arms)}arms.csv")
    d.to_csv(outp, index=False, encoding="utf-8-sig")
    if args.limit:
        print(f"\n（--limit 冒烟，跳过汇总）")
        return 0

    m = d[d.sorter == "B mq_max"]
    print("\n" + "=" * 116)
    print(f"【主表｜多查询 `B mq_max`】臂对照（语料规模 **{args.corpus}**"
          f"＝{len(keep)} 篇 ｜ 同真值 ｜ 同查询集）")
    cols = [("MRecall@5", "MRecall@5"), ("MRecall@10", "MRecall@10"),
            ("StRecall@5", "StRecall@5"), ("StRecall@10", "StRecall@10"),
            ("setP@10", "集合P@10"), ("setF1@10", "集合F1@10"), ("aNDCG@10", "α-nDCG@10")]
    print(f"  {'臂':<13}{'块数':>7}" + "".join(f"{lab:>12}" for _, lab in cols))
    for arm in arms:
        s = m[m.arm == arm]
        if not len(s):
            continue
        print(f"  {arm:<13}{s['n_chunks'].mean():>7.0f}"
              + "".join(f"{s[c].mean():>12.3f}" for c, _ in cols))

    print(f"\n  【证据召回】⚠️ 块数口径**不公平**（块越大越占便宜）→ 以**等字符预算**为准")
    print(f"  {'臂':<13}{'@6块':>9}{'@12块':>9}   {'@6k字符':>10}{'@12k字符':>11}"
          f"{'实选字符/篇@12k':>17}")
    for arm in arms:
        s = m[m.arm == arm]
        if not len(s):
            continue
        print(f"  {arm:<13}{s['ev_recall@6'].mean():>9.3f}{s['ev_recall@12'].mean():>9.3f}   "
              f"{s['ev_recall_c6k'].mean():>10.3f}{s['ev_recall_c12k'].mean():>11.3f}"
              f"{s['chars_c12k'].mean():>17,.0f}")
    print(f"\n  {'对比（相对旧基线 A_ls_fix）':<30}{'Δ':>9}")
    for arm in ("B_pdf_fix", "C_pdf_prod"):
        s, a = m[m.arm == arm], m[m.arm == "A_ls_fix"]
        if not len(s) or not len(a):
            continue
        for c, lab in (("StRecall@10", "StRecall@10"), ("setF1@10", "集合F1@10"),
                       ("ev_recall@12", "证据召回@12")):
            print(f"    {arm} 的 {lab:<22}{s[c].mean() - a[c].mean():>+9.3f}")

    print(f"\n  【单查询 `A base`】StRecall@10 / 集合F1@10 / 证据召回@12")
    a0 = d[d.sorter == "A base"]
    for arm in arms:
        s = a0[a0.arm == arm]
        if len(s):
            print(f"    {arm:<13}{s['StRecall@10'].mean():>8.3f}"
                  f"{s['setF1@10'].mean():>12.3f}{s['ev_recall@12'].mean():>14.3f}")

    print(f"\n  【块粒度】")
    for arm in arms:
        s = m[m.arm == arm]
        if len(s):
            print(f"    {arm:<13}块数 {s['n_chunks'].mean():>7.0f}"
                  f" ｜ 均 gold {s['n_gold'].mean():>5.2f}"
                  f" ｜ facet 数 {s.groupby(['cluster', 'facet']).ngroups}")
    print(f"\n已写 {outp}")
    _emit_report(arms, m, keep, KDIR, GOLDF, outp)
    return 0


def _emit_report(arms, m, keep, KDIR, GOLDF, outp) -> None:
    """把**主表**（`sorter == "B mq_max"`）的汇总落进统一记录格式。

    为什么：本脚本原先只写**逐簇 CSV**，汇总只 print —— 无法与 L1/L3 比、
    也无法与历史比（要翻 stdout 考古）。

    ⚠️ **失败绝不影响评测**；且**算不出（NaN）就不 emit 那条**，不写 NaN
    （`evals/report.py` 会拒收 NaN —— 那类值在表里会变成"NaN"，比大小永远为假）。
    """
    try:
        sys.path.insert(0, str(HERE.parent))
        from evals import report as R  # noqa: PLC0415

        # ★ 口径必须写全：换真值文件 / 换切块 / 换题集，数字都不可比。
        note = (f"R2 证据召回（主表口径 sorter=`B mq_max`）｜ 语料 **{args.corpus}**"
                f"＝{len(keep)} 篇 ｜ 真值 `{GOLDF.name}` ｜ 切块 "
                f"`{KDIR.parent.name}/{KDIR.name}` ｜ 题集 combos={args.combos}"
                f" ｜ ★ 块口径**不公平**（块越大越占便宜）→ 以 "
                f"`ev_recall_c12k`（等字符预算）为准")
        # 指标 -> 列名。`MRecall/StRecall/aNDCG` 来自 `_r2_std_metrics`（文献口径）。
        COLS = (("MRecall@10", "r2.mrecall@10"), ("StRecall@10", "r2.strecall@10"),
                ("setF1@10", "r2.setf1@10"), ("aNDCG@10", "r2.andcg@10"),
                ("ev_recall@12", "r2.ev_recall@12"),
                ("ev_recall_c12k", "r2.ev_recall_c12k"))
        recs: list[dict] = []
        for arm in arms:
            s = m[m.arm == arm]
            if not len(s):
                continue
            for col, metric in COLS:
                if col not in s.columns:
                    continue
                v = float(s[col].mean())
                if not np.isfinite(v):
                    continue      # ★ 算不出来就别 emit（不要写 NaN）
                recs.append(dict(layer="L2", name=f"r2_{arm}", metric=metric,
                                 value=v, n=int(len(s)), note=note,
                                 evidence_path=str(outp)))
        if recs:
            p = R.emit(*recs)
            print(f"→ 汇总记录已落 {p.relative_to(HERE.parent)}（{len(recs)} 条）")
    except Exception as e:  # noqa: BLE001  报告失败不能弄挂评测
        print(f"⚠️ 汇总记录落盘失败（不影响本次评测结果）：{type(e).__name__}: {e}",
              file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
