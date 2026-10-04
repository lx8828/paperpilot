"""**交付体积 + 召回指标的退化检查**（回应"k=14 体积比多少 / 召回真有这么好吗 / 只用 dense 够不够"）

## 要回答的三件事
1. **体积比**：k=5/10/14/20 各交付多少（篇数 / 块数 / 字符 / ≈token / 占全库比例 / S/C）
   —— 两种口径：`全文`（整篇塞进上下文）与 `reader`（语义 top-b + 锚点 top-a 块）
2. **召回指标是否退化**：每簇只有 **13~18 篇** → `k/库大小` 是关键；
   **k≥14 时"检索"几乎等于"交全库"** → StRecall 天然接近 1，不是能力
3. **小 k 下只 dense 够不够**：k=3/5/8/10 逐通道对比（回归的**真实区分区间**）

零 LLM。用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_volume.py
"""
from __future__ import annotations

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

f2spec = _iu.spec_from_file_location("facets2", HERE / "tmp" / "_r2_facets_v2.py")
_f2 = _iu.module_from_spec(f2spec)
sys.modules["facets2"] = _f2
f2spec.loader.exec_module(_f2)
F2 = _f2.F2

bspec = _iu.spec_from_file_location("bgem3", HERE / "tmp" / "_bge_m3.py")
_b3 = _iu.module_from_spec(bspec)
sys.modules["bgem3"] = _b3
bspec.loader.exec_module(_b3)

KS = [3, 5, 10, 14, 20]
CHARS_PER_TOK = 4.94          # 与 `R2_SC_VERDICT` 同口径（810k tok / 4M chars）
CTX = 128_000                 # 假定上下文预算（S/C 用）


def main() -> int:
    meta = json.loads((DEV / "clusters" / "meta.json").read_text(encoding="utf-8"))
    subq = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    gg = pd.read_csv(DEV / "gold_final2.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json").read_text(encoding="utf-8"))["per_combo"]}

    m = _b3.BGEM3()
    vol, ch_rows, lib = [], [], []
    for ci in range(3):
        ch = pd.read_parquet(DEV / "prodchunk" / "mineru" / f"c{ci}.parquet")
        ls = pd.read_parquet(DEV / "clusters" / f"c{ci}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"])]
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        lens = np.array([len(c) for c in chunks], dtype=np.int64)
        idx = {d: np.where(owner == d)[0] for d in docs}
        E = m.encode(chunks, batch=4)
        lib.append(dict(cluster=ci + 1, n_docs=len(docs), n_chunks=len(chunks),
                        chars=int(lens.sum()), med_chunk=int(np.median(lens)),
                        chars_per_doc=float(lens.sum() / len(docs))))
        print(f"  【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块 ｜ 全库 {lens.sum():,} 字符"
              f" ｜ 均 {lens.sum() / len(docs):,.0f} 字符/篇 ｜ 块长中位 {int(np.median(lens))}", flush=True)

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            _, zh, anchor, para = F2[facet]
            qs = [zh] + list(subq.get(facet, []))
            QE = m.encode(qs, batch=4)
            S = (QE["dense"] @ E["dense"].T).max(axis=0)
            pm = {d: float(S[idx[d]].max()) for d in docs}
            ranked = sorted(docs, key=lambda d: -pm[d])

            # reader 口径的每篇块（语义 top-b + 锚点 top-a，与 `_r2_reader` 同）
            import re as _re
            a_hit = np.array([bool(_re.search(F2[facet][0], c, _re.I)) for c in chunks])
            rch = {}
            for d in docs:
                sem = [int(j) for j in idx[d][np.argsort(-S[idx[d]])[:6]]]
                ah = sorted((int(j) for j in idx[d][a_hit[idx[d]]]), key=lambda j: -S[j])[:3]
                for j in ah:
                    if j not in sem:
                        sem.append(j)
                rch[d] = sem

            for k in KS:
                dk = ranked[:k]
                txt = int(sum(lens[idx[d]].sum() for d in dk))
                rchars = int(sum(lens[np.array(rch[d])].sum() for d in dk if rch[d]))
                nblk = int(sum(len(rch[d]) for d in dk))
                t = len(set(dk) & gold) / len(gold) if gold else float("nan")
                # ⚠️ `k/库` 用**实际交付篇数**（k 超过库大小时被截断 → 不虚高）
                nd = len(dk)
                vol.append(dict(cluster=ci + 1, facet=facet, k=k, n_gold=len(gold),
                                n_lib=len(docs), n_deliv=nd, k_over_lib=nd / len(docs),
                                lib_chars=int(lens.sum()),
                                full_chars=txt, full_kchars=txt / 1000,
                                frac_lib=txt / float(lens.sum()),
                                reader_chars=rchars, reader_kchars=rchars / 1000,
                                reader_blocks=nblk,
                                full_tok=int(txt / CHARS_PER_TOK),
                                reader_tok=int(rchars / CHARS_PER_TOK),
                                rand=min(nd / len(docs), 1.0)))
                ch_rows.append(dict(cluster=ci + 1, facet=facet, k=k, StRecall=t,
                                    n_deliv=nd, n_lib=len(docs),
                                    rand=min(nd / len(docs), 1.0)))
        print(f"     facets {len([1 for (c2, _x) in combos if c2 == ci + 1])} 完成", flush=True)

    v = pd.DataFrame(vol)
    c = pd.DataFrame(ch_rows)
    v.to_csv(HERE / "results" / "R2_VOLUME.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lib).to_csv(HERE / "results" / "R2_LIB_SIZE.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 122)
    print("【表 1｜交付体积】dense 排序下 k 篇的交付量（28 题均值）")
    print(f"  {'k':>3}{'k/库':>7}{'交付篇':>7} ｜{'全文 字符':>12}{'全文 k tok':>12}"
          f"{'占本簇库':>10} ｜{'reader 块数':>12}{'reader 字符':>12}{'reader tok':>11}{'S/C':>7}")
    for k in KS:
        t = v[v.k == k]
        print(f"  {k:>3}{t.k_over_lib.mean():>7.2f}{t.n_deliv.mean():>7.1f} ｜{t.full_chars.mean():>12,.0f}"
              f"{t.full_tok.mean() / 1000:>12.0f}{t.frac_lib.mean():>10.1%} ｜"
              f"{t.reader_blocks.mean():>12.0f}{t.reader_chars.mean():>12,.0f}"
              f"{t.reader_tok.mean() / 1000:>11.0f}{t.reader_tok.mean() / CTX:>7.2f}")
    print(f"  说明：`k=20` 时被库大小截断（每簇仅 13/16/18 篇）→ **实际交付 = 全部篇**"
          f"（故 k/库 的分子用实际篇数，不虚高）")

    print("\n" + "=" * 122)
    print("【表 2｜★ 召回指标的退化检查】库就这么大 → k 一大，「检索」就等于「交全库」")
    print(f"  每簇篇数：{ [x['n_docs'] for x in lib] }（均 {np.mean([x['n_docs'] for x in lib]):.1f}）")
    print(f"  {'k':>3}{'k/库':>7}{'交付篇':>7}{'StRecall(dense)':>17}{'随机交付同样多篇':>18}"
          f"{'超出随机':>10}  判读")
    for k in KS:
        t = c[c.k == k]
        d = t.StRecall.mean() - t.rand.mean()
        note = ("✅ 真实区分区间" if k <= 8 else
                "⚠️ 交付 ≥2/3 全库" if k <= 10 else "❌ **退化**（≈交全库）")
        print(f"  {k:>3}{(t.n_deliv / t.n_lib).mean():>7.2f}{t.n_deliv.mean():>7.1f}"
              f"{t.StRecall.mean():>17.3f}{t.rand.mean():>18.3f}{d:>+10.3f}   {note}")
    print("\n  读法：`随机交付同样多篇` = 交付篇数/库大小（**随机**能覆盖的 gold 比例，是该方法的下界基线）。")
    for k in (10, 14, 20):
        t = c[c.k == k]
        print(f"        k={k}：随机就有 **{t.rand.mean():.3f}**，dense {t.StRecall.mean():.3f}"
              f" → **只超出 {t.StRecall.mean() - t.rand.mean():+.3f}**")
    print("  → 库只有 13~18 篇，**k≥14 时交付的几乎是整个库** → 那些召回数字**不构成能力证据**。")

    print("\n" + "=" * 122)
    print("【表 3｜全库规模】用于 S/C 判断")
    for x in lib:
        print(f"  簇{x['cluster']}：{x['n_docs']} 篇 / {x['n_chunks']} 块 / {x['chars']:,} 字符"
              f"（≈{x['chars'] / CHARS_PER_TOK / 1000:.0f}k token）｜ 均 {x['chars_per_doc']:,.0f} 字符/篇")
    tc = sum(x["chars"] for x in lib)
    print(f"  合计：{sum(x['n_docs'] for x in lib)} 篇 / {tc:,} 字符 ≈ {tc / CHARS_PER_TOK / 1000:.0f}k token"
          f" ｜ **S/C = {tc / CHARS_PER_TOK / CTX:.2f}**（vs 128k）")
    print(f"\n  → 已写 results/R2_VOLUME.csv + R2_LIB_SIZE.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
