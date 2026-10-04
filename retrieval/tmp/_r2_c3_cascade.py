r"""**C3 级联小样对照**（3 个 facet = 每簇 1 个，不全跑）

## 要回答什么
上一轮标定说：**NLI 判别力差（AUC 0.53），但做粗筛好（τ=0.1 砍 62% 块、gold 篇留存 99%）**。
本脚本把这句话变成**端到端实测**：级联是否保住判官质量，同时真的省下调用？

## 对照的两臂（**同一批候选块**、同一判官协议、同一真值口径）
| 臂 | 做法 | 调用数/题 |
|---|---|---|
| **① 全判** | 对**融合池 247 块**逐块问判官"这块支持该 facet 吗" | 247 |
| **② 级联** | `NLI τ 粗筛` → `regex 命中 = 零成本判正` → **只有剩余块问判官** | ~80 |

doc 级判定：**该篇有任何一块被判正 → 该篇为正**。
指标 = 与现有 `gold_final3.csv`（**doc 级** LLM 判官真值）的 setP/setR/setF1/StRecall。

## 与 gold 的口径差异（必须声明）
- gold 协议：**每篇 1 次调用**，输入是"（facet 正则命中块）∪（干净查询语义 top-k 块）"拼成的
  **~4,205 字符窗口**；**A+B 双判官 + 分歧第三轮**。
- 本脚本：**逐块**调用、**单判官**（`PAPERPILOT_LLM`）。
  → 两臂**内部可比**（完全同口径）；**与 gold 的差异**含"逐块 vs 窗口""单判官 vs 双判官"两种混杂，
  读与 gold 的差距时要知道这点。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_cascade.py --per-cluster 1 --tau 0.10
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_cascade.py --dry-run     # 只看候选/粗筛量，不调 LLM
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
CACHE = HERE / "results" / "nli_cache"
RRF_C = 60
D_ZH, D_SD, D_SB = 100, 50, 50
RHO = 4.0


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = _load("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")
SYS, call = gr.SYS, gr.call
_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2
LOCK = threading.Lock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cluster", type=int, default=1, help="每簇取几个 facet（小样用 1）")
    ap.add_argument("--tau", type=float, default=0.10, help="NLI 粗筛阈值")
    ap.add_argument("--nli-arm", default="nli_xnli", choices=["nli_xnli", "nli_fever"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pmap = json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
    pm_ok = {d for d, v in pmap.items() if v.get("ok")}
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}

    # ── 每簇选 1 个 facet：取 **gold 篇数中位** 的那个（代表性优于随手取首个）──
    per: dict[int, list[tuple[str, int]]] = {}
    for (c, f), g in NG.items():
        per.setdefault(c, []).append((f, len(g)))
    picks = []
    for c in (1, 2, 3):
        lst = sorted(per[c], key=lambda x: x[1])
        mid = lst[len(lst) // 2]
        picks += [(c, f) for f, _ in lst if f == mid[0]][: args.per_cluster]
    print("=" * 122)
    print(f"【C3 级联小样】{len(picks)} 个 facet ｜ NLI 臂 {args.nli_arm} τ={args.tau}"
          f" ｜ 判官 = 单 {('PAPERPILOT_LLM')} ｜ 融合 d=(100,50,50) ρ={RHO:.0f}")
    for c, f in picks:
        print(f"  簇{c} {f}：gold **{len(NG[(c, f)])}** 篇 / 50 篇")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    t0 = time.time()

    rows, lab_rows = [], []
    tot = {"pool": 0, "casc": 0, "nli_keep": 0, "regex": 0}
    for ci, facet in picks:
        c0 = ci - 1
        ch = pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{c0}.parquet")
        ls = pd.read_parquet(DEV / "corpus50" / f"c{c0}.parquet")
        docs = [d for d in ls["docid"].tolist()
                if d in set(ch["docid"]) and d in pm_ok]
        titles = {str(r["docid"]): str(r.get("title") or "") for _, r in ls.iterrows()}
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        nb = len(chunks)
        gold = (NG.get((ci, facet)) or set()) & set(docs)

        # ── 融合池（RRF，d=(100,50,50)，ρ=1:4）──
        E = enc.encode(chunks, batch_size=8, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)
        zh = str(F2[facet][1])
        sq = [str(q) for q in json.loads((DEV / "subqueries_v2.json")
                                         .read_text(encoding="utf-8")).get(facet, [])
              if str(q).strip()][:3] or [str(F2[facet][3])]

        def rd(q: str) -> np.ndarray:
            qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                            convert_to_numpy=True).astype(np.float32)
            r = np.empty(nb, dtype=np.int64)
            r[np.argsort(-(qv @ E.T)[0])] = np.arange(nb)
            return r

        def rb(q: str) -> np.ndarray:
            r = np.empty(nb, dtype=np.int64)
            r[np.argsort(-bm.scores(q).astype(np.float32))] = np.arange(nb)
            return r

        cls = {"zh-dense": [rd(zh)], "sq-dense": [rd(q) for q in sq],
               "sq-bm25": [rb(q) for q in sq]}
        dd = {"zh-dense": D_ZH, "sq-dense": D_SD, "sq-bm25": D_SB}
        sc = np.zeros(nb, dtype=np.float64)
        for c, rs in cls.items():
            w = RHO if c == "sq-bm25" else 1.0
            for r in rs:
                m = r < dd[c]
                sc[m] += w / (RRF_C + r[m] + 1)
        pool = np.where(sc > 0)[0]
        pool_docs = set(owner[pool])

        # ── NLI 粗筛（复用已缓存的块分）+ regex ──
        z = np.load(CACHE / f"{args.nli_arm}_c{c0}.npz")
        nli = z[facet].astype(np.float32)
        surv = pool[nli[pool] >= args.tau]
        rx = np.array([bool(_m.judge(chunks[j], facet)) for j in pool])
        A1 = pool[rx]                       # 幸存且正则命中 → 零成本判正
        A2 = pool[~rx & (nli[pool] >= args.tau)]   # 幸存但需 LLM
        B = pool[nli[pool] < args.tau]      # 被筛掉 → 级联视为否定
        tot["pool"] += len(pool)
        tot["casc"] += len(A2)
        tot["nli_keep"] += len(surv)
        tot["regex"] += len(A1)
        print(f"\n  簇{ci} {facet}：池 {len(pool)} 块 ｜ NLI τ={args.tau} 存活 {len(surv)}"
              f"（{len(surv) / len(pool):.1%}）｜ regex 判正 {len(A1)} ｜ 需 LLM {len(A2)}"
              f" ｜ 筛掉 {len(B)} ｜ 编码+检索 {time.time() - t0:.0f}s", flush=True)

        if args.dry_run:
            rows.append(dict(cluster=ci, facet=facet, pool=len(pool), surv=len(surv),
                             a1=len(A1), a2=len(A2), dropped=len(B),
                             n_gold=len(gold), pool_docs=len(pool_docs)))
            continue

        # ── 判官：臂① 全判（pool）；臂② 只判 A2 ──
        def judge1(j: int) -> tuple[int, str]:
            user = (f"论文标题：{titles.get(str(owner[j]), '')}\n\n论断：**{zh}**\n\n"
                    f"论文原文片段：\n{chunks[j]}")
            try:
                o = call("PAPERPILOT_LLM", SYS, user)
                return int(j), str(o.get("label") or "ERR").lower()
            except Exception:  # noqa: BLE001
                return int(j), "ERR"

        # ★ `A2 ⊂ pool` 且判官每次调用**独立同 prompt** → 只需判 pool 一次，
        #    臂② 的 A2 标签直接取子集（**省掉重复的 |A2| 次调用**）。
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            l1 = list(ex.map(judge1, pool.tolist()))
        y1 = {j: lab for j, lab in l1}
        a2set = set(A2.tolist())
        y2 = {j: lab for j, lab in y1.items() if j in a2set}
        set1 = {str(owner[j]) for j, lab in y1.items() if lab == "entail"}
        set2 = {str(owner[j]) for j in A1} | {str(owner[j]) for j, lab in y2.items()
                                              if lab == "entail"}
        # ★ 块级标签落盘（否则指标只能靠重跑 LLM 才能复算）
        a1set, bset = set(A1.tolist()), set(B.tolist())
        for j, lab in l1:
            arm = "A1" if j in a1set else ("A2" if j in a2set else "B")
            lab_rows.append(dict(cluster=ci, facet=facet, chunk=j, docid=str(owner[j]),
                                 arm=arm, label=lab, in_pool=1, nli=float(nli[j]),
                                 regex=int(j in a1set)))

        def m4(s: set) -> dict:
            """★ 集合口径 P/R/F1。

            ⚠️ 不能用 `setpf(o, gold, 50)` —— k=50 = 全部篇数时 `o[:50]` 含**所有 50 篇**，
            setF1 退化成只反映 `|gold|`，**两臂必然相同**（第一版就踩了这个坑，
            跑出"Δ=+0.000"的假结论）。
            """
            inter = len(s & gold)
            p = inter / max(len(s), 1)
            r = inter / max(len(gold), 1)
            return dict(setP=p, setR=r, setF1=2 * p * r / (p + r) if (p + r) else 0.0,
                        n_pred=len(s), inter=inter)
        r1, r2 = m4(set1), m4(set2)
        err1 = sum(1 for _, lab in l1 if lab == "ERR")
        err2 = sum(1 for j, lab in y2.items() if lab == "ERR")
        print(f"    臂①全判 {len(pool)} 次（ERR {err1}）→ 判正 {r1['n_pred']} 篇 ｜ "
              f"P {r1['setP']:.3f} R {r1['setR']:.3f} **F1 {r1['setF1']:.3f}**")
        print(f"    臂②级联 {len(A2)} 次（ERR {err2}）→ 判正 {r2['n_pred']} 篇 ｜ "
              f"P {r2['setP']:.3f} R {r2['setR']:.3f} **F1 {r2['setF1']:.3f}**"
              f" ｜ 与①集合差异 {len(set1 ^ set2)} 篇")
        rows.append(dict(cluster=ci, facet=facet, pool=len(pool), surv=len(surv),
                         a1=len(A1), a2=len(A2), dropped=len(B), n_gold=len(gold),
                         pool_docs=len(pool_docs),
                         full_calls=len(pool), casc_calls=len(A2),
                         full_setP=r1["setP"], full_setR=r1["setR"], full_setF1=r1["setF1"],
                         full_pred=r1["n_pred"], full_inter=r1["inter"],
                         casc_setP=r2["setP"], casc_setR=r2["setR"], casc_setF1=r2["setF1"],
                         casc_pred=r2["n_pred"], casc_inter=r2["inter"],
                         sym_diff=len(set1 ^ set2)))
    pd.DataFrame(rows).to_csv(HERE / "results" / "R2_C3_CASCADE.csv",
                              index=False, encoding="utf-8-sig")
    if lab_rows:
        pd.DataFrame(lab_rows).to_csv(HERE / "results" / "R2_C3_LABELS.csv",
                                      index=False, encoding="utf-8-sig")

    print("\n" + "=" * 122)
    print(f"【汇总】{len(picks)} 个 facet ｜ 池 {tot['pool']} 块 ｜ NLI 粗筛后 {tot['nli_keep']}"
          f"（{tot['nli_keep'] / max(tot['pool'], 1):.1%}）｜ regex 判正 {tot['regex']}"
          f" ｜ **需 LLM {tot['casc']}** ｜ **省 {1 - tot['casc'] / max(tot['pool'], 1):.1%}**")
    if not args.dry_run:
        R = pd.DataFrame(rows)
        print(f"\n  {'facet':<18}{'池':>5}{'级联调用':>9}{'节省':>8}"
              f"{'①F1':>7}{'②F1':>7}{'ΔF1':>8}{'①P':>7}{'②P':>7}{'①R':>7}{'②R':>7}{'集合差':>7}")
        for _, r in R.iterrows():
            print(f"  {str(r['facet'])[:17]:<18}{int(r['pool']):>5}{int(r['casc_calls']):>9}"
                  f"{1 - r['casc_calls'] / r['pool']:>8.1%}{r['full_setF1']:>7.3f}"
                  f"{r['casc_setF1']:>7.3f}{r['casc_setF1'] - r['full_setF1']:>+8.3f}"
                  f"{r['full_setP']:>7.3f}{r['casc_setP']:>7.3f}"
                  f"{r['full_setR']:>7.3f}{r['casc_setR']:>7.3f}{int(r['sym_diff']):>7}")
        print(f"\n  均：①F1 {R['full_setF1'].mean():.3f} ｜ ②F1 {R['casc_setF1'].mean():.3f}"
              f" ｜ **ΔF1 {R['casc_setF1'].mean() - R['full_setF1'].mean():+.3f}**"
              f" ｜ ①R {R['full_setR'].mean():.3f} ｜ ②R {R['casc_setR'].mean():.3f}"
              f" ｜ 调用 {int(R['full_calls'].sum())} → {int(R['casc_calls'].sum())}")
        print("  ⚠️ 两臂**同口径可比**；与 `gold_final3`（窗口协议+双判官）的差距含口径混杂，")
        print("     别直接归因给级联（逐块 vs 窗口、单判官 vs 双判官）。")
    print(f"\n  → 已写 results/R2_C3_CASCADE.csv"
          f"{' + R2_C3_LABELS.csv' if lab_rows else ''}（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
