r"""**C3 = 窗口判**（定稿协议）：每交付篇 1 次调用、输入 ≈4,200 字符窗口

## 协议（**逐字复用产出 `gold_final3` 的那套**，保证可比）
窗口构造（`_r2_gold_recalib.py` 原文参数，**一字不改**）：
```
pat, zh = F2[facet][0], F2[facet][1]
hit      = 该篇内 facet 正则命中的块
desens   = 子查询去掉与 pat 重合词（防泄露）后的版本
sim      = bge-m3([zh] + desens) @ 块向量        ← 取该篇内 **max** over 查询
sem_top  = 该篇内 sim 最大的 EV_TOPK_SEM=3 块
anch     = 命中的块前 EV_MAX_CHUNKS=4
orderl   = 去重(anch + sem_top)[:4]
窗口     = 按 orderl 顺序拼块，直到 EV_MAX_CHARS=4200 字符
```
判官：A=`PAPERPILOT_LLM`、B=`PAPERPILOT_JUDGE`，**四档**（entail/…）；
双方 entail → yes；双方非 entail → no；**分歧走第三轮**（`SYS_STRICT` 对抗式）→ 每对 ≈2.1 次调用。

## 三个对照臂（同一批交付篇，同真值）
| 臂 | 做法 | 调用/题 |
|---|---|---|
| **① 交付即正** | 融合排序取 top-k 篇，**全部当正例**（不判） | **0** |
| **② C3 窗口判** | ①的每篇建窗口 → 双判官 → 判正者留下 | ≈2.1·k |
| （上限） | `gold_final3` 自身 | — |

→ **①vs②** 直接回答"C3 的判官加了多少值"；**②vs gold** 是端到端召回。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_window.py --k 20 --per-cluster 1   # 小样
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_window.py --k 20                    # 全部 28 facet
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
RRF_C = 60
D_ZH, D_SD, D_SB = 100, 50, 50
RHO = 4.0
EV_TOPK_SEM, EV_MAX_CHUNKS, EV_MAX_CHARS = 3, 4, 4200      # ★ 与 gold 同参数
LOCK = threading.Lock()


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = _load("goldrecal", HERE / "tmp" / "_r2_gold_recalib.py")
SYS, SYS_STRICT, call, rx_toks = gr.SYS, gr.SYS_STRICT, gr.call, gr.rx_toks
_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="交付篇数（上一轮定的 20）")
    ap.add_argument("--per-cluster", type=int, default=0, help="每簇取几个 facet（0 = 全部）")
    ap.add_argument("--subqueries", default="subqueries.json",
                    help="窗口构造用的子查询（默认 v1 = 与 gold 同源）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="R2_C3_WINDOW.csv")
    args = ap.parse_args()

    pmap = json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
    pm_ok = {d for d, v in pmap.items() if v.get("ok")}
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    subq_all = json.loads((DEV / args.subqueries).read_text(encoding="utf-8"))

    sel = []
    for ci in (1, 2, 3):
        fs = sorted(f for (c, f) in combos if c == ci)
        if args.per_cluster:
            lst = sorted(fs, key=lambda f: len(NG.get((ci, f), set())))
            mid = lst[len(lst) // 2]
            fs = [mid] if mid in fs else fs[:1]
        sel += [(ci, f) for f in fs]
    print("=" * 122)
    print(f"【C3 窗口判】{len(sel)} 个 facet ｜ k={args.k} ｜ 窗口 = 锚点∪语义top"
          f"（{EV_TOPK_SEM} 块 / ≤{EV_MAX_CHUNKS} 块 / ≤{EV_MAX_CHARS} 字符）")
    print(f"  判官 = A(PAPERPILOT_LLM) + B(PAPERPILOT_JUDGE) + 分歧第三轮"
          f" ｜ 子查询 {args.subqueries}（与 gold 同源）")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    t0 = time.time()

    jobs, metas = [], []
    # ⚠️ 第一版把"读 parquet + 编码整簇"放在 **facet 循环里** → 每个 facet 重编码一次
    #    （实测 9 facet×30s ≈ 270s/簇，28 facet 共 1,052s）。这里提到**簇级**，编码只做一次。
    for ci in (1, 2, 3):
        c0 = ci - 1
        ch = pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{c0}.parquet")
        ls = pd.read_parquet(DEV / "corpus50" / f"c{c0}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"]) and d in pm_ok]
        titles = {str(r["docid"]): str(r.get("title") or "")[:120] for _, r in ls.iterrows()}
        chunks, owner = [], []
        for d in docs:
            cs = ch[ch.docid == d]["text"].astype(str).tolist()
            chunks += cs
            owner += [d] * len(cs)
        owner = np.array(owner)
        nb = len(chunks)
        idx = {d: np.where(owner == d)[0] for d in docs}
        E = enc.encode(chunks, batch_size=8, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)
        low = [c.lower() for c in chunks]
        print(f"  簇{ci} 就绪：{len(docs)} 篇 / {nb} 块 ｜ 编码 {time.time() - t0:.0f}s", flush=True)

        for facet in [f for (c, f) in sel if c == ci]:
            zh = str(F2[facet][1])
            sq = [str(q) for q in subq_all.get(facet, []) if str(q).strip()][:3] \
                or [str(F2[facet][3])]

            def rk(v: np.ndarray, n: int = nb) -> np.ndarray:
                r = np.empty(n, dtype=np.int64)
                r[np.argsort(-v)] = np.arange(n)
                return r

            def rv(q: str) -> np.ndarray:
                qv = enc.encode([q], normalize_embeddings=True, show_progress_bar=False,
                                convert_to_numpy=True).astype(np.float32)
                return (qv @ E.T)[0]

            cls = {"zh-dense": [rv(zh)], "sq-dense": [rv(q) for q in sq],
                   "sq-bm25": [bm.scores(q).astype(np.float32) for q in sq]}
            dd = {"zh-dense": D_ZH, "sq-dense": D_SD, "sq-bm25": D_SB}
            sc = np.zeros(nb, dtype=np.float64)
            for c, ss in cls.items():
                w = RHO if c == "sq-bm25" else 1.0
                for s in ss:
                    r = rk(s)
                    m = r < dd[c]
                    sc[m] += w / (RRF_C + r[m] + 1)
            dmax = {d: float(sc[idx[d]].max()) for d in docs}
            order = sorted(docs, key=lambda x: -dmax[x])          # ★ 完整排序（供 k 扫描）
            deliv = order[: args.k]
            gold = (NG.get((ci, facet)) or set()) & set(docs)

            # ── 窗口构造（与 gold 同法，一字不改）──
            hit = np.array([bool(_m.judge(c, facet)) for c in low])
            rx = rx_toks(str(F2[facet][0]))
            desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s
                      for s in subq_all.get(facet, [])]
            qv = enc.encode([zh] + desens, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
            sim = qv @ E.T
            claim = (F2[facet][1] if str(F2[facet][1]).startswith("本文")
                     else f"该论文{F2[facet][1]}")
            for rank0, d in enumerate(deliv):
                ids = idx[d]
                sem = ids[np.argsort(-sim[:, ids].max(axis=0))[:EV_TOPK_SEM]]
                anch = ids[hit[ids]][:EV_MAX_CHUNKS]
                orderl, seen = [], set()
                for j in list(anch) + list(sem):
                    if int(j) not in seen:
                        seen.add(int(j))
                        orderl.append(int(j))
                ev, tot = [], 0
                for j in orderl[:EV_MAX_CHUNKS]:
                    t2 = chunks[j]
                    if tot + len(t2) > EV_MAX_CHARS:
                        t2 = t2[: max(0, EV_MAX_CHARS - tot)]
                    if not t2:
                        break
                    ev.append(t2)
                    tot += len(t2)
                jobs.append(dict(cluster=ci, facet=facet, docid=str(d), rank=rank0 + 1,
                                 title=titles.get(str(d), ""), claim=str(claim),
                                 evidence="\n---\n".join(ev)))
            metas.append(dict(cluster=ci, facet=facet, docs=docs, gold=gold,
                              order=order, n_gold=len(gold)))
            print(f"  簇{ci} {facet}：gold {len(gold)} ｜ 交付 {len(deliv)} ｜ "
                  f"交付∩gold {len(set(deliv) & gold)} ｜ {time.time() - t0:.0f}s", flush=True)
    del enc

    print(f"\n待判 {len(jobs)} 对 ｜ 每对 2 次 + 分歧第三轮 → 约 {len(jobs) * 2.1:.0f} 次调用\n", flush=True)

    def work(j: dict) -> dict:
        user = (f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                f"论文原文片段：\n{j['evidence']}")
        a = call("PAPERPILOT_LLM", SYS, user)
        b = call("PAPERPILOT_JUDGE", SYS, user)
        la = str(a.get("label") or "ERR").lower()
        lb = str(b.get("label") or "ERR").lower()
        final, third = "", ""
        if la == "entail" and lb == "entail":
            final = "yes"
        elif "entail" not in (la, lb) and "ERR" not in (la, lb):
            final = "no"
        else:
            t3 = call("PAPERPILOT_LLM", SYS_STRICT, user)
            third = str(t3.get("answer") or "")
            final = "yes" if third.upper().startswith("Y") else "no"
        with LOCK:
            n = len(J)
            if n % 40 == 0:
                print(f"    {n}/{len(jobs)} …", flush=True)
        return dict(cluster=j["cluster"], facet=j["facet"], docid=j["docid"],
                    rank=int(j.get("rank", 0)), ev_chars=len(j["evidence"]),
                    A=la, B=lb, third=third, final=final)

    J: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, jobs):
            J.append(r)
    jd = pd.DataFrame(J)
    jd.to_csv(HERE / "results" / args.out, index=False, encoding="utf-8-sig")
    print(f"\n  判完（{time.time() - t0:.0f}s）｜ A/B 分歧 {int((jd.A != jd.B).sum())}"
          f" ｜ ERR {int((jd.A.astype(str).str.contains('ERR')).sum() + (jd.B.astype(str).str.contains('ERR')).sum())}")

    # ── 三臂指标 ──
    yes = {(int(r.cluster), str(r.facet), str(r.docid)) for r in jd.itertuples()
           if r.final == "yes"}
    rows = []
    for m in metas:
        deliv = m["order"][: args.k]
        pred_deliv = set(deliv)                                   # ① 交付即正
        pred_c3 = {d for d in deliv if (m["cluster"], m["facet"], d) in yes}
        for tag, s in (("①交付即正", pred_deliv), ("②C3窗口判", pred_c3)):
            inter = len(s & m["gold"])
            p = inter / max(len(s), 1)
            r = inter / max(len(m["gold"]), 1)
            rows.append(dict(cluster=m["cluster"], facet=m["facet"], arm=tag,
                             n_deliv=len(deliv), n_pred=len(s), n_gold=m["n_gold"],
                             inter=inter, setP=p, setR=r,
                             setF1=2 * p * r / (p + r) if (p + r) else 0.0))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / args.out.replace(".csv", "_ARMS.csv"),
             index=False, encoding="utf-8-sig")

    print("\n" + "=" * 122)
    print(f"【结果】k={args.k} ｜ 交付即正 vs C3 窗口判（真值 `gold_final3`）")
    print(f"  {'簇':>3}{'facet':<20}{'gold':>6}{'交付':>6}{'交付∩gold':>10}"
          f"{'①P':>7}{'①R':>7}{'①F1':>7}{'②留':>6}{'②P':>7}{'②R':>7}{'②F1':>7}")
    for m in metas:
        a1 = R[(R.cluster == m["cluster"]) & (R.facet == m["facet"]) & (R.arm == "①交付即正")].iloc[0]
        a2 = R[(R.cluster == m["cluster"]) & (R.facet == m["facet"]) & (R.arm == "②C3窗口判")].iloc[0]
        print(f"  {m['cluster']:>3}{m['facet'][:19]:<20}{m['n_gold']:>6}{min(args.k, len(m['order'])):>6}"
              f"{int(a1.inter):>10}{a1.setP:>7.3f}{a1.setR:>7.3f}{a1.setF1:>7.3f}"
              f"{int(a2.n_pred):>6}{a2.setP:>7.3f}{a2.setR:>7.3f}{a2.setF1:>7.3f}")
    for tag in ("①交付即正", "②C3窗口判"):
        t = R[R.arm == tag]
        print(f"\n  【{tag}】均 setP {t.setP.mean():.3f} ｜ setR {t.setR.mean():.3f}"
              f" ｜ **setF1 {t.setF1.mean():.3f}** ｜ 判正篇数 {t.n_pred.mean():.1f}")
    d1 = R[R.arm == "①交付即正"].setF1.mean()
    d2 = R[R.arm == "②C3窗口判"].setF1.mean()
    print(f"\n  ★ **C3 判官增益 ΔF1 = {d2 - d1:+.3f}**（①→②）｜ 调用 ≈{len(jobs) * 2.1:.0f}"
          f" ｜ 判官每对 ≈2.1 次")
    print(f"  → 已写 results/{args.out} + {args.out.replace('.csv', '_ARMS.csv')}"
          f"（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
