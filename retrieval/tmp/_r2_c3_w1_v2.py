r"""**W1：窗口内容 v1 → v2 子查询**（只重判受影响的对，配对同批）

## 为什么可以"很快"
窗口构造里只有一项来自子查询 —— `desens`（防泄露后的子查询）：
```
sem_top = 该篇内  sim = bge-m3([中文论断] + desens) @ 块向量 的 top-3
```
`subqueries.json`（v1）与 `subqueries_v2.json`（v2）**只差 3 个 facet**：
`prompt_eng`（v1 **缺**子查询）、`knowledge_distill`（v1 泄露 45.8%）、`case_study`（v1 冗余 0.67）。
→ **其余 facet 的窗口逐字节相同** ⇒ 不必重判（且复用更好：不引入新抖动）。

## 做法（配对同批，控制 LLM 非确定性）
1. 对全部 28 facet × 50 篇 **重建两套窗口**，逐字节比较 → 列出真正变化的对（**零 LLM**）；
2. 只对**变化的对**，用 **v1 窗口与 v2 窗口各判一次**（A+B+第三轮）→ **同批配对比较**；
3. 合并：未变的对沿用 `R2_C3_WINDOW_FULL.csv` 的 v1 判果；变化的对用步骤 2 的结果；
4. 两套标签各跑一次 k 扫描 → 报 v1 vs v2 各项指标。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_c3_w1_v2.py --k 30
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
EV_TOPK_SEM, EV_MAX_CHUNKS, EV_MAX_CHARS = 3, 4, 4200
KS = [13, 20, 30, 40, 50]
CALLS_PER_PAIR = 2.1
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


def build_window(chunks, E, enc, ids, hit, sim, subqs) -> str:
    """与 gold 同法的窗口（≤EV_MAX_CHUNKS 块 / ≤EV_MAX_CHARS 字符）。"""
    sem = ids[np.argsort(-sim[:, ids].max(axis=0))[:EV_TOPK_SEM]]
    anch = ids[hit[ids]][:EV_MAX_CHUNKS]
    orderl, seen = [], set()
    for j in list(anch) + list(sem):
        if int(j) not in seen:
            seen.add(int(j))
            orderl.append(int(j))
    ev, tot = [], 0
    for j in orderl[:EV_MAX_CHUNKS]:
        t = chunks[j]
        if tot + len(t) > EV_MAX_CHARS:
            t = t[: max(0, EV_MAX_CHARS - tot)]
        if not t:
            break
        ev.append(t)
        tot += len(t)
    return "\n---\n".join(ev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-pairs", type=int, default=0, help="限制重判对数（0=不限）")
    args = ap.parse_args()

    sq1 = json.loads((DEV / "subqueries.json").read_text(encoding="utf-8"))
    sq2 = json.loads((DEV / "subqueries_v2.json").read_text(encoding="utf-8"))
    diff_f = sorted(f for f in set(sq1) | set(sq2)
                    if [str(x) for x in sq1.get(f, [])] != [str(x) for x in sq2.get(f, [])])
    print("=" * 122)
    print(f"【W1 窗口内容 v1→v2】子查询差异 facet（{len(diff_f)}）：{diff_f}")
    for f in diff_f:
        print(f"  {f:<20} v1 {len(sq1.get(f, []))} 条 → v2 {len(sq2.get(f, []))} 条")

    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"])
          for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    pm_ok = {d for d, v in json.loads((DEV / "corpus50" / "pdf_map_all.json")
                                      .read_text(encoding="utf-8")).items() if v.get("ok")}
    # 既有 v1 判果（含 rank → 交付顺序）
    base = pd.read_csv(HERE / "results" / "R2_C3_WINDOW_FULL.csv")
    base["final"] = base["final"].astype(str).str.lower()
    RANK = {(int(r.cluster), str(r.facet), str(r.docid)): int(r.rank)
            for r in base.itertuples()}

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    t0 = time.time()

    # ── ① 重建两套窗口并逐字节比较（零 LLM）──
    changed, meta = [], {}
    for ci in (1, 2, 3):
        c0 = ci - 1
        ch = pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{c0}.parquet")
        ls = pd.read_parquet(DEV / "corpus50" / f"c{c0}.parquet")
        docs = [d for d in ls["docid"].tolist()
                if d in set(ch["docid"]) and d in pm_ok]
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
        low = [c.lower() for c in chunks]
        print(f"  簇{ci} 就绪 {len(docs)} 篇 / {nb} 块 ｜ {time.time() - t0:.0f}s", flush=True)

        for facet in sorted(f for (c, f) in combos if c == ci):
            rankmap = {d: RANK.get((ci, facet, d), 999) for d in docs}
            order = sorted(docs, key=lambda d: rankmap[d])
            gold = (NG.get((ci, facet)) or set()) & set(docs)
            meta[(ci, facet)] = dict(order=order, gold=gold,
                                     n_gold=len(gold))
            hit = np.array([bool(_m.judge(c, facet)) for c in low])
            zh = str(F2[facet][1])
            claim = (F2[facet][1] if str(F2[facet][1]).startswith("本文")
                     else f"该论文{F2[facet][1]}")
            wins = {}
            for tag, sqall in (("v1", sq1), ("v2", sq2)):
                rx = rx_toks(str(F2[facet][0]))
                desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s
                          for s in sqall.get(facet, [])]
                qv = enc.encode([zh] + desens, normalize_embeddings=True,
                                convert_to_numpy=True).astype(np.float32)
                sim = qv @ E.T
                wins[tag] = {d: build_window(chunks, E, enc, idx[d], hit, sim, sqall)
                             for d in order}
            nch = sum(1 for d in order if wins["v1"][d] != wins["v2"][d])
            if nch:
                diffs = [d for d in order if wins["v1"][d] != wins["v2"][d]]
                print(f"    ★ 簇{ci} {facet}：窗口变化 {nch}/{len(order)} 篇")
                for d in diffs:
                    changed.append(dict(cluster=ci, facet=facet, docid=str(d),
                                        title=titles.get(str(d), ""), claim=str(claim),
                                        w1=wins["v1"][d], w2=wins["v2"][d]))
            else:
                print(f"    · 簇{ci} {facet}：窗口**逐字节相同** → 复用 v1 判果")
    del enc
    print(f"\n  ★ 窗口真正变化的对：**{len(changed)}**（其余复用 v1 判果）"
          f" ｜ 构建耗 {time.time() - t0:.0f}s", flush=True)
    if not changed:
        print("  → 无变化，v1 = v2，无需重判。")
        return 0
    if args.max_pairs:
        changed = changed[: args.max_pairs]

    # ── ② 配对同批重判（同一批对子，v1/v2 各一次）──
    def judge_all(j) -> dict:
        out = {"cluster": j["cluster"], "facet": j["facet"], "docid": j["docid"]}
        for tag, ev in (("v1", j["w1"]), ("v2", j["w2"])):
            user = (f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                    f"论文原文片段：\n{ev}")
            a = call("PAPERPILOT_LLM", SYS, user)
            b = call("PAPERPILOT_JUDGE", SYS, user)
            la = str(a.get("label") or "ERR").lower()
            lb = str(b.get("label") or "ERR").lower()
            if la == "entail" and lb == "entail":
                fin = "yes"
            elif "entail" not in (la, lb) and "ERR" not in (la, lb):
                fin = "no"
            else:
                t3 = call("PAPERPILOT_LLM", SYS_STRICT, user)
                fin = "yes" if str(t3.get("answer") or "").upper().startswith("Y") else "no"
            out[tag] = fin
        with LOCK:
            n = len(D)
            if n % 20 == 0:
                print(f"    {n}/{len(changed)} …", flush=True)
        return out

    print(f"\n  重判 {len(changed)} 对 × 2（v1/v2）≈ {len(changed) * 2 * CALLS_PER_PAIR:.0f} 次调用\n")
    D: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(judge_all, changed):
            D.append(r)
    dd = pd.DataFrame(D)
    dd.to_csv(HERE / "results" / "R2_C3_W1_PAIRED.csv", index=False, encoding="utf-8-sig")
    flip = dd[dd.v1 != dd.v2]
    print(f"\n  重判完成（{time.time() - t0:.0f}s）｜ 判果翻转 {len(flip)}/{len(dd)} 对")
    for r in flip.head(12).itertuples():
        print(f"    {r.cluster} {r.facet} {r.docid}: v1={r.v1} → v2={r.v2}")

    # ── ③ 合并成两套标签 ──
    b1 = {(int(r.cluster), str(r.facet), str(r.docid)): r.final for r in base.itertuples()}
    L1, L2 = dict(b1), dict(b1)
    for r in dd.itertuples():
        k = (r.cluster, r.facet, r.docid)
        L1[k], L2[k] = r.v1, r.v2

    # ── ④ k 扫描两版 ──
    def sweep(L: dict) -> pd.DataFrame:
        rows = []
        for (ci, fa), m in meta.items():
            order, gold = m["order"], m["gold"]
            for k in KS:
                top = order[:k]
                for tag, s in (("①交付即正", set(top)),
                               ("②C3窗口判", {d for d in top if L.get((ci, fa, d)) == "yes"})):
                    inter = len(s & gold)
                    p = inter / max(len(s), 1)
                    r = inter / max(len(gold), 1)
                    rows.append(dict(cluster=ci, facet=fa, k=k, arm=tag, n_pred=len(s),
                                     n_gold=len(gold), inter=inter, setP=p, setR=r,
                                     setF1=2 * p * r / (p + r) if (p + r) else 0.0))
        return pd.DataFrame(rows)
    S1, S2 = sweep(L1), sweep(L2)
    S1.to_csv(HERE / "results" / "R2_C3_W1_SWEEP_v1.csv", index=False, encoding="utf-8-sig")
    S2.to_csv(HERE / "results" / "R2_C3_W1_SWEEP_v2.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 122)
    print(f"【表① 只受影响的 facet：v1 vs v2（②C3窗口判，k={args.k}）】")
    aff = sorted({(r.cluster, r.facet) for r in dd.itertuples()})
    print(f"  {'簇':>3}{'facet':<20}{'gold':>6}{'v1留':>6}{'v1P':>7}{'v1R':>7}{'v1F1':>7}"
          f"{'v2留':>6}{'v2P':>7}{'v2R':>7}{'v2F1':>7}{'ΔF1':>8}")
    for ci, fa in aff:
        a = S1[(S1.cluster == ci) & (S1.facet == fa) & (S1.k == args.k)
               & (S1.arm == "②C3窗口判")].iloc[0]
        b = S2[(S2.cluster == ci) & (S2.facet == fa) & (S2.k == args.k)
               & (S2.arm == "②C3窗口判")].iloc[0]
        print(f"  {ci:>3}{str(fa)[:19]:<20}{int(a.n_gold):>6}{int(a.n_pred):>6}{a.setP:>7.3f}"
              f"{a.setR:>7.3f}{a.setF1:>7.3f}{int(b.n_pred):>6}{b.setP:>7.3f}{b.setR:>7.3f}"
              f"{b.setF1:>7.3f}{b.setF1 - a.setF1:>+8.3f}")

    print("\n" + "=" * 122)
    print("【表② 全局 28 facet：v1 vs v2 的 k 曲线（②C3窗口判）】")
    print(f"  {'k':>4}{'v1P':>7}{'v1R':>7}{'v1F1':>7}{'v2P':>7}{'v2R':>7}{'v2F1':>7}{'ΔF1':>8}"
          f"{'翻转对':>8}")
    for k in KS:
        a = S1[(S1.k == k) & (S1.arm == "②C3窗口判")]
        b = S2[(S2.k == k) & (S2.arm == "②C3窗口判")]
        print(f"  {k:>4}{a.setP.mean():>7.3f}{a.setR.mean():>7.3f}{a.setF1.mean():>7.3f}"
              f"{b.setP.mean():>7.3f}{b.setR.mean():>7.3f}{b.setF1.mean():>7.3f}"
              f"{b.setF1.mean() - a.setF1.mean():>+8.3f}{len(flip):>8}")

    print("\n" + "=" * 122)
    print("【表③ ①不判（参照，与窗口无关，两版应相同）】")
    for k in KS:
        a = S1[(S1.k == k) & (S1.arm == "①交付即正")]
        print(f"  k={k:<4}①P {a.setP.mean():.3f} ｜ ①R {a.setR.mean():.3f} ｜ ①F1 {a.setF1.mean():.3f}")
    print(f"\n  → 已写 results/R2_C3_W1_PAIRED.csv + R2_C3_W1_SWEEP_v1.csv"
          f" + R2_C3_W1_SWEEP_v2.csv（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
