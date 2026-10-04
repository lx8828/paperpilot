"""**A｜真值敏感性**：Phase1 的臂排序在三种真值下是否稳定？

这是"能不能回去调参"的**决定性问题**：
- 若臂排序在三种真值下一致 → **相对结论稳健** → 可以调参（绝对值仍受真值影响）
- 若排序翻转 → 必须先修尺子

三种真值：
  A `anchor`  —— 现行词面锚点（宽松）
  B `new`     —— 双 LLM 重标（严格；证据只有 4 块 → 有假阴性）
（另报每个真值下的真值规模与"可用真值数"，空真值的 facet 自动剔除）

打分固定为**干净查询**（`zh` + 分解子查询，无锚点词），与 `_r2_phase1_clean.py` 同源。
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "data" / "r2dev" / "clusters"
SUBQ = HERE / "data" / "r2dev" / "subqueries.json"
RECAL = HERE / "data" / "r2dev" / "gold_recalib.csv"
OUT = HERE / "results"

CHUNK, OVERLAP, POOL = 1000, 100, 40
KS = [5, 10]
ARMS = ["A base", "B mq_max", "C mq_rr", "D mmr λ=0.5", "E mq_mmr λ=0.5"]

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf, alpha_ndcg = _m.mrecall, _m.strecall, _m.setpf, _m.alpha_ndcg

subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))
rec = pd.read_csv(RECAL)
rec["anchor_gold"] = rec["anchor_gold"].astype(bool)
rec["new_gold"] = rec["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
# ── 最终真值 `final` = 4 块证据的严格真值，**用富证据复判结果覆盖** ──
RICH = HERE / "data" / "r2dev" / "gold_recalib_rich.csv"
rec["final_gold"] = rec["new_gold"]
if RICH.exists():
    rich = pd.read_csv(RICH)
    rich["yes"] = rich["new_gold_rich"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    mp = {(int(r.cluster), str(r.facet), str(r.docid)): bool(r.yes) for r in rich.itertuples()}
    n_ov = 0
    for i, r in rec.iterrows():
        k = (int(r["cluster"]), str(r["facet"]), str(r["docid"]))
        if k in mp:
            rec.at[i, "final_gold"] = mp[k]
            n_ov += 1
    print(f"最终真值：{n_ov} 条被富证据复判覆盖（占 {n_ov / len(rec):.1%}）")

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def zn(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def paired_ci(a, b, n=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bt = d[idx].mean(axis=1)
    return d.mean(), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))


meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
enc.max_seq_length = 512
if dev == "cuda":
    enc.half()

rows, pertruth, goldsize = [], [], []
t0 = time.time()
for ci in range(len(meta)):
    sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
    docs = sub["docid"].tolist()
    chunks, owner = [], []
    for i, t in enumerate(sub["full_paper"].tolist()):
        cs = _m.chunks_of(t)
        chunks += cs
        owner += [docs[i]] * len(cs)
    owner = np.array(owner)
    C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    idx = {d: np.where(owner == d)[0] for d in docs}
    print(f"【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

    for facet in meta[ci]["usable"]:
        g = rec[(rec["cluster"] == ci + 1) & (rec["facet"] == facet)]
        if not len(g):
            continue
        golds = {"anchor": set(g[g["anchor_gold"]]["docid"]),
                 "new": set(g[g["new_gold"]]["docid"]),
                 "final": set(g[g["final_gold"]]["docid"])}
        if any(not v or len(v) == len(docs) for v in golds.values()):
            continue                                  # 空真值/全正例 → 该题不可用
        goldsize.append(dict(cluster=ci + 1, facet=facet,
                             n_anchor=len(golds["anchor"]), n_new=len(golds["new"]),
                             n_final=len(golds["final"])))
        zh = F[facet][1]
        qs = [zh] + list(subq_all.get(facet, []))
        S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True
                       ).astype(np.float32) @ C.T
        Smax = S.max(axis=0)
        rankA = sorted(docs, key=lambda d: -S[0][idx[d]].max())
        rankB = sorted(docs, key=lambda d: -Smax[idx[d]].max())
        orders = [list(np.argsort(-S[j])) for j in range(len(qs))]
        inter, seen = [], set()
        for rnd in range(len(chunks)):
            for o in orders:
                if rnd < len(o) and int(o[rnd]) not in seen:
                    seen.add(int(o[rnd]))
                    inter.append(int(o[rnd]))
        rankC = list(dict.fromkeys(owner[inter].tolist()))

        def mmr_rank(base_score, lam):
            pool = list(np.argsort(-base_score)[:POOL])
            sel, cand = [], np.array(pool)
            bz = zn(base_score[cand])
            while len(sel) < min(POOL, len(pool)):
                if not sel:
                    pick = int(np.argmax(bz))
                else:
                    red = (C[cand] @ C[np.array(sel)].T).max(axis=1)
                    pick = int(np.argmax(lam * bz - (1 - lam) * zn(red)))
                sel.append(int(cand[pick]))
                cand = np.delete(cand, pick)
                bz = np.delete(bz, pick)
            return list(dict.fromkeys(owner[np.array(sel)].tolist()))

        arms = {"A base": rankA, "B mq_max": rankB, "C mq_rr": rankC,
                "D mmr λ=0.5": mmr_rank(S[0], 0.5), "E mq_mmr λ=0.5": mmr_rank(Smax, 0.5)}
        for gname, gold in golds.items():
            for nm, rk in arms.items():
                for k in KS:
                    p, r, f1 = setpf(rk, gold, k)
                    rows.append(dict(cluster=ci + 1, facet=facet, gold=gname, arm=nm, k=k,
                                     n_gold=len(gold), MRecall=mrecall(rk, gold, k),
                                     StRecall=strecall(rk, gold, k), setP=p, setF1=f1,
                                     aNDCG=alpha_ndcg(rk, gold, k)))
                    if k == 10:
                        pertruth.append(dict(cluster=ci + 1, facet=facet, gold=gname,
                                             arm=nm, StRecall=strecall(rk, gold, k)))
    print(f"  …{time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
pt = pd.DataFrame(pertruth)
gs = pd.DataFrame(goldsize)
df.to_csv(OUT / "R2_GOLD_SENSITIVITY.csv", index=False, encoding="utf-8")
nq = df.groupby(["cluster", "facet"]).ngroups
def agg(g: str, a: str, k: int, col: str) -> float:
    t = df[(df["gold"] == g) & (df["arm"] == a) & (df["k"] == k)]
    return float(t[col].mean()) if len(t) else float("nan")


def rank(g: str) -> list[str]:
    return [a for a, _ in sorted(((a, agg(g, a, 10, "StRecall")) for a in ARMS),
                                 key=lambda t: -t[1])]


print(f"\n{'=' * 118}\n【真值规模】可用题 {nq}（空真值/全正例的已剔除）"
      f"｜锚点正例均 {gs['n_anchor'].mean():.1f} → 严格 {gs['n_new'].mean():.1f}"
      f" → 最终 {gs['n_final'].mean():.1f} / 20 篇")
print(f"\n【①】臂 × 真值 × k=10")
print(f"  {'臂':<18}{'anchor StR':>12}{'严格 StR':>10}{'最终 StR':>10}{'Δ(最终−锚点)':>14}"
      f"{'锚点 F1':>9}{'最终 F1':>9}")
for a in ARMS:
    va, vn, vf = (agg("anchor", a, 10, "StRecall"), agg("new", a, 10, "StRecall"),
                  agg("final", a, 10, "StRecall"))
    print(f"  {a:<18}{va:>12.3f}{vn:>10.3f}{vf:>10.3f}{vf - va:>+14.3f}"
          f"{agg('anchor', a, 10, 'setF1'):>9.3f}{agg('final', a, 10, 'setF1'):>9.3f}")

print(f"\n【②】**臂排序**（StRecall@10，降序）")
for g in ("anchor", "new", "final"):
    print(f"  {g:<8}" + "  >  ".join(f"{a}({agg(g, a, 10, 'StRecall'):.3f})" for a in rank(g)))
ra, rf = rank("anchor"), rank("final")
print(f"  → 锚点 vs 最终 排序一致？ {'✅ 一致' if ra == rf else f'❌ **不一致**'}")
if ra != rf:
    print(f"     锚点：{ra}")
    print(f"     最终：{rf}")

print(f"\n【③】配对 bootstrap 95% CI（三种真值各做一遍）")
for g in ("anchor", "new", "final"):
    print(f"  —— 真值 = {g} ——")
    for a, b in (("B mq_max", "A base"), ("C mq_rr", "A base"), ("D mmr λ=0.5", "A base")):
        A = pt[(pt["gold"] == g) & (pt["arm"] == a)].sort_values(["cluster", "facet"])
        B = pt[(pt["gold"] == g) & (pt["arm"] == b)].sort_values(["cluster", "facet"])
        m = A.merge(B, on=["cluster", "facet"], suffixes=("_a", "_b"))
        d, lo, hi = paired_ci(m["StRecall_a"].values, m["StRecall_b"].values)
        sig = "✅ 显著" if (lo > 0 or hi < 0) else "· 不显著"
        print(f"    {a:<15} − {b:<10} Δ {d:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
print(f"\n已写 {OUT / 'R2_GOLD_SENSITIVITY.csv'}")
