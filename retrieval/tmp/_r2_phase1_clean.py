"""**Phase1 复跑（干净版）**：多查询 / 轮询聚合 / MMR 在**无泄露打分**下到底值不值

## 为什么要复跑（见 `R2_LEAK_AUDIT_20260929.md`）
第一版 `_r2_phase1.py` 的查询集是 `qs = [f"{zh} {anchor}"] + subqueries`：
- 第一条查询 = 中文题面 **+ 英文锚点词**，而锚点词 **63% 与 gold 正则同词** → **泄露**
- 于是"多查询 / 轮询 / MMR"的相对增益可能全建立在泄露上

## 本脚本的三种查询集
| 变体 | 查询集 | 状态 |
|---|---|---|
| `contam` | `[zh+anchor] + subqueries` | ❌ 旧版，仅作对照 |
| **`clean`** | `[zh] + subqueries` | ✅ 子查询由 LLM 从**中文题面**生成（`subqueries.json`） |
| **`desens`** | `[zh] + 净化子查询` | ✅ **最严**：把子查询里与 gold 正则重合的词删掉 |

同时报**子查询 ↔ gold 正则**的词汇重合率（"消融实验"译成 "ablation" 是**语言必然**，
不是我们泄露；但仍需量化并给出净化臂作为下界）。

## 臂（沿用第一版）
A base（单查询）／B mq_max（多查询取最大）／C mq_rr（轮询聚合，AMER）／
D mmr（覆盖导向重排，λ 扫描）／E mq_mmr

## 指标
MRecall@k / StRecall@k（标准口径）+ 集合 P/F1 + α-nDCG，附**随机基线与 oracle 上界**，
并对关键对照做**配对 bootstrap 置信区间**（35 个真值，不能只看均值）。
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
OUT = HERE / "results"

CHUNK, OVERLAP, POOL = 1000, 100, 40
KS = [5, 10, 20]
LAMBDAS = [1.0, 0.7, 0.5, 0.3]

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf, alpha_ndcg = _m.mrecall, _m.strecall, _m.setpf, _m.alpha_ndcg


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", str(s).lower())}


def rx_tokens(pat: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", re.sub(r"\\[a-zA-Z]", " ", str(pat)).lower())}


subq_all: dict[str, list[str]] = json.loads(SUBQ.read_text(encoding="utf-8"))

# ── 子查询 ↔ gold 正则 的词汇重合（诚实量化）────────────────────────────────
print("=" * 118)
print("【①】子查询（LLM 从中文题面生成）与 gold 正则的词汇重合")
rows_ov = []
for f, subs in subq_all.items():
    if f not in F:
        continue
    rx = rx_tokens(F[f][0])
    ov = [sorted(tokens(s) & rx) for s in subs]
    rows_ov.append(dict(facet=f, n_sub=len(subs),
                        ov_rate=float(np.mean([len(o) / max(1, len(tokens(s)))
                                               for o, s in zip(ov, subs)])),
                        ov_words=sorted({w for o in ov for w in o})))
dov = pd.DataFrame(rows_ov)
print(f"  facet 数 {len(dov)} ｜ 子查询平均 {dov['n_sub'].mean():.1f} 条 ｜ "
      f"**平均与 gold 正则的重合 token 占比 {dov['ov_rate'].mean():.1%}**")
print("  重合最多的几个 facet：")
for _, r in dov.sort_values("ov_rate", ascending=False).head(5).iterrows():
    print(f"    {r['facet']:<18}{r['ov_rate']:>7.1%}  {r['ov_words']}")
print("  注：`消融实验 → ablation` 属**语言必然**（该概念只有这一个英文词），"
      "非我们按 gold 反推；但仍用 `desens` 臂给出下界。")

# ── 主循环 ──────────────────────────────────────────────────────────────────
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def zn(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def paired_ci(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """配对 bootstrap 95% CI（a − b）。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
enc.max_seq_length = 512
if dev == "cuda":
    enc.half()

VARIANTS = ["contam", "clean", "desens"]
rows, pertruth = [], []
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
    low = [c.lower() for c in chunks]
    idx = {d: np.where(owner == d)[0] for d in docs}
    print(f"\n【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

    for facet in meta[ci]["usable"]:
        pat, zh, anchor, para = F[facet]
        hit = np.array([_m.judge(c, facet) for c in low])
        gold = {d for d in docs if hit[idx[d]].any()}
        if not gold or len(gold) == len(docs):
            continue
        subs = subq_all.get(facet, [])
        rx = rx_tokens(pat)
        desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s for s in subs]
        qsets = {"contam": [f"{zh} {anchor}"] + list(subs),
                 "clean": [zh] + list(subs),
                 "desens": [zh] + desens}

        for vname, qs in qsets.items():
            S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True
                           ).astype(np.float32) @ C.T
            rankA = sorted(docs, key=lambda d: -S[0][idx[d]].max())
            Smax = S.max(axis=0)
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
                sel: list[int] = []
                cand = np.array(pool)
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

            arms = {"A base": rankA, "B mq_max": rankB, "C mq_rr": rankC}
            for lam in LAMBDAS:
                arms[f"D mmr λ={lam}"] = mmr_rank(S[0], lam)
            for lam in (0.5, 0.3):
                arms[f"E mq_mmr λ={lam}"] = mmr_rank(Smax, lam)
            # 随机基线（200 次排列均值）与 oracle 上界
            rng = np.random.default_rng(0)
            rand_perm = [rng.permutation(docs).tolist() for _ in range(200)]
            arms["random(200×)"] = None
            arms["oracle(上界)"] = sorted(docs, key=lambda d: (d not in gold, d))

            for nm, rk in arms.items():
                for k in KS:
                    if nm == "random(200×)":
                        vals = [setpf(p_, gold, k) for p_ in rand_perm]
                        st = [strecall(p_, gold, k) for p_ in rand_perm]
                        mr = [mrecall(p_, gold, k) for p_ in rand_perm]
                        an = [alpha_ndcg(p_, gold, k) for p_ in rand_perm]
                        p, r, f1 = (float(np.mean([x[0] for x in vals])),
                                    float(np.mean([x[1] for x in vals])),
                                    float(np.mean([x[2] for x in vals])))
                        stv, mrv, anv = float(np.mean(st)), float(np.mean(mr)), float(np.mean(an))
                    else:
                        p, r, f1 = setpf(rk, gold, k)
                        stv, mrv, anv = strecall(rk, gold, k), mrecall(rk, gold, k), alpha_ndcg(rk, gold, k)
                    rows.append(dict(cluster=ci + 1, facet=facet, variant=vname, arm=nm, k=k,
                                     n_gold=len(gold), MRecall=mrv, StRecall=stv,
                                     setP=p, setF1=f1, aNDCG=anv))
                    if k == 10 and nm in ("A base", "B mq_max", "C mq_rr", "D mmr λ=0.5"):
                        pertruth.append(dict(cluster=ci + 1, facet=facet, variant=vname,
                                             arm=nm, StRecall=stv, setF1=f1))
    print(f"  …{time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "R2_PHASE1_CLEAN.csv", index=False, encoding="utf-8")
pt = pd.DataFrame(pertruth)
nq = df.groupby(["cluster", "facet"]).ngroups
print(f"\n{'=' * 118}\n【②】Phase1 复跑结果（{nq} 个真值 × {len(VARIANTS)} 种查询集 × 臂 × k）")

for k in (5, 10):
    print(f"\n  ── k={k}（k=20 无信息量：随机基线 StRecall 也 = 1.000，故不列）──")
    print(f"  {'臂':<18}{'contam StRecall':>17}{'clean StRecall':>16}{'desens StRecall':>17}"
          f"{'clean 集合F1':>14}{'clean α-nDCG':>13}")
    arms = [a for a in df["arm"].unique() if a != "random(200×)"]
    for a in ["A base", "B mq_max", "C mq_rr"] + \
             [f"D mmr λ={l}" for l in LAMBDAS] + ["E mq_mmr λ=0.5", "E mq_mmr λ=0.3", "oracle(上界)"]:
        if a not in arms:
            continue
        vals = {}
        for v in VARIANTS:
            t = df[(df["arm"] == a) & (df["variant"] == v) & (df["k"] == k)]
            vals[v] = t["StRecall"].mean()
        tc = df[(df["arm"] == a) & (df["variant"] == "clean") & (df["k"] == k)]
        print(f"  {a:<18}{vals['contam']:>17.3f}{vals['clean']:>16.3f}{vals['desens']:>17.3f}"
              f"{tc['setF1'].mean():>14.3f}{tc['aNDCG'].mean():>13.3f}")
    tr = df[(df["arm"] == "random(200×)") & (df["variant"] == "clean") & (df["k"] == k)]
    print(f"  {'random(200×)':<18}{'—':>17}{tr['StRecall'].mean():>16.3f}{'—':>17}"
          f"{tr['setF1'].mean():>14.3f}{tr['aNDCG'].mean():>13.3f}")

print(f"\n{'=' * 118}\n【③】关键对照：配对 bootstrap 95% CI（n={nq} 个真值，指标 = StRecall@10）")
print(f"  查询集 = **clean**（无泄露）")
for a, b in (("B mq_max", "A base"), ("C mq_rr", "A base"), ("C mq_rr", "B mq_max"),
             ("D mmr λ=0.5", "A base"), ("E mq_mmr λ=0.5", "B mq_max")):
    A = pt[(pt["variant"] == "clean") & (pt["arm"] == a)].sort_values(["cluster", "facet"])
    B = pt[(pt["variant"] == "clean") & (pt["arm"] == b)].sort_values(["cluster", "facet"])
    m = A.merge(B, on=["cluster", "facet"], suffixes=("_a", "_b"))
    if len(m) < 5:
        continue
    d, lo, hi = paired_ci(m["StRecall_a"].values, m["StRecall_b"].values)
    sig = "✅ 显著" if (lo > 0 or hi < 0) else "· 不显著"
    print(f"  {a:<16} − {b:<14} Δ {d:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

print(f"\n{'=' * 118}\n【④】结论")
base_c = df[(df["arm"] == "A base") & (df["variant"] == "clean") & (df["k"] == 10)]["StRecall"].mean()
b_c = df[(df["arm"] == "B mq_max") & (df["variant"] == "clean") & (df["k"] == 10)]["StRecall"].mean()
base_x = df[(df["arm"] == "A base") & (df["variant"] == "contam") & (df["k"] == 10)]["StRecall"].mean()
b_x = df[(df["arm"] == "B mq_max") & (df["variant"] == "contam") & (df["k"] == 10)]["StRecall"].mean()
print(f"  · 泄露影响：A base StRecall@10 contam {base_x:.3f} → clean {base_c:.3f}"
      f"（**{base_c - base_x:+.3f}**）")
print(f"  · 多查询增益：clean 下 B mq_max − A base = **{b_c - base_c:+.3f}**"
      f"（contam 下为 {b_x - base_x:+.3f}）")
print(f"  · → {'多查询在干净口径下仍有增益' if b_c > base_c else '多查询的增益是泄露产物，干净口径下不成立'}")
print(f"\n已写 {OUT / 'R2_PHASE1_CLEAN.csv'}（逐真值逐臂明细）")
