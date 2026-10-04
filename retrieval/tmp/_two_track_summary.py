"""**两条线汇总表**：① QAMPARI（维基语料，官方可比指标）② 论文语料（R2，最终校准真值）

产出可直接贴的对照表，含：
  · QAMPARI：官方 em/coverage/subspan_em + ALCE 官方 P/R + 论文 Table 2 参照线 + 闭卷对照
  · 论文语料：MRecall/StRecall/集合 P/F1/α-nDCG + 随机基线 + oracle 上界 + **交付 k 的权衡**
"""
from __future__ import annotations

import importlib.util as _iu
import json
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
RES = HERE / "results"
CACHE = HERE / "data" / "r2dev" / "clusters"
SUBQ = HERE / "data" / "r2dev" / "subqueries.json"
RECAL = HERE / "data" / "r2dev" / "gold_recalib.csv"
RICH = HERE / "data" / "r2dev" / "gold_recalib_rich.csv"
CHUNK, OVERLAP, POOL = 1000, 100, 40

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf, alpha_ndcg = _m.mrecall, _m.strecall, _m.setpf, _m.alpha_ndcg

# ══════════════ ① QAMPARI ══════════════
print("#" * 118)
print("① QAMPARI（语料 = 维基百科，LoFT 官方 128k test，100 题）｜运行 `exh_k40_k40`")
print("#" * 118)
o = pd.read_csv(RES / "R2_QAMPARI_OFFICIAL_ALLRUNS.csv")
p = pd.read_csv(RES / "R2_QAMPARI_PRECISION_ALLRUNS.csv")
f = pd.read_csv(RES / "R2_QAMPARI_AUDIT_FINAL.csv")
r = o[o["运行"] == "exh_k40_k40"].iloc[0]
pr = p[p["运行"] == "exh_k40_k40"].iloc[0]
fa = f[f["运行"] == "exh_k40_k40"].iloc[0]
cb = pd.read_csv(RES / "R2_CONTAM_P1.csv")

PAPER = {"专用 RAG pipeline": 0.55, "Gemini 1.5 Pro（直读）": 0.44,
         "GPT-4o（直读）": 0.27, "Claude（直读）": 0.25}
rows = [
    ("**subspan_em**（LoFT 主指标）", f"{r['官方subspan_em']:.3f}", "官方 `evaluation/rag.py`"),
    ("em（集合完全一致）", f"{r['官方em']:.3f}", "官方"),
    ("coverage（官方报道口径）", f"{r['官方coverage']:.4f}", f"官方；分母剔掉空预测（空预测 {int(fa['空预测题'])} 题）"),
    ("coverage（全行口径）", f"{fa['coverage_全行']:.4f}", "空预测记 0 → 跨运行可比、更保守"),
    ("**precision**（ALCE 官方算术）", f"{pr['ALCE_P']:.4f}", "HELMET/ALCE `eval_alce.py` 逐字移植"),
    ("recall（ALCE 官方算术）", f"{pr['ALCE_R']:.4f}", "同上"),
    ("F1（ALCE 官方算术）", f"{pr['ALCE_F1']:.4f}", "同上"),
    ("recall@5（ALCE 官方）", f"{pr['ALCE_R5']:.4f}", "同上"),
    ("平均预测答案数 / gold", f"{pr['预测均']:.2f} / {pr['gold均']:.2f}", f"比值 {pr['比值']:.3f} → **无 recall 灌水**"),
]
print(f"\n  {'指标':<32}{'我们':>10}   {'口径'}")
for a, b, c in rows:
    print(f"  {a:<32}{b:>10}   {c}")

print(f"\n  【外部参照（论文 Table 2，128k test；**别的模型**）】")
for k, v in PAPER.items():
    d = r["官方subspan_em"] - v
    print(f"    {k:<24}{v:>6.2f}   我们 {r['官方subspan_em']:.3f} → **{d:+.3f}**")

print(f"\n  【污染/闭卷对照（无文档直答，100 题）】")
print(f"    {'':<24}{'闭卷':>10}{'带检索':>10}{'检索净增益':>12}")
for nm, a, b in (("subspan_em", cb["subspan_em"].mean(), r["官方subspan_em"]),
                 ("coverage", cb["coverage"].mean(), fa["coverage_全行"]),
                 ("em", cb["em"].mean(), r["官方em"])):
    print(f"    {nm:<24}{a:>10.3f}{b:>10.3f}{b - a:>+12.3f}")

# ══════════════ ② 论文语料 ══════════════
print("\n" + "#" * 118)
print("② 论文语料（R2：**用户自带 20 篇/题**，3 簇 × 20 篇；真值 = 双 LLM 校准后的最终真值）")
print("#" * 118)
rec = pd.read_csv(RECAL)
rec["anchor_gold"] = rec["anchor_gold"].astype(bool)
rec["new_gold"] = rec["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
rec["final_gold"] = rec["new_gold"]
rich = pd.read_csv(RICH)
rich["yes"] = rich["new_gold_rich"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
mp = {(int(x.cluster), str(x.facet), str(x.docid)): bool(x.yes) for x in rich.itertuples()}
for i, x in rec.iterrows():
    kk = (int(x["cluster"]), str(x["facet"]), str(x["docid"]))
    if kk in mp:
        rec.at[i, "final_gold"] = mp[kk]
print(f"  真值校准：锚点 {rec['anchor_gold'].mean() * 20:.1f} 篇/题 → 最终 {rec['final_gold'].mean() * 20:.1f} 篇/题"
      f"（全 {(rec.groupby(['cluster', 'facet']).ngroups)} 个真值口径）")

subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
enc.max_seq_length = 512
if dev == "cuda":
    enc.half()


def zn(v):
    return (v - v.mean()) / (v.std() + 1e-9)


KW = [5, 10, 20]
rows2, per_facet = [], []
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
    for facet in meta[ci]["usable"]:
        g = rec[(rec["cluster"] == ci + 1) & (rec["facet"] == facet)]
        if not len(g):
            continue
        gold = set(g[g["final_gold"]]["docid"])
        if not gold or len(gold) == len(docs):
            continue
        qs = [F[facet][1]] + list(subq_all.get(facet, []))
        S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True
                       ).astype(np.float32) @ C.T
        Smax = S.max(axis=0)
        rankB = sorted(docs, key=lambda d: -Smax[idx[d]].max())
        rankA = sorted(docs, key=lambda d: -S[0][idx[d]].max())
        rng = np.random.default_rng(0)
        rperm = [rng.permutation(docs).tolist() for _ in range(200)]
        rorac = sorted(docs, key=lambda d: (d not in gold, d))
        arms = {"B mq_max（多查询）": rankB, "A base（单查询）": rankA,
                "random（200×）": None, "oracle（上界）": rorac}
        for nm, rk in arms.items():
            for k in KW:
                if nm.startswith("random"):
                    pv = [setpf(x, gold, k) for x in rperm]
                    st = float(np.mean([strecall(x, gold, k) for x in rperm]))
                    mr = float(np.mean([mrecall(x, gold, k) for x in rperm]))
                    an = float(np.mean([alpha_ndcg(x, gold, k) for x in rperm]))
                    pp, rr, ff = (float(np.mean([x[0] for x in pv])),
                                  float(np.mean([x[1] for x in pv])),
                                  float(np.mean([x[2] for x in pv])))
                else:
                    pp, rr, ff = setpf(rk, gold, k)
                    st, mr, an = strecall(rk, gold, k), mrecall(rk, gold, k), alpha_ndcg(rk, gold, k)
                rows2.append(dict(cluster=ci + 1, facet=facet, arm=nm, k=k, n_gold=len(gold),
                                  MRecall=mr, StRecall=st, setP=pp, setF1=ff, aNDCG=an))
        # 交付 k 曲线（最佳臂）
        for k in range(1, 21):
            pp, rr, ff = setpf(rankB, gold, k)
            rows2.append(dict(cluster=ci + 1, facet=facet, arm="B mq_max（多查询）", k=k,
                              n_gold=len(gold), MRecall=mrecall(rankB, gold, k),
                              StRecall=strecall(rankB, gold, k), setP=pp, setF1=ff,
                              aNDCG=alpha_ndcg(rankB, gold, k)))
        per_facet.append(dict(cluster=ci + 1, facet=facet, n_gold=len(gold),
                              StRecall10=strecall(rankB, gold, 10),
                              setF1_10=setpf(rankB, gold, 10)[2],
                              setP_10=setpf(rankB, gold, 10)[0]))
    print(f"  簇{ci + 1} 完成（{time.time() - t0:.0f}s）", flush=True)

d2 = pd.DataFrame(rows2)
d2.to_csv(RES / "R2_SETS_TRACK_SUMMARY.csv", index=False, encoding="utf-8")
nq = d2.groupby(["cluster", "facet"]).ngroups
gmean = d2.groupby("facet")["n_gold"].first().mean()
print(f"\n  可用题 {nq} ｜ 平均 gold **{gmean:.1f}/20 篇**")

print(f"\n  {'臂':<20}{'k':>3}{'MRecall':>9}{'StRecall':>10}{'集合P':>8}{'集合F1':>8}{'α-nDCG':>9}")
for arm in ("A base（单查询）", "B mq_max（多查询）", "random（200×）", "oracle（上界）"):
    for k in KW:
        t = d2[(d2["arm"] == arm) & (d2["k"] == k)]
        if not len(t):
            continue
        print(f"  {arm:<20}{k:>3}{t['MRecall'].mean():>9.3f}{t['StRecall'].mean():>10.3f}"
              f"{t['setP'].mean():>8.3f}{t['setF1'].mean():>8.3f}{t['aNDCG'].mean():>9.3f}")

print(f"\n  【交付 k 的权衡（最佳臂 B mq_max）——产品直接相关】")
print(f"  {'交付 k':>6}{'集合P':>9}{'集合R(=StRecall)':>17}{'集合F1':>9}  说明")
best = d2[(d2["arm"] == "B mq_max（多查询）")].groupby("k")["setF1"].mean()
kb = int(best.idxmax())
for k in range(1, 21):
    t = d2[(d2["arm"] == "B mq_max（多查询）") & (d2["k"] == k)]
    note = ""
    if k == kb:
        note = "← **F1 最优**"
    if k == 20:
        note = "（=全给，P 锁死在 gold 比例）"
    print(f"  {k:>6}{t['setP'].mean():>9.3f}{t['StRecall'].mean():>17.3f}{t['setF1'].mean():>9.3f}  {note}")

print(f"\n  【逐 facet（最佳臂，k=10）】")
pf = pd.DataFrame(per_facet).sort_values("StRecall10", ascending=False)
print(f"  {'簇':>3}{'facet':<18}{'gold数':>7}{'StRecall@10':>12}{'集合P':>8}{'集合F1':>8}")
for _, x in pf.iterrows():
    print(f"  {int(x['cluster']):>3}{x['facet']:<18}{int(x['n_gold']):>7}"
          f"{x['StRecall10']:>12.3f}{x['setP_10']:>8.3f}{x['setF1_10']:>8.3f}")
pf.to_csv(RES / "R2_SETS_TRACK_PER_FACET.csv", index=False, encoding="utf-8")
print(f"\n已写 {RES / 'R2_SETS_TRACK_SUMMARY.csv'} ／ R2_SETS_TRACK_PER_FACET.csv")
