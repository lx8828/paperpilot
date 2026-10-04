"""**泄露审计**（用户要求）

① 索引与排序全程不用 gold evidence 段落 ID
② "每篇保底轮询"只在检索排序结果上做、不碰 gold 标注
③ 多查询 prompt 里只有问题本身

同时**量化**已发现的那个泄露点：
  `dense_zh_en` 的查询 = f"{zh} {anchor}"，而 `anchor`（英文锚点词）与 gold 正则**同词汇**
  → 设计实验：把锚点中**与 gold 正则重合的词删掉**，再测 → 若性能崩，说明该通道靠泄露得分。
"""
from __future__ import annotations

import importlib.util as _iu
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "data" / "r2dev" / "clusters"
CHUNK, OVERLAP = 1000, 100

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F


def toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", str(s).lower())}


def rx_toks(pat: str) -> set[str]:
    """正则里的词：先剥掉 `\\b` 这类转义，否则 `\\bablation` 会被切成 `bablation`（会低估重合）。"""
    return {t for t in re.findall(r"[a-z]{3,}", re.sub(r"\\[a-zA-Z]", " ", str(pat)).lower())}


print("=" * 112)
print("【① 静态证据：锚点词 与 gold 正则 的词汇重合度】")
print(f"  {'facet':<17}{'锚点词数':>9}{'正则词数':>9}{'重合':>6}{'重合/锚点':>10}"
      f"{'重合/正则':>10}  重合的词")
tot_o, tot_a, tot_p = 0, 0, 0
rows1 = []
for f, (pat, zh, anchor, para) in F.items():
    rxt = rx_toks(pat)
    ant = toks(anchor)
    pt = toks(para)
    ov = ant & rxt
    tot_o += len(ov); tot_a += len(ant); tot_p += len(rxt)
    rows1.append(dict(facet=f, n_anchor=len(ant), n_rx=len(rxt), n_ov=len(ov),
                      ov_in_anchor=len(ov) / max(1, len(ant)), ov_in_rx=len(ov) / max(1, len(rxt)),
                      overlap=sorted(ov), para_ov=len(pt & rxt)))
    print(f"  {f:<17}{len(ant):>9}{len(rxt):>9}{len(ov):>6}"
          f"{len(ov) / max(1, len(ant)):>10.2f}{len(ov) / max(1, len(rxt)):>10.2f}  {sorted(ov)}")
d1 = pd.DataFrame(rows1)
print(f"\n  **合计**：锚点词 {tot_a}，其中 {tot_o} 个（**{tot_o / tot_a:.0%}**）同时是 gold 正则词")
print(f"  对照：`para`（英文改写句，设计上排除锚点原词）与正则重合 "
      f"**{d1['para_ov'].sum()} 个**（应为 0 → 证明「改写句」这条路是干净的）")
print(f"  中文题面 `zh` 全是汉字 → 与英文正则零重合（干净）")

print("\n" + "=" * 112)
print("【② 动态实验：把锚点里与 gold 正则重合的词删掉，性能会不会崩】")
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
enc.max_seq_length = 512
if dev == "cuda":
    enc.half()


def auc(score: dict[str, float], gold: set[str]) -> float:
    pos = [v for k, v in score.items() if k in gold]
    neg = [v for k, v in score.items() if k not in gold]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


rows2 = []
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
    bm = _m.BM25(chunks)
    low = [c.lower() for c in chunks]
    idx = {d: np.where(owner == d)[0] for d in docs}
    for facet in meta[ci]["usable"]:
        pat, zh, anchor, para = F[facet]
        hit = np.array([_m.judge(c, facet) for c in low])
        gold = {d for d in docs if hit[idx[d]].any()}
        if not gold or len(gold) == len(docs):
            continue
        rxt = rx_toks(pat)
        kw = [t for t in anchor.split() if t.lower() not in rxt]
        desens = " ".join(kw)          # 删掉与正则重合的词后的"净化锚点"
        variants = {"zh  (干净)": zh, "zh+anchor (现行·污染)": f"{zh} {anchor}",
                    "anchor (最污染)": anchor, "anchor 净化后": desens}
        for nm, q in variants.items():
            if not q.strip():
                continue
            qv = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            s = C @ qv
            per = {d: float(s[idx[d]].max()) for d in docs}
            rows2.append(dict(cluster=ci + 1, facet=facet, variant=nm, auc=auc(per, gold),
                              n_gold=len(gold)))
        for nm, q in (("bm25_para (干净)", para), ("bm25_anchor (污染)", anchor)):
            s = bm.scores(q)
            per = {d: float(s[idx[d]].max()) for d in docs}
            rows2.append(dict(cluster=ci + 1, facet=facet, variant=nm, auc=auc(per, gold),
                              n_gold=len(gold)))
    print(f"  簇{ci + 1} 完成（{len(docs)} 篇 / {len(chunks)} 块）", flush=True)

d2 = pd.DataFrame(rows2)
print(f"\n  真值数 {d2.groupby(['cluster', 'facet']).ngroups} ｜ 篇级 AUC（0.5 = 瞎猜）")
print(f"  {'查询变体':<26}{'AUC 均值':>10}{'相对 zh 的 Δ':>14}{'判定':>10}")
zh_auc = d2[d2["variant"] == "zh  (干净)"]["auc"].mean()
for nm, g in d2.groupby("variant", sort=False):
    a = g["auc"].mean()
    tag = ""
    if "污染" in nm or nm.startswith("anchor"):
        tag = f"泄露 +{a - zh_auc:+.3f}" if a > zh_auc else "无优势"
    print(f"  {nm:<26}{a:>10.3f}{a - zh_auc:>+14.3f}{tag:>10}")
print("\n  → 读法：若 **anchor 净化后** 的 AUC 明显低于 **anchor（原样）**，"
      "说明那条通道的高分**主要来自与 gold 正则的重合词**。")

print("\n" + "=" * 112)
print("【③ 第 2 层结论在干净打分下是否还成立】（轮询 vs 全局 top-B，打分 = zh 干净 vs zh_en 污染）")
rows3 = []
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
    for facet in meta[ci]["usable"]:
        pat, zh, anchor, para = F[facet]
        hit = np.array([_m.judge(c, facet) for c in low])
        gold = {d for d in docs if hit[idx[d]].any()}
        if not gold or len(gold) == len(docs):
            continue
        for sc_nm, q in (("zh 干净", zh), ("zh_en 污染", f"{zh} {anchor}")):
            qv = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            s = C @ qv
            for b in (1, 3, 6, 12):
                B = min(b * len(docs), len(chunks))
                gl = np.argsort(-s)[:B]
                per = {d: idx[d][np.argsort(-s[idx[d]])] for d in docs}
                rr = np.array([per[d][rnd] for rnd in range(b) for d in docs if rnd < len(per[d])][:B])
                for strat, sel in (("global_topB", gl), ("round_robin", rr)):
                    used = list(dict.fromkeys(owner[sel].tolist()))
                    rows3.append(dict(cluster=ci + 1, facet=facet, scorer=sc_nm, budget=b,
                                      strategy=strat, n_used=len(used),
                                      cover=len(set(used) & gold) / len(gold),
                                      ev_recall=float(hit[sel].sum() / max(1, hit.sum()))))

d3 = pd.DataFrame(rows3)
print(f"  {'打分':<12}{'预算B':>6}{'策略':<14}{'涉及篇':>7}{'覆盖率':>8}{'证据召回':>9}")
for sc in ("zh 干净", "zh_en 污染"):
    for b in (1, 3, 6, 12):
        for st in ("global_topB", "round_robin"):
            t = d3[(d3["scorer"] == sc) & (d3["budget"] == b) & (d3["strategy"] == st)]
            print(f"  {sc:<12}{b:>6}{st:<14}{t['n_used'].mean():>7.1f}"
                  f"{t['cover'].mean():>8.3f}{t['ev_recall'].mean():>9.3f}")
    print()

print("【结论对照：'全局 top-B 证据召回更好' 在两种打分下都成立吗】")
for sc in ("zh 干净", "zh_en 污染"):
    t = d3[d3["scorer"] == sc]
    for b in (3, 12):
        g = t[(t["budget"] == b) & (t["strategy"] == "global_topB")]["ev_recall"].mean()
        r = t[(t["budget"] == b) & (t["strategy"] == "round_robin")]["ev_recall"].mean()
        print(f"  {sc:<12} B={b:<3} 全局 {g:.3f} vs 轮询 {r:.3f} → 全局{'更好' if g > r else '更差'}"
              f"（Δ {g - r:+.3f}）")
d1.to_csv(HERE / "results" / "R2_LEAK_STATIC.csv", index=False, encoding="utf-8")
d2.to_csv(HERE / "results" / "R2_LEAK_AUC.csv", index=False, encoding="utf-8")
d3.to_csv(HERE / "results" / "R2_LEAK_L2.csv", index=False, encoding="utf-8")
print(f"\n已写 R2_LEAK_STATIC.csv / R2_LEAK_AUC.csv / R2_LEAK_L2.csv")
