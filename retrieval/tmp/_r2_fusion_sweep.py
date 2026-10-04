"""**融合方式 × 权重 × k 的联合扫描**（50 篇口径，零 LLM）

## 用户口径
**只用两条通道**：`dense`(bge-m3) + `bm25`(ASCII 字面)。**不要 `sparse`**。
要回答：**三种融合方式 + 权重 + 交付 k，哪个配置最优？**

## ★ 先厘清三件会让结论出错的事

### ① `k` 和"融合方式"是**两种不同的旋钮**（必须分开判读）
· 融合/权重决定**排序**（谁在前）
· `k` 决定**截断**（交付几篇）
→ 若目标函数是 `StRecall@k`，**k 越大必然越好**（k=库 时 =1.0）→ 用它选 k 是无意义的。
→ 必须**同时报 `setF1@k` / `MRecall@k`**，并按"交付预算"或"F1 最优"来定 k。

### ② 融合**层级**是隐藏的第四轴（生产在**块级**融合，我们的任务是**篇级**）
· `chunk`：块级融合 → 篇分 = 篇内块分 max（**生产 `search_hybrid` 的做法**）
· `doc`  ：先把每通道聚到篇，再融合篇排序（**任务原生**）
→ 两者可能给出不同结论，`--level` 都跑。

### ③ ⚠️ BM25 全零必须**跳过 BM25 路**
纯中文查询在英文语料上 BM25 恒 ~0，若仍参与融合，**名次会变成"索引序"**（伪位次）。
生产有 `_bm_or_none()` 正是防这个。本脚本同：某题 BM25 全零则该题只用 dense，并计数上报。

## 三种融合方式（都落到生产已有原语）
| 方式 | 生产出处 | 公式 |
|---|---|---|
| `rrf` | `embedder.rrf_order` / `pull_chunk._rrf_interleave_hits` | `Σ w_c/(60+rank_c)`（名次级） |
| `normsum` | （`_r2_recall_ceiling` 的 `union3` 是其等权版） | `z(s_d) + ρ·z(s_b)`（分数级，z 标准化） |
| `quota` | `query_optimizer.quota_union` | 主路取前 `q_d` **整块**拼接、其余各路依次接后（**不交错**） |

## 参数网格
· 权重 ρ = `w_bm25 / w_dense` ∈ {0（纯 dense）, 0.25, 0.5, 1, 2, 4, ∞（纯 bm25）}
· `quota` 的配额由 ρ 与 k 推出：`q_d = round(k/(1+ρ))`, `q_b = k − q_d`（保证三者**交付篇数相同**，可比）
· k ∈ {3, 5, 8, 10, 13, 15, 20, 25, 30}
· 查询口径 `Q_prod`（1 条英文检索式，≈生产）/ `Q_dev`（3 条英文子查询，偏富）

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fusion_sweep.py --corpus 50 --level both
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
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
RRF_K = 60                      # RRF 常量（生产默认 60，实测 k=10/20/60 打平）


def _load(name: str, path: Path):
    spec = _iu.spec_from_file_location(name, path)
    mod = _iu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("stdm", HERE / "tmp" / "_r2_std_metrics.py")
F2 = _load("facets2", HERE / "tmp" / "_r2_facets_v2.py").F2

# 配置：(标签, ρ)  ρ = w_bm25 / w_dense；None = 纯 bm25
# ⚠️ 标签写的是 **dense : bm25 的比例**：`4:1` ⇒ ρ=0.25（dense 重 4 倍）
CFGS = [("dense", 0.0), ("4:1", 0.25), ("2:1", 0.5), ("1:1", 1.0),
        ("1:2", 2.0), ("1:4", 4.0), ("bm25", None)]
KS = [3, 5, 8, 10, 13, 15, 20, 25, 30]


def zscore(v: np.ndarray) -> np.ndarray:
    s = float(v.std())
    return (v - float(v.mean())) / s if s > 1e-9 else np.zeros_like(v)


def minmax(v: np.ndarray) -> np.ndarray:
    """min-max 归一化（分数融合的另一种常用做法）。"""
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(v)


def metrics_at(order: list[str], gold: set[str], k: int) -> dict:
    p, r, f1 = _m.setpf(order, gold, k)
    return dict(StRecall=_m.strecall(order, gold, k), setP=p, setF1=f1,
                MRecall=_m.mrecall(order, gold, k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="50", choices=["20", "50"])
    ap.add_argument("--level", default="both", choices=["doc", "chunk", "both"])
    ap.add_argument("--chunktag", default="mineru")
    ap.add_argument("--qmodes", default="Q_prod,Q_dev")
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
    levels = ["doc", "chunk"] if args.level == "both" else [args.level]
    qmodes = [x for x in args.qmodes.split(",") if x]

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 8192
    if dev == "cuda":
        enc.half()
    print("=" * 124)
    print(f"【融合 × 权重 × k 联合扫描】语料 **{args.corpus}**（{CDIR.name}）｜ 真值 {GF.name}"
          f" ｜ 通道 **dense + bm25**（不含 sparse）｜ 层级 {levels} ｜ 查询 {qmodes}")
    print(f"  RRF_K={RRF_K} ｜ 权重 ρ=w_bm25/w_dense ∈ "
          f"{[c[0] for c in CFGS]} ｜ k ∈ {KS}")

    rows = []
    n_bm_skip = 0
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
        idx = {d: np.where(owner == d)[0] for d in docs}
        E = enc.encode(chunks, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = _m.BM25(chunks, tok=_m._tok)          # ASCII 字面 BM25（英文查询可用）
        print(f"  簇{ci + 1}：{len(docs)} 篇 / {len(chunks)} 块 ｜ 编码 {time.time() - t0:.0f}s",
              flush=True)

        for facet in sorted(x for (c2, x) in combos if c2 == ci + 1):
            gold = (NG.get((ci + 1, facet)) or set()) & set(docs)
            if not gold or len(gold) == len(docs):
                continue
            _, zh, _anchor, para = F2[facet]
            SQ = list(subq.get(facet, []))
            QS = {"Q_prod": [zh, para], "Q_dev": [zh] + SQ}
            for qn in qmodes:
                qlist = [q for q in QS[qn] if str(q).strip()]
                S_d = (enc.encode(qlist, normalize_embeddings=True, show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32) @ E.T).max(axis=0)
                # ⚠️ BM25 只用**英文**查询（含 CJK 的整条丢掉）；全零则整题跳过 BM25 路
                eq = [q for q in qlist
                      if not any("\u4e00" <= c <= "\u9fff" for c in str(q))] or qlist
                S_b = np.max([bm.scores(q) for q in eq], axis=0).astype(np.float32)
                bm_ok = float(S_b.max()) > 0
                if not bm_ok:
                    n_bm_skip += 1
                # 通道内块名次
                r_d = np.empty(len(S_d), dtype=np.int64)
                r_d[np.argsort(-S_d)] = np.arange(len(S_d))
                r_b = (np.empty(len(S_b), dtype=np.int64)
                       if bm_ok else np.full(len(S_b), len(S_b)))
                if bm_ok:
                    r_b[np.argsort(-S_b)] = np.arange(len(S_b))
                # 篇级通道分（篇内 max）
                dm_d = np.array([float(S_d[idx[d]].max()) for d in docs])
                dm_b = np.array([float(S_b[idx[d]].max()) for d in docs]) if bm_ok \
                    else np.zeros(len(docs))
                ord_d = [docs[i] for i in np.argsort(-dm_d)]
                ord_b = [docs[i] for i in np.argsort(-dm_b)] if bm_ok else list(ord_d)

                # ★ 两通道**并集上界**（oracle）：dense top-k ∪ bm25 top-k 里挑最优 k 个
                #    → 任何"k 限交付"方法（含任何融合/重排）在**这两条通道内**的召回上界。
                #    ⚠️ 这比 `R2_RECALL_CEILING_50` 的三通道上界**低**（少一条通道）。
                for k in KS:
                    if k > len(docs):
                        continue
                    uni = list(dict.fromkeys(ord_d[:k] + ord_b[:k]))
                    tp = len(set(uni) & gold)
                    bk = min(tp, k)
                    p = bk / k
                    r = bk / len(gold)
                    rows.append(dict(
                        cluster=ci + 1, facet=facet, qmode=qn, level="union2",
                        cfg="dense+bm25", method="oracle2", k=k,
                        n_gold=len(gold), n_docs=len(docs),
                        gold_rate=len(gold) / len(docs), bm25_ok=int(bm_ok),
                        StRecall=r, setP=p,
                        setF1=(2 * p * r / (p + r) if (p + r) else 0.0),
                        MRecall=float(tp >= min(len(gold), k))))

                for lvl in levels:
                    for lab, rho in CFGS:
                        # ── 统一产出：三方法 × 该权重（见下方 helper）──
                        for meth, o in _orders(rho, lvl, docs, ord_d, ord_b,
                                               r_d, r_b, idx, S_d, S_b, dm_d, dm_b,
                                               bm_ok, RRF_K).items():
                            for k in KS:
                                if k > len(docs):
                                    continue
                                rec = dict(cluster=ci + 1, facet=facet, qmode=qn,
                                           level=lvl, cfg=lab, method=meth, k=k,
                                           n_gold=len(gold), n_docs=len(docs),
                                           gold_rate=len(gold) / len(docs),
                                           bm25_ok=int(bm_ok))
                                rec.update(metrics_at(o, gold, k))
                                rows.append(rec)

    d = pd.DataFrame(rows)
    suf = f"_n{args.corpus}"
    d.to_csv(HERE / "results" / f"R2_FUSION{suf}.csv", index=False, encoding="utf-8-sig")
    print(f"\n  完成 {len(rows)} 行（{time.time() - t0:.0f}s）"
          f" ｜ ⚠️ BM25 全零跳过的 (题,口径) 数 = {n_bm_skip}")
    print(f"  → results/R2_FUSION{suf}.csv")
    return 0


def _orders(rho, lvl: str, docs, ord_d, ord_b, r_d, r_b, idx,
            S_d, S_b, dm_d, dm_b, bm_ok: bool, rrf_k: int) -> dict[str, list[str]]:
    """按 (方法) 产出篇级排序（同一 ρ 下三方法**交付篇数相同**，可比）。

    · `rrf`     ：名次级 → `w_d/(rrf_k+rank_d) + w_b/(rrf_k+rank_b)`
                  （`lvl="chunk"` 用**块内最好名次**，`lvl="doc"` 用篇级名次）
    · `normsum` ：分数级 → `w_d·z(s_d) + w_b·z(s_b)`（z 标准化，消除量纲）
    · `quota`   ：主路(dense) 取前 `q_d` **整块**拼接、副路(bm25) 取前 `q_b` 接后，**不交错**
                  → `q_d = round(N/(1+ρ))`、`q_b = N − q_d`（N=库大小）
    """
    wb = 0.0 if rho is None else float(rho)
    wd = 0.0 if rho is None else 1.0
    if rho is None and not bm_ok:       # 纯 bm25 但通道失效 → 退化 dense
        return {"rrf": list(ord_d), "normsum": list(ord_d), "quota": list(ord_d)}
    n = len(docs)
    if lvl == "doc":
        rank_d = {d: i for i, d in enumerate(ord_d)}
        rank_b = {d: i for i, d in enumerate(ord_b)}
        zd, zb = zscore(dm_d), (zscore(dm_b) if bm_ok else np.zeros(n))
        md, mb = minmax(dm_d), (minmax(dm_b) if bm_ok else np.zeros(n))
        zmap_d = {d: float(zd[i]) for i, d in enumerate(docs)}
        zmap_b = {d: float(zb[i]) for i, d in enumerate(docs)}
        mmap_d = {d: float(md[i]) for i, d in enumerate(docs)}
        mmap_b = {d: float(mb[i]) for i, d in enumerate(docs)}
    else:
        rank_d = {d: int(r_d[idx[d]].min()) for d in docs}
        rank_b = {d: int(r_b[idx[d]].min()) for d in docs}
        zac_d, zac_b = zscore(S_d), (zscore(S_b) if bm_ok else np.zeros_like(S_b))
        mnc_d, mnc_b = minmax(S_d), (minmax(S_b) if bm_ok else np.zeros_like(S_b))
        zmap_d = {d: float(zac_d[idx[d]].max()) for d in docs}
        zmap_b = {d: float(zac_b[idx[d]].max()) for d in docs}
        mmap_d = {d: float(mnc_d[idx[d]].max()) for d in docs}
        mmap_b = {d: float(mnc_b[idx[d]].max()) for d in docs}
    return {
        # ★ 排序融合（名次级，无量纲假设）
        "rrf": sorted(docs, key=lambda d: -(wd / (rrf_k + rank_d[d] + 1)
                                           + wb / (rrf_k + rank_b[d] + 1))),
        # ★ 打分融合（分数级）两种归一化
        "normsum_z": sorted(docs, key=lambda d: -(wd * zmap_d[d] + wb * zmap_b[d])),
        "normsum_mm": sorted(docs, key=lambda d: -(wd * mmap_d[d] + wb * mmap_b[d])),
    }


if __name__ == "__main__":
    raise SystemExit(main())
