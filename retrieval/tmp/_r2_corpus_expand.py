"""**扩语料：每簇补到 50 篇「可下载」**（同领域干扰项，取自 RAG-1 的 arxiv 池，零 LLM）

## 为什么用 RAG-1 的 arxiv 池，而不是 LitSearch 池
| 池 | 规模 | 全文 | 有 arxiv 号 | 可下载 → 可 MinerU |
|---|---|---|---|---|
| LitSearch `corpus_clean` | 64,183 | ✅ 全文 | **仅 24.5%**（15,697） | ❌ 不可靠 |
| **RAG-1 `data/arxiv`** ★ | **540,052** | title+abstract（1,374 字符） | **100%** | ✅ **全部** |
→ 下一步要 MinerU，**必须保证每篇都能下到 PDF** → 只能用 arxiv 池。

## 基准篇数
现有 60 篇里，**只有 47 篇拿到了 PDF**（16/18/13 篇/簇；13 篇无 arXiv/开放 PDF）。
→ 补齐口径 = **每簇「可下载」篇数到 50** → 需新增 **34 / 32 / 37 = 103 篇**。

## 干扰项怎么选（复用现有检索工具）
· 索引 = `data/arxiv/emb/`（540k × 1024，bge-m3，现成）
· 查询 = 该簇现有论文 `title + abstract`（bge-m3 @512，与索引同口径）的**质心**
· 归属 = 三质心 argmax → 每簇取 top-N（按该簇质心相似度）
· 排除 = 现有 60 篇（arxiv 号 + 标题归一化双重）+ 跨簇重复

## 产物
· `data/r2dev/corpus50/c{0,1,2}.parquet`：现有 20 篇（docid 不变）+ 新增 `C{i}D{j}`（含 `arxiv_id`）
· `data/r2dev/corpus50/expand_report.json`：新篇清单 + 相似度，供抽检

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_corpus_expand.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
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
CLUSTERS = DEV / "clusters"
ARX = HERE / "data" / "arxiv"
OUT = DEV / "corpus50"

N_TARGET = 50                 # ★ 每簇目标「可下载」篇数
MIN_ABS = 300                 # 摘要太短的不要（检索质量）
POOL_MULT = 8                 # 冗余池倍数（供去重）


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())[:80]


def base_id(a: str) -> str:
    """arxiv 号去版本：`2211.07067v1` → `2211.07067`。"""
    return re.sub(r"v\d+$", "", str(a or "").strip())


def main() -> int:
    print("=" * 116)
    print(f"【扩语料】每簇补到 {N_TARGET} 篇「可下载」（干扰项取自 RAG-1 arxiv 池）")

    # ── 现有语料 + 可下载判定 ──
    arx = json.loads((DEV / "arxiv_map.json").read_text(encoding="utf-8"))
    pmp = json.loads((DEV / "pdf_map.json").read_text(encoding="utf-8"))
    okd = {k for k, v in pmp.items() if v.get("ok")}
    old: dict[int, pd.DataFrame] = {}
    need: dict[int, int] = {}
    for ci in range(3):
        d = pd.read_parquet(CLUSTERS / f"c{ci}.parquet")
        old[ci] = d
        avail = [x for x in d["docid"] if x in okd]
        need[ci] = N_TARGET - len(avail)
        print(f"  簇{ci + 1}: 现有 {len(d)} 篇 ｜ 可下载 {len(avail)} 篇 → **需补 {need[ci]} 篇**")
    print(f"  合计需新增 **{sum(need.values())}** 篇（总语料 {sum(len(v) for v in old.values())} "
          f"→ {len(okd) + sum(need.values())} 篇可用）")

    # ── arxiv 池元数据 ──
    pool = pd.read_parquet(ARX / "meta" / "corpus_text.parquet",
                           columns=["docid", "arxiv_id", "title", "abstract", "n_chars"])
    pool["aid"] = pool["arxiv_id"].astype(str).map(base_id)
    pool["nt"] = pool["title"].astype(str).map(norm_title)
    print(f"  池 {len(pool):,} 篇 ｜ 摘要 ≥{MIN_ABS} 字符 {(pool.n_chars >= MIN_ABS).mean():.1%}")

    used_aid = {base_id(arx[d].get("arxiv")) for d in arx if base_id(arx[d].get("arxiv"))}
    used_nt = {norm_title(t) for d in old.values() for t in d["title"]}
    print(f"  排除：现有 arxiv 号 {len(used_aid)} 个 ｜ 标题 {len(used_nt)} 个")

    # ⚠️ `clusters/*.parquet` **没有 abstract 列** → 从 LitSearch 池按 corpusid 补
    #    （60 篇本来就是从该池选出的，都能命中）
    ct = pd.read_parquet(HERE / "data" / "litsearch" / "derived" / "corpus_text.parquet",
                         columns=["corpusid", "abstract"])
    ct = ct.drop_duplicates("corpusid").set_index("corpusid")["abstract"].astype(str).to_dict()
    for ci in range(3):
        old[ci]["abstract"] = [str(ct.get(int(c), "")) for c in old[ci]["corpusid"]]
        n_ok = sum(1 for a in old[ci]["abstract"] if a.strip())
        print(f"    簇{ci + 1} 摘要补齐 {n_ok}/{len(old[ci])}")

    # ── 查询：该簇现有论文 title+abstract 的质心 ──
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512                      # ⚠️ 与索引口径一致（池是 512）
    if dev == "cuda":
        enc.half()
    cent = []
    for ci in range(3):
        tx = [f"{t} {a}" for t, a in zip(old[ci]["title"], old[ci]["abstract"])]
        V = enc.encode(tx, batch_size=16, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        v = V.mean(axis=0)
        cent.append(v / (np.linalg.norm(v) + 1e-9))
    Q = np.stack(cent).astype(np.float32)
    print(f"  查询向量（{dev}，512 口径）：3 个簇质心 ✓")

    # ── 分片打分（mmap 逐片，避免 2.2GB 常驻）──
    # 一次遍历同时记录：全局 argmax 归属（int8）与最大相似度（float32）
    ids = np.load(ARX / "emb" / "docid.npy")
    parts = sorted(glob.glob(str(ARX / "emb" / "part_*.npy")))
    N = int(len(ids))
    ARG = np.empty(N, dtype=np.int8)
    MX = np.empty(N, dtype=np.float32)
    K = int(max(need.values()) * POOL_MULT)
    top: dict[int, tuple[np.ndarray, np.ndarray]] = {c: (np.array([], dtype=np.int64),
                                                         np.array([], dtype=np.float32))
                                                     for c in range(3)}
    off = 0
    for p in parts:
        M = np.load(p, mmap_mode="r")
        S = np.asarray(M, dtype=np.float32) @ Q.T              # n × 3
        ARG[off:off + len(S)] = S.argmax(axis=1).astype(np.int8)
        MX[off:off + len(S)] = S.max(axis=1)
        for ci in range(3):
            s = S[:, ci]
            k = min(K, len(s))
            loc = (np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s)))
            allg = np.concatenate([top[ci][0], loc + off])
            alls = np.concatenate([top[ci][1], s[loc]])
            o = np.argsort(-alls)[:K]
            top[ci] = (allg[o], alls[o])
        off += len(S)
    assert off == N, f"分片合计 {off} ≠ docid {N}"
    print(f"  打分完成：{len(parts)} 片 / {off:,} 篇 × 3 质心 ｜ "
          f"归属分布 {np.bincount(ARG, minlength=3).tolist()}")

    # ── 挑选：每簇在「归属本簇」的候选里按相似度取 top-N ──
    is_dup = pool["nt"].duplicated(keep="first").to_numpy()
    picks: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    seen_nt = set(used_nt)
    i2r = pool.reset_index(drop=True)
    for ci in range(3):
        g, s = top[ci]
        for gi, ss in zip(g, s):
            if len(picks[ci]) >= need[ci]:
                break
            gi = int(gi)
            if ARG[gi] != ci:                     # 归属校验（全局 argmax）
                continue
            r = i2r.iloc[gi]
            if r["aid"] in used_aid:
                continue
            if is_dup[gi]:
                continue
            nt = str(r["nt"])
            if nt in seen_nt or float(r["n_chars"]) < MIN_ABS:
                continue
            seen_nt.add(nt)
            picks[ci].append(dict(docid=f"C{ci + 1}D{len(picks[ci]) + 1}",
                                  corpusid=int(r["docid"]), arxiv_id=str(r["arxiv_id"]),
                                  title=str(r["title"]), abstract=str(r["abstract"]),
                                  n_abs=int(r["n_chars"]), sim=float(ss)))
    short = {ci: len(picks[ci]) for ci in range(3) if len(picks[ci]) < need[ci]}
    if short:
        print(f"  ⚠️ 不足额：{short}（归属约束下候选不够，可放宽 POOL_MULT/ARG 校验）")

    # ── 写产物 ──
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict[str, object] = {
        "n_target": N_TARGET, "base": "现有可下载（有 PDF）篇数",
        "source": "RAG-1 arxiv 池（data/arxiv，540k，全带 arxiv_id → 可下载）",
        "method": "索引=arxiv/emb（bge-m3,512）；查询=该簇现有论文 title+abstract 质心；argmax 归属",
        "clusters": [],
    }
    print(f"\n  {'簇':>3}{'原有':>5}{'可下载':>7}{'新增':>5}{'合计可用':>9}"
          f"{'新篇 sim 中位':>14}{'摘要字符中位':>13}")
    for ci in range(3):
        p = picks[ci]
        add = pd.DataFrame(p)[["docid", "corpusid", "arxiv_id", "title", "abstract"]]
        add["src"] = "distractor"
        add["cluster"] = ci + 1
        keepold = old[ci].copy()
        keepold["arxiv_id"] = [base_id(arx[d].get("arxiv")) for d in keepold["docid"]]
        keepold["cluster"] = ci + 1
        keepold["src"] = "original"
        full = pd.concat([keepold[["docid", "corpusid", "arxiv_id", "title", "abstract",
                                   "cluster", "src"]], add], ignore_index=True)
        full.to_parquet(OUT / f"c{ci}.parquet", index=False)
        s = np.array([x["sim"] for x in p]) if p else np.array([0.0])
        print(f"  {ci + 1:>3}{len(old[ci]):>5}{len([x for x in old[ci]['docid'] if x in okd]):>7}"
              f"{len(p):>5}{len([x for x in old[ci]['docid'] if x in okd]) + len(p):>9}"
              f"{np.median(s):>14.3f}{np.median([x['n_abs'] for x in p]) if p else 0:>13,.0f}")
        rep["clusters"].append({
            "cluster": ci + 1, "n_original": len(old[ci]),
            "n_downloadable_now": len([x for x in old[ci]["docid"] if x in okd]),
            "n_added": len(p), "sim_median": round(float(np.median(s)), 4),
            "sim_min": round(float(s.min()), 4), "added": p})
    (OUT / "expand_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                            encoding="utf-8")

    print(f"\n  【干扰项相似度 vs 原篇】")
    for ci in range(3):
        tx = [f"{t} {a}" for t, a in zip(old[ci]["title"], old[ci]["abstract"])]
        V = enc.encode(tx, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)
        s0 = V @ Q[ci]
        s1 = np.array([x["sim"] for x in picks[ci]]) if picks[ci] else np.array([0.0])
        print(f"    簇{ci + 1}: 原篇自相似 中位 {np.median(s0):.3f} ｜ "
              f"新篇 中位 {np.median(s1):.3f}（min {s1.min():.3f}）")
    print(f"\n  已写 {OUT} ｜ expand_report.json ｜ 待下载 PDF {sum(need.values())} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
