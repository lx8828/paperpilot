"""**Phase 1：把文献方法真正上到检索侧**（覆盖导向重排 + 多查询 + 轮询聚合）

## 为什么做这个（用户质疑："真正的优化工作做了吗"）
前面几轮我们一直在**修尺子**（自造指标 → 同源偏袒 → 真值锚点漏 43% → NLI 判定器不可用），
**没上过任何优化方法**。文献里的标准做法（JPR / AMER / MMR / CoverageBench）我们只读了、没做。
本脚本开始做。

## 对照臂（都产出"篇级排序"，再按 k 截断）
| 臂 | 依据 |
|---|---|
| `A base` | **现行基线**：篇级分 = 篇内 max（dense 中文+英文词） |
| `B mq_max` | **多查询取最大**：3 条 LLM 生成的多侧面英文子查询 + 原查询，逐块取最大 |
| `C mq_rr` | **多查询 + 轮询聚合**（AMER 的 round-robin）：4 条查询各自排序后交错，按首次出现定篇序 |
| `D mmr` | **MMR 覆盖导向重排**（Clarke 1998；AMER 的 baseline）：在候选池上做 λ 扫描 |
| `E mq_mmr` | 多查询融合分 + MMR |

## 指标（标准口径，见 `R2_STD_METRICS_20260928.md`）
`MRecall@k`（全有或全无）、`StRecall@k`（覆盖率）、集合 P、`α-nDCG@k`

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_phase1.py
"""
from __future__ import annotations

import importlib.util as _iu
import json
import math
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
CACHE = HERE / "data" / "r2dev" / "clusters"
SUBQ = HERE / "data" / "r2dev" / "subqueries.json"
OUT = HERE / "results" / "R2_PHASE1.csv"
CHUNK, OVERLAP = 1000, 100
KS = [5, 10]
LAMBDAS = [1.0, 0.7, 0.5, 0.3]
POOL = 40

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F
mrecall, strecall, setpf, alpha_ndcg = _m.mrecall, _m.strecall, _m.setpf, _m.alpha_ndcg

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

SUBQ_SYS = """你是信息检索助手。给定一条关于**某篇论文**的中文论断，生成 3 条**互为补充、措辞不同**的
**英文**检索式，用于在论文正文中找出支持该论断的段落。

要求：
1. 覆盖论断的**不同侧面/不同表述**（例如"做了什么实验""报告了什么指标""在什么数据上做"）；
2. 不要只是把论断里的词堆在一起，要像论文作者会写的句子；
3. 只输出 JSON：{"queries": ["q1", "q2", "q3"]}"""


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def get_subqueries(facets: list[str]) -> dict[str, list[str]]:
    cache = json.loads(SUBQ.read_text(encoding="utf-8")) if SUBQ.exists() else {}
    todo = [f for f in facets if f not in cache]
    if todo:
        print(f"LLM 生成子查询（{len(todo)} 个 facet × 3 条）…", flush=True)
        for i, f in enumerate(todo):
            try:
                obj = llm.chat_json(SUBQ_SYS, f"论断：{F[f][1]}", temperature=0.3,
                                    max_tokens=300, prefix="PAPERPILOT_LLM")
                qs = obj.get("queries") if isinstance(obj, dict) else obj
                qs = [str(q) for q in (qs or [])][:3]
            except Exception as e:  # noqa: BLE001
                print(f"  {f} 失败 {type(e).__name__}")
                qs = []
            if not qs:
                qs = [F[f][2]]                       # 兜底：用英文锚点词
            cache[f] = qs
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(todo)} …", flush=True)
        SUBQ.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def zn(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    all_facets = sorted({f for m in meta for f in m["usable"]})
    subq = get_subqueries(all_facets)
    print(f"子查询就绪（{len(subq)} 个 facet）")

    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    rows = []
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].astype(str).tolist()):
            cs = chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        idx = {d: np.where(owner == d)[0] for d in docs}
        print(f"【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

        for facet in meta[ci]["usable"]:
            pat, zh, anchor, _ = F[facet]
            rx = re.compile(pat, re.I)
            gold = {d for d in docs if any(rx.search(c) for c in np.array(chunks)[idx[d]])}
            if not gold or len(gold) == len(docs):
                continue
            qs = [f"{zh} {anchor}"] + list(subq.get(facet, [anchor]))
            S = enc.encode(qs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32) @ C.T
            # ① A: 基线（单查询篇内 max）
            rankA = sorted(docs, key=lambda d: -S[0][idx[d]].max())
            # ② B: 多查询取最大（逐块 max over queries）
            Smax = S.max(axis=0)
            rankB = sorted(docs, key=lambda d: -Smax[idx[d]].max())
            # ③ C: 多查询 + 轮询聚合（AMER round-robin）
            orders = [list(np.argsort(-S[j])) for j in range(len(qs))]
            inter, seen = [], set()
            for rnd in range(len(chunks)):
                for o in orders:
                    if rnd < len(o) and int(o[rnd]) not in seen:
                        seen.add(int(o[rnd]))
                        inter.append(int(o[rnd]))
            rankC = list(dict.fromkeys(owner[inter].tolist()))
            # ④ D/E: MMR（覆盖导向重排）
            def mmr_rank(base_score, lam):
                pool = list(np.argsort(-base_score)[:POOL])
                sel: list[int] = []
                cand = np.array(pool)
                bz = zn(base_score[cand])
                while len(sel) < min(POOL, len(pool)):
                    if not sel:
                        pick = int(np.argmax(bz))
                    else:
                        sim = C[cand] @ C[np.array(sel)].T           # 候选 × 已选
                        red = sim.max(axis=1)
                        val = lam * bz - (1 - lam) * zn(red)
                        pick = int(np.argmax(val))
                    sel.append(int(cand[pick]))
                    cand = np.delete(cand, pick)
                    bz = np.delete(bz, pick)
                return list(dict.fromkeys(owner[np.array(sel)].tolist()))

            arms = {"A base": rankA, "B mq_max": rankB, "C mq_rr": rankC}
            for lam in LAMBDAS:
                nm = "D mmr" if lam != 1.0 else "D mmr(λ=1)"
                arms[f"{nm} λ={lam}"] = mmr_rank(S[0], lam)
            for lam in (0.5, 0.3):
                arms[f"E mq_mmr λ={lam}"] = mmr_rank(Smax, lam)

            for nm, rk in arms.items():
                for k in KS:
                    p, r, f1 = setpf(rk, gold, k)
                    rows.append(dict(cluster=ci + 1, facet=facet, arm=nm, k=k,
                                     n_gold=len(gold),
                                     MRecall=mrecall(rk, gold, k),
                                     StRecall=strecall(rk, gold, k),
                                     setP=p, setF1=f1, aNDCG=alpha_ndcg(rk, gold, k)))
        print(f"  完成 {sum(1 for r in rows if r['cluster'] == ci + 1)} 条", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    nq = df.groupby(["cluster", "facet"]).ngroups
    print(f"\n{'=' * 104}\n【Phase 1 结果】{nq} 个真值 × {df['arm'].nunique()} 臂 × {len(KS)} 个 k")
    base = df[df["arm"] == "A base"].set_index(["cluster", "facet", "k"])
    print(f"\n  {'臂':<20}{'k':>3}{'MRecall':>9}{'ΔMRec':>8}{'StRecall':>10}{'ΔStRec':>8}"
          f"{'集合P':>8}{'α-nDCG':>9}")
    for arm, sub in df.groupby("arm", sort=False):
        for k in KS:
            t = sub[sub["k"] == k].set_index(["cluster", "facet"]).join(
                base.xs(k, level="k")[["MRecall", "StRecall"]], rsuffix="_b", how="inner")
            dm = t["MRecall"].mean() - t["MRecall_b"].mean()
            ds = t["StRecall"].mean() - t["StRecall_b"].mean()
            print(f"  {arm:<20}{k:>3}{t['MRecall'].mean():>9.3f}{dm:>+8.3f}"
                  f"{t['StRecall'].mean():>10.3f}{ds:>+8.3f}"
                  f"{t['setP'].mean():>8.3f}{t['aNDCG'].mean():>9.3f}")
    print(f"\n  → 已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
