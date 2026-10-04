"""**新 RAG-2 · 打分区分度诊断**：扩 top-k 有用吗？还是得重打分？（零 LLM）

## 要回答的三个问题
1. **全局 top-k 扩大有用吗？** → 测 k=5…320 时的"涉及篇数 / 篇级召回 / F1"曲线。
   （判据：**k 大到能覆盖全部 gold 篇时，涉及的篇数是否已 ≈ 全语料** → 是则"扩 k"不是检索，是"全给"）
2. **每篇预算 b 该给多少块？** → **块级召回**（X 的证据块有多少被取到 = 喂给 LLM 的输入质量）
   + 篇级 oracle F1（判定能力）。
3. **重打分有空间吗？** → **篇级 AUC**（区分度上界）+ **dense × BM25 融合扫描**（w 扫描）。
   ⚠️ 关键：中文提问打英文语料时 **BM25 全 0**（RAG-2 已踩过的坑）→ 词法路**必须用英文检索式**
   （= XLING 的"双语检索式"，dense 走中文题面、BM25 走英文锚点词，再融合）。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_score.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
CLEAN = HERE / "data" / "litsearch" / "corpus_clean"
EMB = HERE / "data" / "litsearch" / "derived" / "emb"
DEV = HERE / "data" / "r2dev"
CACHE = DEV / "clusters"
CHUNK, OVERLAP = 1000, 100
N_CLUSTER, N_FINAL = 240, 20
TOPICS = [
    "retrieval augmented generation for knowledge-intensive question answering",
    "instruction tuning and alignment of large language models",
    "efficient inference and long-context modeling of transformers",
]
FACETS = {
    "ablation": (r"\bablation", "做了消融实验（把方法拆掉一部分看效果）",
                 "ablation study component analysis removing parts"),
    "code_release": (r"github\.com|code (?:is|will be) (?:publicly )?available|we (?:release|open[- ]source)",
                     "公开了代码或数据（给了可获取的开源链接）",
                     "our code and data are publicly available github open source release"),
    "human_eval": (r"human (?:evaluation|study|annotation|rating)|annotators|human judges",
                   "做了人工评估或人工标注", "human evaluation annotators human ratings"),
    "multilingual": (r"multilingual|cross-lingual", "做了多语言或跨语言的实验",
                     "multilingual cross-lingual experiments in many languages"),
    "efficiency": (r"inference (?:time|cost|speed|latency)|\blatency\b|\bthroughput\b|FLOPs",
                   "报告了推理效率指标（时间 / 延迟 / 吞吐）",
                   "inference latency throughput time cost efficiency"),
    "significance": (r"significance (?:test|level)|p\s*<\s*0\.0|multiple (?:runs|seeds)|standard deviation",
                     "报告了多次运行的方差或统计显著性",
                     "statistical significance multiple runs standard deviation"),
    "new_dataset": (r"we (?:introduce|present|construct|release|collect) (?:a )?(?:new )?(?:dataset|benchmark|corpus)",
                    "自己提出或发布了新的数据集 / 基准", "we introduce a new dataset benchmark corpus"),
    "case_study": (r"case stud", "给出了案例分析", "case study examples qualitative analysis"),
    "error_analysis": (r"error analysis|failure (?:case|analysis|mode)",
                       "做了错误分析或失败案例分析", "error analysis failure cases analysis"),
    "prompt_eng": (r"prompt (?:engineering|template|design|format)|few[- ]shot prompt",
                   "讨论了提示词工程 / 提示模板设计", "prompt engineering template design few-shot"),
    "fine_tuning": (r"fine[- ]tun|finetun", "做了微调训练（不是只用现成模型）",
                    "fine-tuning finetuning train the model"),
    "zero_few_shot": (r"zero[- ]shot|few[- ]shot", "报告了零样本或少样本的结果",
                      "zero-shot few-shot evaluation without training"),
    "rl_training": (r"reinforcement learning|\bRLHF\b|\bPPO\b", "用了强化学习来训练或对齐",
                    "reinforcement learning RLHF PPO policy optimization"),
    "knowledge_distill": (r"distillation|distill(?:ing|ed)? (?:knowledge|the model)",
                          "用了知识蒸馏", "knowledge distillation teacher student"),
    "proof_theory": (r"\btheorem\b|\blemma\b|\bproof\b", "给了理论分析或证明（不只是实验）",
                     "theoretical analysis theorem proof guarantee"),
    "safety_bias": (r"toxic|safety|fairness|bias(?:ed)?\b", "讨论了安全 / 公平 / 偏见问题",
                    "safety toxicity fairness bias evaluation"),
    "context_length": (r"long[- ]context|context window|context length",
                       "处理了长上下文 / 上下文长度问题", "long context window length extension"),
    "deployment": (r"deploy|production system|real[- ]world application|online serving",
                   "面向真实部署或线上系统", "deployment production real-world online serving"),
    "annotation_cost": (r"annotation cost|crowdsourc|cheap(?:er)? (?:label|annotation)|labeling cost",
                        "关注标注成本（用了廉价 / 自动标注）",
                        "annotation cost crowdsourcing cheap labeling"),
}


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def auc(score: dict[str, float], gold: set[str]) -> float:
    """篇级 AUC（Mann-Whitney）。0.5 = 无区分度，1.0 = 完美。"""
    pos = [score[d] for d in score if d in gold]
    neg = [score[d] for d in score if d not in gold]
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def oracle_f1(score: dict[str, float], gold: set[str]) -> tuple[float, float]:
    vals = np.array(sorted(set(score.values())))
    best, bt = 0.0, float("nan")
    for t in np.concatenate([vals - 1e-9, [vals.max() + 1e-9]]):
        f = prf({d for d, s in score.items() if s >= t}, gold)[2]
        if f > best:
            best, bt = f, float(t)
    return best, bt


def zn(v: dict[str, float]) -> dict[str, float]:
    a = np.array(list(v.values()))
    return {k: (v[k] - a.mean()) / (a.std() + 1e-9) for k in v}


# ── 极简 BM25（避免引依赖；语料小，够用）──
def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.tf = [Counter(_tok(d)) for d in docs]
        self.len = [max(1, sum(t.values())) for t in self.tf]
        self.avg = float(np.mean(self.len))
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        N = len(docs)
        self.idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(_tok(q)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def build_or_load(enc) -> list[dict]:
    """3 个主题簇（20 篇全文），缓存到 parquet。"""
    meta_f = CACHE / "meta.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        return [dict(topic=m["topic"], sub=pd.read_parquet(CACHE / f"c{i}.parquet"),
                     usable=m["usable"]) for i, m in enumerate(meta)]

    ids = np.load(EMB / "corpusid.npy")
    M = np.concatenate([np.load(p) for p in sorted(EMB.glob("part_*.npy"))], axis=0)
    print("读全文语料（6 片，~930MB）…", flush=True)
    full = pd.concat([pd.read_parquet(f, columns=["corpusid", "title", "full_paper"])
                      for f in sorted(CLEAN.glob("*.parquet"))], ignore_index=True)
    full["n_chars"] = full["full_paper"].astype(str).str.len()
    full = full[full["n_chars"] > 8000].reset_index(drop=True)
    row_of = {int(c): i for i, c in enumerate(full["corpusid"])}

    CACHE.mkdir(parents=True, exist_ok=True)
    out, meta = [], []
    for ti, topic in enumerate(TOPICS, 1):
        qv = enc.encode([topic], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        top = np.argsort(-(M @ qv))[:N_CLUSTER]
        cids = [int(ids[i]) for i in top if int(ids[i]) in row_of]
        sub = full.iloc[[row_of[c] for c in cids]].drop_duplicates("corpusid").reset_index(drop=True)
        txt = sub["full_paper"].astype(str).str.lower()
        usable = [k for k, (pat, _, _) in FACETS.items() if 0.15 <= txt.str.contains(pat, na=False).mean() <= 0.75]
        score = np.zeros(len(sub))
        for k in usable:
            v = txt.str.contains(FACETS[k][0], na=False).astype(float).values
            score += v * (1 - v)
        sel = sorted(np.argsort(-score)[:N_FINAL].tolist())
        s2 = sub.iloc[sel].reset_index(drop=True)
        s2.insert(0, "docid", [f"C{ti}P{i + 1}" for i in range(len(s2))])
        s2[["docid", "corpusid", "title", "full_paper", "n_chars"]].to_parquet(CACHE / f"c{ti - 1}.parquet", index=False)
        meta.append({"topic": topic, "usable": usable})
        out.append(dict(topic=topic, sub=s2, usable=usable))
        print(f"  簇{ti}: {len(s2)} 篇 / {len(usable)} facet", flush=True)
    meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    clusters = build_or_load(enc)

    KS = [5, 10, 20, 40, 80, 160, 320]
    BS = [1, 2, 3, 6, 12]
    WS = [0.0, 0.25, 0.5, 0.75, 1.0]
    kcurve, bcurve, arows, fuse = [], [], [], []

    for ci, cl in enumerate(clusters, 1):
        sub = cl["sub"]
        docs = sub["docid"].tolist()
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].tolist()):
            cs = chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        bm = BM25(chunks)
        idx = {d: np.where(owner == d)[0] for d in docs}
        low = [c.lower() for c in chunks]
        print(f"\n【簇{ci}】{len(docs)} 篇 / {len(chunks)} 块（平均每篇 {len(chunks) / len(docs):.0f} 块）",
              flush=True)

        for facet in cl["usable"]:
            pat, zh, en = FACETS[facet]
            rx = re.compile(pat)
            chunk_hit = np.array([bool(rx.search(c)) for c in low])          # **块级真值**
            gold = {docs[i] for i in range(len(docs)) if chunk_hit[idx[docs[i]]].any()}
            if not gold or len(gold) == len(docs):
                continue
            n_gold_chunk = int(chunk_hit.sum())

            # ── 三种打分 ──
            dz = C @ enc.encode([zh], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            de = C @ enc.encode([en], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            bs = bm.scores(en)
            pm = lambda s: {d: float(s[idx[d]].max()) for d in docs}            # 篇级分 = 篇内 max
            s_dz, s_de, s_bs = pm(dz), pm(de), pm(bs)

            # ① 全局 top-k 曲线（dense zh）
            for k in KS:
                kk = min(k, len(chunks))
                involve = set(owner[np.argsort(-dz)[:kk]].tolist())
                p, r, f = prf(involve, gold)
                kcurve.append(dict(cluster=ci, facet=facet, k=kk, n_gold=len(gold),
                                   n_used_papers=len(involve), p=p, r=r, f1=f))

            # ② 每篇预算 b 曲线：块级召回（证据块是否取到）+ 篇级 oracle F1
            for b in BS:
                cov = []
                for d in docs:
                    ii = idx[d]
                    take = ii[np.argsort(-dz[ii])[:b]]
                    cov.append(chunk_hit[take].any())
                chunk_rec = (sum(chunk_hit[idx[d]][np.argsort(-dz[idx[d]])[:b]].sum() for d in docs)
                             / max(1, n_gold_chunk))
                of, _ = oracle_f1(zn(s_dz), gold)
                bcurve.append(dict(cluster=ci, facet=facet, b=b, n_gold=len(gold),
                                   paper_hit=float(np.mean(cov)), chunk_recall=float(chunk_rec),
                                   oracleF1=of))

            # ③ 区分度：AUC + 融合扫描
            base = {"dense_zh": s_dz, "dense_en": s_de, "bm25_en": s_bs}
            for nm, s in base.items():
                of, _ = oracle_f1(zn(s), gold)
                arows.append(dict(cluster=ci, facet=facet, scorer=nm, auc=auc(s, gold),
                                  oracleF1=of, n_gold=len(gold)))
            zd, zb = zn(s_dz), zn(s_bs)
            for w in WS:                                    # w = dense 权重
                fz = {d: w * zd[d] + (1 - w) * zb[d] for d in docs}
                of, _ = oracle_f1(fz, gold)
                fuse.append(dict(cluster=ci, facet=facet, w=w, auc=auc(fz, gold), oracleF1=of,
                                 n_gold=len(gold)))
        print(f"  已完成 {sum(1 for r in kcurve if r['cluster'] == ci)} 条 top-k 记录", flush=True)

    kc, bc, ar, fu = (pd.DataFrame(kcurve), pd.DataFrame(bcurve),
                      pd.DataFrame(arows), pd.DataFrame(fuse))
    for df, nm in ((kc, "topk"), (bc, "budget"), (ar, "auc"), (fu, "fuse")):
        df.to_csv(HERE / "results" / f"R2_SCORE_{nm}.csv", index=False, encoding="utf-8-sig")

    nq = kc.groupby(["cluster", "facet"]).ngroups
    print(f"\n{'=' * 104}\n【实验】{nq} 个 (簇,facet) ｜ 每簇约 {len(chunks)} 块\n")

    print("① 全局 chunk top-k 扩大有用吗？（dense 中文题面，均值）")
    print(f"   {'k':>5}{'涉及篇数':>10}{'篇级召回':>10}{'篇级P':>9}{'篇级F1':>9}   备注")
    base_r = {}
    for k, sub in kc.groupby("k"):
        n_used, r, p, f = sub["n_used_papers"].mean(), sub["r"].mean(), sub["p"].mean(), sub["f1"].mean()
        note = ""
        if k == KS[0]:
            base_r["k5"] = r
        if r >= 0.9 and not base_r.get("first90"):
            base_r["first90"] = k
        if n_used > 17:
            note = "← 涉及≈全部 20 篇（已不是检索）"
        print(f"   {k:>5}{n_used:>10.1f}{r:>10.2f}{p:>9.2f}{f:>9.3f}   {note}")
    print(f"   → 召回 0.9 需 k={base_r.get('first90', '>320')}；而 k=5 召回 {base_r.get('k5', 0):.2f}")

    print("\n② 每篇预算 b：**块级召回**（证据块被取到）＋ 篇级 oracle F1")
    print(f"   {'b':>3}{'块级召回':>10}{'篇命中率':>10}{'篇级oracleF1':>14}")
    for b, sub in bc.groupby("b"):
        print(f"   {b:>3}{sub['chunk_recall'].mean():>10.2f}{sub['paper_hit'].mean():>10.2f}"
              f"{sub['oracleF1'].mean():>14.3f}")

    print("\n③ 打分区分度（篇级 AUC，0.5=无区分度）＋ 融合扫描 w=dense权重")
    print(f"   {'打分':<12}{'AUC':>8}{'oracleF1':>11}")
    for nm, sub in ar.groupby("scorer"):
        print(f"   {nm:<12}{sub['auc'].mean():>8.3f}{sub['oracleF1'].mean():>11.3f}")
    for w, sub in fu.groupby("w"):
        tag = "  ← 纯 BM25" if w == 0 else ("  ← 纯 dense" if w == 1 else "")
        print(f"   fuse w={1 - w:.2f}d+{w:.2f}b".ljust(12) if False else
              f"   dense×{w:<5.2f}  {sub['auc'].mean():>8.3f}{sub['oracleF1'].mean():>11.3f}{tag}")

    print(f"\n  → 已写 R2_SCORE_topk / budget / auc / fuse .csv")

    # 按 gold 规模：小集合是否天生难
    print("\n④ 按 gold 规模分层（dense_zh）")
    t = ar[ar["scorer"] == "dense_zh"].copy()
    t["档"] = np.where(t["n_gold"] <= 5, "gold≤5", np.where(t["n_gold"] <= 10, "gold 6~10", "gold≥11"))
    for g, sub in t.groupby("档"):
        print(f"   {g:<9} n={len(sub):>2} ｜ AUC {sub['auc'].mean():.3f} ｜ oracle F1 {sub['oracleF1'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
