"""**新 RAG-2 · 打分口径续测（v2）**：排除同源偏袒 + 补齐融合。

## v1 暴露的问题
`bm25_en` 用**锚点原词**检索，而真值就是那批锚点 → **同源偏袒**，AUC 0.894 被高估。
另外 `dense(zh+en)`、RRF（名次级融合）漏测。

## 本脚本补的对照
| 打分 | 查询 | 用途 |
|---|---|---|
| `bm25_anchor` | 英文**锚点原词** | v1 的口径（同源，作上界参考） |
| `bm25_para` | 英文**改写句**（**刻意不含锚点原词**） | **非同源**词法路 → 真实水位 |
| `bm25_zh` | 中文题面 | 预期 ≈ 全 0（跨语言词法失效，RAG-2 已踩） |
| `dense_zh` / `dense_zh_en` / `dense_en` | — | 跨语言语义的损耗 |

融合：加权 z-norm（w 扫描）+ **RRF**（名次级，k=60）。
判定：oracle 阈值（上界）+ **固定阈值 z≥0**（可部署）。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_score2.py
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
DEV = HERE / "data" / "r2dev"
CACHE = DEV / "clusters"
CHUNK, OVERLAP = 1000, 100

# facet → (锚点正则=真值, 中文题面, 英文锚点词(v1 口径), **改写英文句**(不含锚点原词))
FACETS: dict[str, tuple[str, str, str, str]] = {
    "ablation": (r"\bablation", "做了消融实验（把方法拆掉一部分看效果）",
                 "ablation study component analysis removing parts",
                 "we remove or drop parts of the model and see how the results change"),
    "code_release": (r"github\.com|code (?:is|will be) (?:publicly )?available|we (?:release|open[- ]source)",
                     "公开了代码或数据（给了可获取的开源链接）",
                     "our code and data are publicly available github open source release",
                     "our implementation can be downloaded so others can reproduce the results"),
    "human_eval": (r"human (?:evaluation|study|annotation|rating)|annotators|human judges",
                   "做了人工评估或人工标注", "human evaluation annotators human ratings",
                   "people were asked to read the outputs and judge their quality"),
    "multilingual": (r"multilingual|cross-lingual", "做了多语言或跨语言的实验",
                     "multilingual cross-lingual experiments in many languages",
                     "we test the method in languages other than English"),
    "efficiency": (r"inference (?:time|cost|speed|latency)|\blatency\b|\bthroughput\b|FLOPs",
                   "报告了推理效率指标（时间 / 延迟 / 吞吐）",
                   "inference latency throughput time cost efficiency",
                   "how much time or memory it takes to run the model"),
    "significance": (r"significance (?:test|level)|p\s*<\s*0\.0|multiple (?:runs|seeds)|standard deviation",
                     "报告了多次运行的方差或统计显著性",
                     "statistical significance multiple runs standard deviation",
                     "we repeat each experiment several times and report the variation"),
    "new_dataset": (r"we (?:introduce|present|construct|release|collect) (?:a )?(?:new )?(?:dataset|benchmark|corpus)",
                    "自己提出或发布了新的数据集 / 基准", "we introduce a new dataset benchmark corpus",
                    "a new collection of examples that others can use for training and testing"),
    "case_study": (r"case stud", "给出了案例分析", "case study examples qualitative analysis",
                   "we look closely at a few concrete examples from our system"),
    "error_analysis": (r"error analysis|failure (?:case|analysis|mode)",
                       "做了错误分析或失败案例分析", "error analysis failure cases analysis",
                       "we examine the situations where our method does not work"),
    "prompt_eng": (r"prompt (?:engineering|template|design|format)|few[- ]shot prompt",
                   "讨论了提示词工程 / 提示模板设计", "prompt engineering template design few-shot",
                   "how the wording and the format of the instruction affect the results"),
    "fine_tuning": (r"fine[- ]tun|finetun", "做了微调训练（不是只用现成模型）",
                    "fine-tuning finetuning train the model",
                    "we continue training the model on our own data"),
    "zero_few_shot": (r"zero[- ]shot|few[- ]shot", "报告了零样本或少样本的结果",
                      "zero-shot few-shot evaluation without training",
                      "results when the model is given no examples or only a couple of them"),
    "rl_training": (r"reinforcement learning|\bRLHF\b|\bPPO\b", "用了强化学习来训练或对齐",
                    "reinforcement learning RLHF PPO policy optimization",
                    "we improve the model using reward signals coming from feedback"),
    "knowledge_distill": (r"distillation|distill(?:ing|ed)? (?:knowledge|the model)",
                          "用了知识蒸馏", "knowledge distillation teacher student",
                          "a smaller model is taught by copying the behaviour of a bigger one"),
    "proof_theory": (r"\btheorem\b|\blemma\b|\bproof\b", "给了理论分析或证明（不只是实验）",
                     "theoretical analysis theorem proof guarantee",
                     "we provide a formal argument with mathematical statements"),
    "safety_bias": (r"toxic|safety|fairness|bias(?:ed)?\b", "讨论了安全 / 公平 / 偏见问题",
                    "safety toxicity fairness bias evaluation",
                    "we examine whether the outputs are harmful or unfair to some groups"),
    "context_length": (r"long[- ]context|context window|context length",
                       "处理了长上下文 / 上下文长度问题", "long context window length extension",
                       "how the model behaves when the input becomes very large"),
    "deployment": (r"deploy|production system|real[- ]world application|online serving",
                   "面向真实部署或线上系统", "deployment production real-world online serving",
                   "we describe how the system is used in practice outside of research"),
    "annotation_cost": (r"annotation cost|crowdsourc|cheap(?:er)? (?:label|annotation)|labeling cost",
                        "关注标注成本（用了廉价 / 自动标注）",
                        "annotation cost crowdsourcing cheap labeling",
                        "getting labels is expensive or hard so we look for a cheaper way"),
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
    pos = np.array([score[d] for d in score if d in gold])
    neg = np.array([score[d] for d in score if d not in gold])
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def oracle_f1(score: dict[str, float], gold: set[str]) -> float:
    vals = np.array(sorted(set(score.values())))
    best = 0.0
    for t in np.concatenate([vals - 1e-9, [vals.max() + 1e-9]]):
        best = max(best, prf({d for d, s in score.items() if s >= t}, gold)[2])
    return best


def zn(v: dict[str, float]) -> dict[str, float]:
    a = np.array(list(v.values()))
    return {k: (v[k] - a.mean()) / (a.std() + 1e-9) for k in v}


def rrf(*ss: dict[str, float], k: int = 60) -> dict[str, float]:
    docs = list(ss[0])
    out = {d: 0.0 for d in docs}
    for s in ss:
        order = sorted(docs, key=lambda d: -s[d])
        for rank, d in enumerate(order, 1):
            out[d] += 1.0 / (k + rank)
    return out


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


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    recs, fusions = [], []
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
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
        low = [c.lower() for c in chunks]
        idx = {d: np.where(owner == d)[0] for d in docs}
        print(f"\n【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

        for facet in meta[ci]["usable"]:
            pat, zh, anchor, para = FACETS[facet]
            rx = re.compile(pat)
            gold = {d for d in docs if any(rx.search(low[j]) for j in idx[d])}
            if not gold or len(gold) == len(docs):
                continue
            # dense 三变体（中文 / 中文+英文 / 英文）
            qv = {nm: enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
                  for nm, q in (("dense_zh", zh), ("dense_zh_en", f"{zh} {anchor}"), ("dense_en", anchor))}
            S = {
                "dense_zh": {d: float((C @ qv["dense_zh"])[idx[d]].max()) for d in docs},
                "dense_zh_en": {d: float((C @ qv["dense_zh_en"])[idx[d]].max()) for d in docs},
                "dense_en": {d: float((C @ qv["dense_en"])[idx[d]].max()) for d in docs},
                # 词法四变体
                "bm25_anchor": {d: float(bm.scores(anchor)[idx[d]].max()) for d in docs},
                "bm25_para": {d: float(bm.scores(para)[idx[d]].max()) for d in docs},
                "bm25_zh": {d: float(bm.scores(zh)[idx[d]].max()) for d in docs},
            }
            for nm, s in S.items():
                z = zn(s)
                fixed = prf({d for d in docs if z[d] >= 0.0}, gold)
                recs.append(dict(cluster=ci + 1, facet=facet, scorer=nm, n_gold=len(gold),
                                 auc=auc(s, gold), oracleF1=oracle_f1(z, gold),
                                 fixed_f1=fixed[2], fixed_p=fixed[0], fixed_r=fixed[1],
                                 n_pred=sum(1 for d in docs if z[d] >= 0.0)))
            # 融合：加权 z-norm（w=词法权重）+ RRF（dense_zh_en × bm25_para，**非同源**）
            for lname in ("bm25_anchor", "bm25_para"):
                zd, zl = zn(S["dense_zh_en"]), zn(S[lname])
                for w in (0.0, 0.25, 0.5, 0.75, 1.0):
                    fz = {d: (1 - w) * zd[d] + w * zl[d] for d in docs}
                    fusions.append(dict(cluster=ci + 1, facet=facet, kind="z",
                                        pair=f"dense_zh_en+{lname}", w=w,
                                        auc=auc(fz, gold), oracleF1=oracle_f1(fz, gold)))
            for lname in ("bm25_anchor", "bm25_para"):
                fr = rrf(S["dense_zh_en"], S[lname])
                fusions.append(dict(cluster=ci + 1, facet=facet, kind="rrf",
                                    pair=f"dense_zh_en+{lname}", w=np.nan,
                                    auc=auc(fr, gold), oracleF1=oracle_f1(fr, gold)))
            # 纯 dense_zh_en 作融合基线
            fusions.append(dict(cluster=ci + 1, facet=facet, kind="single", pair="dense_zh_en",
                                w=np.nan, auc=auc(S["dense_zh_en"], gold),
                                oracleF1=oracle_f1(zn(S["dense_zh_en"]), gold)))
        print(f"  完成 {sum(1 for r in recs if r['cluster'] == ci + 1)} 条", flush=True)

    d = pd.DataFrame(recs)
    f = pd.DataFrame(fusions)
    d.to_csv(HERE / "results" / "R2_SCORE2_scorers.csv", index=False, encoding="utf-8-sig")
    f.to_csv(HERE / "results" / "R2_SCORE2_fuse.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 104}\n【A】单打分（{d.groupby(['cluster', 'facet']).ngroups} 个真值）")
    print(f"  {'打分':<14}{'AUC':>7}{'oracleF1':>10}{'固定阈F1':>10}{'P':>7}{'R':>7}{'判正数':>8}   说明")
    note = {"dense_zh": "中文题面（跨语言）", "dense_zh_en": "中文+英文词", "dense_en": "纯英文（同源语义）",
            "bm25_anchor": "词法·**锚点原词**（同源⚠️）", "bm25_para": "词法·**改写句**（非同源✓）",
            "bm25_zh": "词法·中文题面（预期≈0）"}
    for nm, sub in d.groupby("scorer"):
        print(f"  {nm:<14}{sub['auc'].mean():>7.3f}{sub['oracleF1'].mean():>10.3f}"
              f"{sub['fixed_f1'].mean():>10.3f}{sub['fixed_p'].mean():>7.2f}"
              f"{sub['fixed_r'].mean():>7.2f}{sub['n_pred'].mean():>8.1f}   {note[nm]}")

    print("\n【B】融合（都基于 dense_zh_en）")
    print(f"  {'方式':<22}{'AUC':>7}{'oracleF1':>10}")
    for (kind, pair, w), sub in f.groupby(["kind", "pair", "w"], dropna=False):
        wl = "—" if pd.isna(w) else f"w(词法)={w:.2f}"
        print(f"  {kind:<4}{pair:<18}{wl:<12}{sub['auc'].mean():>7.3f}{sub['oracleF1'].mean():>10.3f}")

    print("\n【C】读法")
    ba = d[d["scorer"] == "bm25_anchor"]["oracleF1"].mean()
    bp = d[d["scorer"] == "bm25_para"]["oracleF1"].mean()
    print(f"  · **同源偏袒幅度**：词法 oracleF1 锚点词 {ba:.3f} → 改写句 {bp:.3f}（差 {ba - bp:+.3f}）")
    print(f"  · 跨语言 dense：zh {d[d['scorer'] == 'dense_zh']['oracleF1'].mean():.3f} → "
          f"zh+en {d[d['scorer'] == 'dense_zh_en']['oracleF1'].mean():.3f} → "
          f"en {d[d['scorer'] == 'dense_en']['oracleF1'].mean():.3f}")
    print(f"  · 词法中文题面 bm25_zh oracleF1 "
          f"{d[d['scorer'] == 'bm25_zh']['oracleF1'].mean():.3f} → 若 ≈0 即印证"
          f"「中文问英文语料 BM25 全 0」（RAG-2 已知坑）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
