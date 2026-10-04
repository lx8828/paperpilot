"""**新 RAG-2 · 标准指标重算**（把自造指标换成文献口径，零 LLM、可复现）。

## 为什么（见 `retrieval/results/R2_PRIOR_ART_20260928.md`）
我们之前用的 `coverage@N` 是自造的；文献里的标准是：
- **MRecall@k**（JPR, Min et al. 2021）：全有或全无的集合召回；`m<=k` 时要求覆盖全部 m，`m>k` 时至少覆盖 k
- **StRecall@k**（Zhai et al. 2003；CoverageBench/SIGIR'26 采用）：top-k 覆盖的不同子话题比例
- **α-nDCG@k**（Clarke et al. 2008）：对"重复覆盖同一子话题"折扣，**惩罚冗余**
- **ALCE 式 citation recall/precision**（Gao et al., EMNLP'23）：逐论断查引用是否支持

**子话题（nugget）映射**：本任务的信息需求 = "哪些篇做了 X" → **每篇 gold 论文 = 一个 nugget**。
（严格说 facet 的"不同侧面"才是 nugget，但我们没有侧面级标注 → 这是**可辩护的最粗映射**，需在文档里写明。）

## 三层评测
1. **篇级交付 k 篇**（5 种打分 × k=5/10/20）：MRecall / StRecall / 集合 P / 集合 F1 / α-nDCG
2. **块级等预算 B 块**（全局 top-B vs **轮询 round-robin**）：篇级覆盖 —— 直接对标 AMER 的轮询聚合
3. **ALCE 式（检索侧）引用指标**：claim="P_i 做了 X"，引用=该篇 top-b 块
   - `cite_recall`：被引块中**至少一块含 X** 的篇占比（逐 claim）
   - `cite_precision`：被引块中**含 X** 的比例（逐 citation）
   ⚠️ 这里用**词面真值**当代替 NLI（本地无 NLI 模型）；换成 TRUE/LLM judge 时只替换 `judge()` 一个函数。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_std_metrics.py
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
KS = [5, 10, 20]
BS = [1, 3, 6, 12]          # 每篇块预算（等价块预算 = BS × 20 篇）
ALPHA = 0.9

FACETS = json.loads((HERE / "data" / "r2dev" / "facet_terms.json").read_text(encoding="utf-8")) \
    if (HERE / "data" / "r2dev" / "facet_terms.json").exists() else None

# facet → (锚点正则=真值, 中文题面, 英文锚点词, 英文改写句【不含锚点原词】)
F: dict[str, tuple[str, str, str, str]] = {
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
SCORERS = {
    "dense_zh": ("zh", "中文题面"),
    "dense_zh_en": ("zh_en", "中文+英文词"),
    "dense_en": ("en", "纯英文（同源语义）"),
    "bm25_para": (None, "词法·改写句（非同源）"),
    "bm25_anchor": (None, "词法·锚点原词（同源⚠️）"),
}


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def mrecall(ranked: list[str], gold: set[str], k: int) -> float:
    """MRecall@k（JPR）：m<=k 要求覆盖全部；m>k 要求至少覆盖 k 个。"""
    m = len(gold)
    if m == 0:
        return float("nan")
    c = len(set(ranked[:k]) & gold)
    return 1.0 if (c == m if m <= k else c >= k) else 0.0


def strecall(ranked: list[str], gold: set[str], k: int) -> float:
    """StRecall@k（Zhai；CoverageBench）：top-k 覆盖的 nugget 比例。"""
    return len(set(ranked[:k]) & gold) / len(gold) if gold else float("nan")


def setpf(ranked: list[str], gold: set[str], k: int) -> tuple[float, float, float]:
    topk = set(ranked[:k])
    tp = len(topk & gold)
    p = tp / len(topk) if topk else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def alpha_ndcg(ranked: list[str], gold: set[str], k: int, alpha: float = ALPHA) -> float:
    """α-nDCG@k（Clarke 2008）：同一 nugget 重复出现被折扣。理想排序 = 全部 gold 置顶。"""
    dcg, pfx = 0.0, 1.0
    for i, d in enumerate(ranked[:k], 1):
        g = 1.0 if d in gold else 0.0
        dcg += g / math.log2(i + 1) * pfx
        pfx *= (1 - alpha * g)
    idcg, pfx = 0.0, 1.0
    for i in range(1, min(k, len(gold)) + 1):
        idcg += 1.0 / math.log2(i + 1) * pfx
        pfx *= (1 - alpha)
    return dcg / idcg if idcg else 0.0


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _tok_cjk(s: str) -> list[str]:
    """`_tok` + **CJK 二元组**。

    ⚠️ 原版只保留 `[a-z0-9]+` → **中文查询被整段丢弃、BM25(zh) 恒为 0**
    （`_r2_score2.py` 的 `bm25_zh` scorer 就踩了这个坑）。
    本函数对 CJK 连段做**二元组**切分（免依赖的最简中文分词），对英文部分与原版**完全一致**。
    """
    s = str(s).lower()
    out = re.findall(r"[a-z0-9]+", s)
    for seg in re.findall(r"[\u4e00-\u9fff]+", s):
        out += ([seg[i:i + 2] for i in range(len(seg) - 1)] or [seg])
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75,
                 tok=None):
        # `tok`：可选分词器（默认 `_tok` = 旧行为，**向后兼容**）；传 `_tok_cjk` 可支持中文查询
        self._tok = tok or _tok
        self.tf = [Counter(self._tok(d)) for d in docs]
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
        for t in set(self._tok(q)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def judge(chunk: str, facet: str) -> bool:
    """ALCE 的 NLI 判定器（φ(premise, hypothesis)）的**本地替代**：词面锚点命中。

    ⚠️ 这不是 NLI，是**检索侧代理**；换成 `google/t5_xxl_true_nli_mixture` 或 LLM judge 时
    只替换本函数即可（其余流程不变）。"""
    return bool(re.search(F[facet][0], chunk, re.I))


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    lvl1, lvl2, lvl3 = [], [], []
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
        hit = {f: np.array([judge(c, f) for c in low]) for f in meta[ci]["usable"]}
        print(f"\n【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块 / {len(meta[ci]['usable'])} facet",
              flush=True)

        for facet in meta[ci]["usable"]:
            _, zh, anchor, para = F[facet]
            gold = {d for d in docs if hit[facet][idx[d]].any()}
            if not gold or len(gold) == len(docs):
                continue
            qvec = {nm: enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
                    for nm, q in (("zh", zh), ("zh_en", f"{zh} {anchor}"), ("en", anchor))}
            dsc = {nm: C @ v for nm, v in qvec.items()}
            bsc = {"bm25_para": bm.scores(para), "bm25_anchor": bm.scores(anchor)}

            # ── 第 1 层：篇级交付 k 篇 ──
            def add_l1(nm: str, ranked: list[str], note: str = "") -> None:
                for k in KS:
                    p, r, f1 = setpf(ranked, gold, k)
                    lvl1.append(dict(cluster=ci + 1, facet=facet, scorer=nm, k=k, n_gold=len(gold),
                                     MRecall=mrecall(ranked, gold, k),
                                     StRecall=strecall(ranked, gold, k),
                                     setP=p, setR=r, setF1=f1,
                                     aNDCG=alpha_ndcg(ranked, gold, k), note=note))

            for nm, (which, _) in SCORERS.items():
                s = dsc[which] if which else bsc[nm]
                pm = {d: float(s[idx[d]].max()) for d in docs}          # 篇级分 = 篇内 max
                add_l1(nm, sorted(docs, key=lambda d: -pm[d]))

            # 随机基线 + oracle 上界（否则绝对值无法解读）
            rng = np.random.default_rng(0)
            for nm, perms in (("random", [rng.permutation(docs).tolist() for _ in range(200)]),):
                for k in KS:
                    acc = [mrecall(pm_, gold, k) for pm_ in perms]
                    st = [strecall(pm_, gold, k) for pm_ in perms]
                    pf = [setpf(pm_, gold, k) for pm_ in perms]
                    an = [alpha_ndcg(pm_, gold, k) for pm_ in perms]
                    lvl1.append(dict(cluster=ci + 1, facet=facet, scorer=nm, k=k, n_gold=len(gold),
                                     MRecall=float(np.mean(acc)), StRecall=float(np.mean(st)),
                                     setP=float(np.mean([x[0] for x in pf])),
                                     setR=float(np.mean([x[1] for x in pf])),
                                     setF1=float(np.mean([x[2] for x in pf])),
                                     aNDCG=float(np.mean(an)), note="200 次随机排列均值"))
            add_l1("oracle", sorted(docs, key=lambda d: (d not in gold, d)), "完美排序（上界）")

            # ── 第 2 层：等块预算 B（全局 top-B vs 轮询 round-robin）──
            s = dsc["zh_en"]
            glob_order = np.argsort(-s)
            for b in BS:
                B = min(b * len(docs), len(chunks))
                # 轮询：每篇先取 1 块（取其最好块），再第二轮…
                per = {d: idx[d][np.argsort(-s[idx[d]])] for d in docs}
                rr = [per[d][rnd] for rnd in range(b) for d in docs if rnd < len(per[d])]
                rr = rr[:B]
                for strat, sel in (("global_topB", glob_order[:B]), ("round_robin", np.array(rr))):
                    used = list(dict.fromkeys(owner[sel].tolist()))     # 按首次出现排序
                    topk = used[:]
                    p, r, f1 = setpf(topk, gold, len(topk)) if topk else (0, 0, 0)
                    lvl2.append(dict(cluster=ci + 1, facet=facet, budget=b, n_chunks=int(sel.size),
                                     strategy=strat, n_gold=len(gold), n_used=len(used),
                                     covered=len(set(used) & gold),
                                     cover_rate=len(set(used) & gold) / len(gold),
                                     setP=p, setR=r, setF1=f1,
                                     aNDCG=alpha_ndcg(topk, gold, len(topk)),
                                     # 证据块层面
                                     ev_recall=float(hit[facet][sel].sum() / max(1, hit[facet].sum()))))

            # ── 第 3 层：ALCE 式（检索侧）引用指标 ──
            s = dsc["zh_en"]
            pm = {d: float(s[idx[d]].max()) for d in docs}
            for b in BS:
                # 逐 claim（只对"被判定为有"的篇 = 用 oracle 阈值上界，隔离检索质量）
                thr = float(np.quantile([pm[d] for d in docs], 1 - len(gold) / len(docs)))
                claims = [d for d in docs if pm[d] >= thr]
                cr, cp = [], []
                for d in claims:
                    ii = idx[d][np.argsort(-s[idx[d]])[:b]]
                    sup = hit[facet][ii]
                    cr.append(1.0 if sup.any() else 0.0)                # 至少一块支持
                    cp.append(float(sup.mean()))                        # 引用块里支持的比例
                lvl3.append(dict(cluster=ci + 1, facet=facet, b=b,
                                 n_claim=len(claims), n_gold=len(gold),
                                 claim_in_gold=len(set(claims) & gold) / max(1, len(claims)),
                                 cite_recall=float(np.mean(cr)) if cr else float("nan"),
                                 cite_precision=float(np.mean(cp)) if cp else float("nan")))
        print(f"  完成 L1 {sum(1 for r in lvl1 if r['cluster'] == ci + 1)} / "
              f"L2 {sum(1 for r in lvl2 if r['cluster'] == ci + 1)} / "
              f"L3 {sum(1 for r in lvl3 if r['cluster'] == ci + 1)}", flush=True)

    d1, d2, d3 = pd.DataFrame(lvl1), pd.DataFrame(lvl2), pd.DataFrame(lvl3)
    for df, nm in ((d1, "L1_paper"), (d2, "L2_budget"), (d3, "L3_citation")):
        df.to_csv(HERE / "results" / f"R2_STD_{nm}.csv", index=False, encoding="utf-8-sig")

    nq = d1.groupby(["cluster", "facet"]).ngroups
    print(f"\n{'=' * 108}\n【第 1 层｜篇级交付 k 篇】标准指标（{nq} 个真值 × 打分 × {len(KS)} 个 k）"
          f"｜平均 gold {d1['n_gold'].mean():.1f}/20")
    print(f"  {'打分':<13}{'说明':<18}{'k':>3}{'MRecall':>9}{'StRecall':>10}{'集合P':>8}{'集合F1':>8}{'α-nDCG':>9}")
    labels = {**{nm: SCORERS[nm][1] for nm in SCORERS},
              "random": "**随机基线**", "oracle": "**完美排序（上界）**"}
    for nm in list(SCORERS) + ["random", "oracle"]:
        sub = d1[d1["scorer"] == nm]
        for k in KS:
            t = sub[sub["k"] == k]
            print(f"  {nm:<13}{labels[nm]:<18}{k:>3}{t['MRecall'].mean():>9.3f}"
                  f"{t['StRecall'].mean():>10.3f}{t['setP'].mean():>8.3f}"
                  f"{t['setF1'].mean():>8.3f}{t['aNDCG'].mean():>9.3f}")

    print(f"\n【第 2 层｜等块预算 B 块】全局 top-B vs 轮询（打分 = dense 中文+英文词）")
    print(f"  {'预算B':>6}{'策略':<14}{'块数':>6}{'涉及篇':>7}{'覆盖篇':>7}{'覆盖率':>8}"
          f"{'集合P':>8}{'集合F1':>8}{'α-nDCG':>9}{'证据召回':>9}")
    for b in BS:
        for strat in ("global_topB", "round_robin"):
            t = d2[(d2["budget"] == b) & (d2["strategy"] == strat)]
            print(f"  {b:>6}{strat:<14}{t['n_chunks'].mean():>6.0f}{t['n_used'].mean():>7.1f}"
                  f"{t['covered'].mean():>7.2f}{t['cover_rate'].mean():>8.3f}"
                  f"{t['setP'].mean():>8.3f}{t['setF1'].mean():>8.3f}"
                  f"{t['aNDCG'].mean():>9.3f}{t['ev_recall'].mean():>9.3f}")

    print(f"\n【第 3 层｜ALCE 式引用指标（检索侧代理：词面锚点代 NLI）】")
    print(f"  {'每篇报 b 块':>10}{'claim 数':>9}{'claim在金标':>11}{'cite_recall':>12}{'cite_precision':>15}")
    for b in BS:
        t = d3[d3["b"] == b]
        print(f"  {b:>10}{t['n_claim'].mean():>9.1f}{t['claim_in_gold'].mean():>11.3f}"
              f"{t['cite_recall'].mean():>12.3f}{t['cite_precision'].mean():>15.3f}")

    print(f"\n  → 已写 R2_STD_L1_paper / L2_budget / L3_citation .csv")
    print("""
读法：
  · MRecall@k 是"全有或全无"（对标 JPR）；StRecall@k 是覆盖率（对标 CoverageBench）→ 两者差距 = 差几篇
  · α-nDCG 惩罚"反复覆盖同一篇" → 排序式打分的冗余代价
  · 第 2 层是**等块预算**对比：轮询 vs 全局 top-B（对标 AMER 的 round-robin 聚合）
  · 第 3 层是 ALCE 的检索侧代理：⚠️ 用词面代 NLI，是**上界偏松**（真 NLI 会更严）
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
