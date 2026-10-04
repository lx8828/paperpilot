"""**新 RAG-2 · 集合型检索（多簇 + 多 facet）**：建语料 → 造真值 → 三臂对照。零 LLM。

## 为什么要多簇
第一版只有 1 簇 × 8 个 facet（≈8 个独立真值）→ **任何均值都不可信**。
这里扩到 **多簇 × 多 facet**（每个簇 20 篇全文，facet 按"正例率 15%~75%"自动入选题），
拿到 ~30-50 道题，才能对"覆盖式 vs 排序式"下结论。

## 题与判据
- 题面（中文）：「这 20 篇里，**哪几篇** <做了 X>？列出全部 + 逐篇一句证据」
- 真值：facet 词面锚点（**扩展期用；正式期要人工抽检**）
- 判据：**篇级 P / R / F1**（自动，无 LLM judge）

## 三条臂
- **A｜排序式**（RAG-1 式）：全局 chunk top-k → 涉及哪些篇
- **B｜覆盖式**（本方案）：每篇取篇内最高分 → **跨篇 z-norm 校准** → 阈值判定
- 另报 **oracle 阈值**（上界）vs **固定阈值**（可部署）→ 差距 = "阈值可不可定"

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_setsel.py
"""
from __future__ import annotations

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
CLEAN = HERE / "data" / "litsearch" / "corpus_clean"
EMB = HERE / "data" / "litsearch" / "derived" / "emb"
OUT = HERE / "data" / "r2dev"
CHUNK, OVERLAP = 1000, 100
N_CLUSTER, N_FINAL = 240, 20

TOPICS = [
    "retrieval augmented generation for knowledge-intensive question answering",
    "instruction tuning and alignment of large language models",
    "efficient inference and long-context modeling of transformers",
]

# ── facet 池（锚点=真值来源；zh=题面，**刻意不给英文原词**；en=查询扩展词）──
FACETS: dict[str, dict] = {
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
DROP_EXTREME = {"llm_as_model"}          # 已证实 20 篇里正例恒 0


def chunks_of(t: str) -> list[str]:
    t = str(t)
    return [t[i:i + CHUNK] for i in range(0, len(t), CHUNK - OVERLAP)] or [""]


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def best_f1(z: dict[str, float], gold: set[str]) -> tuple[float, float]:
    vals = np.array(sorted(set(z.values())))
    best, bt = 0.0, float("nan")
    for t in np.concatenate([vals - 1e-6, [vals.max() + 1e-6]]):
        f = prf({d for d, s in z.items() if s >= t}, gold)[2]
        if f > best:
            best, bt = f, float(t)
    return best, bt


def build_clusters() -> list[dict]:
    import torch
    from sentence_transformers import SentenceTransformer

    ids = np.load(EMB / "corpusid.npy")
    M = np.concatenate([np.load(p) for p in sorted(EMB.glob("part_*.npy"))], axis=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    # 一次读遍全文（列不大，6 片 ~930MB，读一次比反复 filter 稳）
    print("读全文语料（6 片）…", flush=True)
    full = []
    for f in sorted(CLEAN.glob("*.parquet")):
        full.append(pd.read_parquet(f, columns=["corpusid", "title", "full_paper"]))
    full = pd.concat(full, ignore_index=True)
    full["n_chars"] = full["full_paper"].astype(str).str.len()
    full = full[full["n_chars"] > 8000].reset_index(drop=True)
    row_of = {int(c): i for i, c in enumerate(full["corpusid"])}
    print(f"  可用全文 {len(full):,} 篇", flush=True)

    clusters = []
    for ti, topic in enumerate(TOPICS):
        qv = enc.encode([topic], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        top = np.argsort(-(M @ qv))[:N_CLUSTER]
        cids = [int(ids[i]) for i in top if int(ids[i]) in row_of]
        sub = full.iloc[[row_of[c] for c in cids]].drop_duplicates("corpusid").reset_index(drop=True)
        txt = sub["full_paper"].astype(str).str.lower()
        mat, usable = {}, []
        for k, (pat, zh, en) in FACETS.items():
            if k in DROP_EXTREME:
                continue
            m = txt.str.contains(pat, regex=True, na=False)
            rate = float(m.mean())
            mat[k] = m.astype(int).tolist()
            if 0.15 <= rate <= 0.75:
                usable.append(k)
        score = np.zeros(len(sub))
        for k in usable:
            v = np.asarray(mat[k], dtype=float)
            score += v * (1 - v)
        sel = sorted(np.argsort(-score)[:N_FINAL].tolist())
        clusters.append(dict(topic=topic, cand=sub, mat=mat, sel=sel, usable=usable))
        print(f"  簇{ti + 1}：候选 {len(sub)} → 选 {len(sel)} 篇 ｜ 可用 facet {len(usable)}"
              f" ｜ {int(sub['n_chars'].iloc[sel].sum()):,} 字符", flush=True)
    return clusters, enc


def main() -> int:
    clusters, enc = build_clusters()

    rows = []
    for ci, cl in enumerate(clusters, 1):
        sub = cl["cand"].iloc[cl["sel"]].reset_index(drop=True)
        docs = [f"C{ci}P{i + 1}" for i in range(len(sub))]
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].tolist()):
            cs = chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        idx_by_doc = {d: np.where(owner == d)[0] for d in docs}
        print(f"\n{'=' * 104}\n【簇{ci}】{cl['topic'][:60]}…\n  {len(docs)} 篇 / {len(chunks)} 块"
              f" / {int(sub['n_chars'].sum()):,} 字符 ≈ {int(sub['n_chars'].sum() / 4 / 1000)}k token",
              flush=True)
        for facet in cl["usable"]:
            pat, zh, en = FACETS[facet]
            gold = {docs[i] for i in cl["sel"] if cl["mat"][facet][cl["sel"][i]]}
            if not gold or len(gold) == len(docs):
                continue
            for vname, q in (("zh", zh), ("zh+en", f"{zh} {en}"), ("en", en)):
                qv = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
                s = C @ qv
                per_max = {d: float(s[idx_by_doc[d]].max()) for d in docs}
                per_top3 = {d: float(np.sort(s[idx_by_doc[d]])[-3:].mean()) for d in docs}
                def zn(v):
                    a = np.array([v[d] for d in docs])
                    return {d: (v[d] - a.mean()) / (a.std() + 1e-9) for d in docs}
                zmax, ztop3 = zn(per_max), zn(per_top3)
                fmax, tmax = best_f1(zmax, gold)
                f3, _ = best_f1(ztop3, gold)
                pf = prf({d for d in docs if zmax[d] >= 0.0}, gold)
                a5 = set(owner[np.argsort(-s)[:5]].tolist())
                a10 = set(owner[np.argsort(-s)[:10]].tolist())
                rows.append(dict(cluster=ci, facet=facet, variant=vname, n_gold=len(gold),
                                 B_oracleF1_max=fmax, B_oracleF1_top3=f3, B_thr=round(tmax, 2),
                                 B_fix_f1=pf[2], B_fix_p=pf[0], B_fix_r=pf[1],
                                 A5_f1=prf(a5, gold)[2], A5_rec=prf(a5, gold)[1],
                                 A10_f1=prf(a10, gold)[2], A10_rec=prf(a10, gold)[1]))
        print(f"  完成 {sum(1 for r in rows if r['cluster'] == ci)} 条记录", flush=True)

    df = pd.DataFrame(rows)
    out = HERE / "results" / "R2_SETSEL_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 104}\n【汇总】{len(df)} 条 = {df['cluster'].nunique()} 簇 × "
          f"{df.groupby(['cluster', 'facet']).ngroups} 个 (簇, facet) × 3 变体")
    print(f"\n  {'变体':<7}{'臂B oracle(max)':>16}{'臂B oracle(top3)':>18}"
          f"{'臂B 固定阈':>12}{'P':>7}{'R':>7}{'臂A@5':>9}{'召':>7}{'臂A@10':>9}{'召':>7}")
    for v, sub in df.groupby("variant"):
        print(f"  {v:<7}{sub['B_oracleF1_max'].mean():>16.3f}{sub['B_oracleF1_top3'].mean():>18.3f}"
              f"{sub['B_fix_f1'].mean():>12.3f}{sub['B_fix_p'].mean():>7.2f}"
              f"{sub['B_fix_r'].mean():>7.2f}{sub['A5_f1'].mean():>9.3f}"
              f"{sub['A5_rec'].mean():>7.2f}{sub['A10_f1'].mean():>9.3f}{sub['A10_rec'].mean():>7.2f}")

    print(f"\n【按 gold 规模分层｜zh 变体】小集合（gold≤5）F1 天然低，要分开看")
    z = df[df["variant"] == "zh"].copy()
    z["档"] = np.where(z["n_gold"] <= 5, "gold≤5", np.where(z["n_gold"] <= 10, "gold 6~10", "gold≥11"))
    for g, sub in z.groupby("档"):
        print(f"  {g:<9} n={len(sub):>2} ｜ 臂B oracle {sub['B_oracleF1_max'].mean():.3f} "
              f"｜ 固定阈 {sub['B_fix_f1'].mean():.3f} ｜ 臂A@5 {sub['A5_f1'].mean():.3f}"
              f"（召 {sub['A5_rec'].mean():.2f}）")
    print(f"\n  → 已写 {out}")
    (OUT / "setsel_summary.json").write_text(
        json.dumps({"n_questions": int(df.groupby(['cluster', 'facet']).ngroups),
                    "mean_by_variant": {v: float(s["B_oracleF1_max"].mean())
                                        for v, s in df.groupby("variant")}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
