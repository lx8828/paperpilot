"""**新 RAG-2（集合型检索）· 开发语料与真值矩阵**（LitSearch 全文，零 LLM）。

## 判据回顾（为什么是这一类题、这个语料）
- 题类 = **集合型存在性**："这 N 篇里，哪几篇 <做了 X>？列全部 + 逐篇一句证据"
  → 判据 = **篇级 F1**（可自动、不需要 LLM judge）
  → 补的正是 RAG-1 的结构性缺口：它给 top-k 排序，**不给枚举、不能声称"其余没有"**
- 语料 = LitSearch `corpus_clean.full_paper`（全文，中位 **36,344 字符**）
  → N=20 ≈ **19 万 token > 128k 窗口** → **直读不可行** ✓
  （生产语料只有 title+abstract（1,374 字符/篇）→ 20 篇才 6.9k token，直读能解决；
    故**开发期先用全文**，调好后再迁生产语料）

## 选样标准：**让题有分辨率**（吃过"题集饱和"的教训）
1. **同主题**：用 LitSearch 自己的向量检索一个主题串 → 取 top-N 作簇（比关键词匹配稳，
   关键词在 2023 前的 NLP 语料上命中率极低：`retrieval-augmented generation` 只命中 6 篇）；
2. **facet 正例率落在 15%~75%** → 篇级 F1 才有区分度（接近 0%/100% 的 facet 必然饱和）；
3. 20 篇之间对同一 facet 有分歧。

## 产物
- `retrieval/data/r2dev/corpus20.parquet`：20 篇全文（docid/title/abstract/full_paper/n_chars）
- `retrieval/data/r2dev/facet_matrix.json`：facet → {中文题面, 锚点, 正例篇}
- 控制台打印矩阵 + 各 facet 正例数（供人工抽检）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 必须在 import torch / sentence_transformers 之前切 HF 离线（否则卡在 Hub 重试）
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
CLEAN = HERE / "data" / "litsearch" / "corpus_clean"
EMB = HERE / "data" / "litsearch" / "derived" / "emb"
OUT = HERE / "data" / "r2dev"

N_CLUSTER = 240          # 主题簇大小（候选池）
N_FINAL = 20             # 最终语料篇数
TOPIC = "retrieval augmented generation for knowledge-intensive question answering"

# ── facet：英文锚点（真值来源，**只用 non-capturing group** 避免 pandas 警告）──
FACETS: dict[str, dict] = {
    "ablation": {"pat": r"\bablation", "zh": "做了消融实验（把方法拆掉一部分看效果）"},
    "code_release": {"pat": r"github\.com|code (?:is|will be) (?:publicly )?available|we (?:release|open[- ]source)",
                     "zh": "公开了代码或数据（给了可获取的开源链接）"},
    "human_eval": {"pat": r"human (?:evaluation|study|annotation|rating)|annotators|human judges",
                   "zh": "做了人工评估或人工标注"},
    "llm_as_model": {"pat": r"\bGPT-4\b|ChatGPT|InstructGPT",
                     "zh": "把 GPT-4 这类商用大模型当作被评测或使用的模型"},
    "multilingual": {"pat": r"multilingual|cross-lingual", "zh": "做了多语言或跨语言的实验"},
    "efficiency": {"pat": r"inference (?:time|cost|speed|latency)|\blatency\b|\bthroughput\b|FLOPs",
                   "zh": "报告了推理效率指标（时间 / 延迟 / 吞吐）"},
    "significance": {"pat": r"significance (?:test|level)|p\s*<\s*0\.0|multiple (?:runs|seeds)|standard deviation",
                     "zh": "报告了多次运行的方差或统计显著性"},
    "new_dataset": {"pat": r"we (?:introduce|present|construct|release|collect) (?:a )?(?:new )?(?:dataset|benchmark|corpus)",
                    "zh": "自己提出或发布了新的数据集 / 基准"},
    "case_study": {"pat": r"case stud", "zh": "给出了案例分析"},
    "error_analysis": {"pat": r"error analysis|failure (?:case|analysis|mode)",
                       "zh": "做了错误分析或失败案例分析"},
}


def topic_cluster(k: int) -> list[int]:
    """用 LitSearch 已有向量检索主题串 → top-k 的 **corpusid**（同主题簇）。"""
    ids = np.load(EMB / "corpusid.npy")
    M = np.concatenate([np.load(p) for p in sorted(EMB.glob("part_*.npy"))], axis=0)
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    qv = enc.encode([TOPIC], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    s = M @ qv[0]
    top = np.argsort(-s)[:k]
    print(f"  主题簇：dense top-{k}（sim {s[top][-1]:.3f}~{s[top][0]:.3f}，设备 {dev}）")
    return [int(ids[i]) for i in top]


def main() -> int:
    print("=" * 100)
    print(f"【1/4】用现有向量选同主题簇（{TOPIC[:60]}…）")
    cids = topic_cluster(N_CLUSTER)

    print("【2/4】读簇内全文（filter 只读命中行）")
    rows = []
    idset = set(cids)
    for f in sorted(CLEAN.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["corpusid", "title", "abstract", "full_paper"])
        d = d[d["corpusid"].isin(idset)]
        if len(d):
            rows.append(d)
    cand = pd.concat(rows, ignore_index=True).drop_duplicates("corpusid")
    cand["n_chars"] = cand["full_paper"].astype(str).str.len()
    cand = cand[cand["n_chars"] > 8000].reset_index(drop=True)
    print(f"  可用全文 {len(cand)} 篇（中位 {int(cand['n_chars'].median()):,} 字符）")

    print("【3/4】facet 矩阵")
    txt = cand["full_paper"].astype(str).str.lower()
    mat: dict[str, list[int]] = {}
    for k, spec in FACETS.items():
        m = txt.str.contains(spec["pat"], regex=True, na=False)
        mat[k] = m.astype(int).tolist()
        print(f"  {k:<14} 正例率 {m.mean():>5.1%}（{int(m.sum())}/{len(m)}）")

    good = [k for k, v in mat.items() if 0.15 <= sum(v) / len(v) <= 0.75]
    print(f"  → 有分辨潜力的 facet：{good}")

    # 选 20 篇：让"有分辨潜力的 facet"在这 20 篇里仍有分歧（按 sum p(1-p) 排序）
    score = np.zeros(len(cand))
    for k in good:
        v = np.asarray(mat[k], dtype=float)
        score += v * (1 - v)
    sel = sorted(np.argsort(-score)[:N_FINAL].tolist())
    sub = cand.iloc[sel].reset_index(drop=True)

    chars = int(sub["n_chars"].sum())
    print(f"\n【4/4】选出 {len(sub)} 篇：合计 {chars:,} 字符 ≈ **{chars / 4 / 1000:.0f}k token**"
          f"（> 128k 窗口 → 直读不可行 ✓）")
    print("  " + " " * 44 + "".join(f"{k[:9]:>10}" for k in FACETS))
    for i in range(len(sub)):
        row = "".join(f"{'●' if mat[k][sel[i]] else '·':>10}" for k in FACETS)
        print(f"  [{i + 1:>2}] {str(sub['title'][i])[:42]:<42}{row}")
    print("\n  各 facet 在 20 篇里的正例数（⚠️=极端，分辨力低）：")
    for k in FACETS:
        n = sum(mat[k][i] for i in sel)
        print(f"    {k:<14} {n:>2}/20" + ("   ⚠️" if n in (0, 1, 19, 20) else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    keep = sub[["corpusid", "title", "abstract", "full_paper", "n_chars"]].copy()
    keep.insert(0, "docid", [f"P{i + 1}" for i in range(len(keep))])
    keep.to_parquet(OUT / "corpus20.parquet", index=False)
    gold = {
        "topic": TOPIC,
        "corpus": "litsearch corpus_clean.full_paper（开发期；后续迁生产语料）",
        "docs": [{"docid": f"P{i + 1}", "corpusid": int(sub["corpusid"][i]),
                  "title": str(sub["title"][i])[:120], "n_chars": int(sub["n_chars"][i])}
                 for i in range(len(sub))],
        "facets": {k: {"zh": FACETS[k]["zh"], "pat": FACETS[k]["pat"],
                       "pos": [f"P{i + 1}" for i in sel if mat[k][i]]} for k in FACETS},
    }
    (OUT / "facet_matrix.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"\n  已写 {OUT / 'corpus20.parquet'} ｜ {OUT / 'facet_matrix.json'}")
    print("  ⚠️ 锚点是**词面**真值 → 下一步先**人工抽检**锚点精度，再跑检索臂。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
