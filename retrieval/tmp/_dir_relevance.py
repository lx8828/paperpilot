"""**方向型题 → 分层合适率**（线上口径：54 万 arXiv / 中文提问）。

## 为什么换这种题（解决上一个报告 §6 的"题集饱和"）
现有 200 题是"**从论文反写 query**"（gold = 那 1 篇）→ CE 交付 5 篇就 99.5%，**饱和无分辨率**。
方向型题**没有唯一 gold** → 判据从"gold 命中"换成 **"前 N 篇是否都合适"（分层合适率）**，
天然避开饱和，且**更贴近产品**（用户问的就是研究方向）。

## 判据（三级）
- **A 核心**：这篇论文正在解决该方向的**核心问题**（可作主参考）
- **B 相关**：同方向/相邻问题（可作补充）
- **C 无关**：不同任务/领域/只是词面相似

## 输出（本脚本要回答的两件事）
1. **前 N 是不是合适论文**：`P(A)@5 / P(A+B)@5 / ... @20 / @50`、以及各级的**梯度**
2. **分层权重该取多少**：用各层合适率**反推**惩罚系数（而不是拍一个）

⚠️ 判级是 **LLM judge（异源 glm 系）**，**必须人工抽查标定**（脚本会导出逐篇明细供人工核）。

用法：
    uv run python retrieval/tmp/_dir_relevance.py --limit 3      # 冒烟
    uv run python retrieval/tmp/_dir_relevance.py                # 全量（默认 20 条方向题）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ **必须在 import torch / sentence_transformers 之前**切 HF 离线（`_hfcache` 文档）：
#    本机 huggingface.co 不可达 → 模型加载会对每个模型发 HEAD 请求，Retry 1/5…5/5
#    → 表现成"卡死几分钟、GPU 占用不动"。`pilot_mixed50.py:60` 就是这么做的。
from _hfcache import ensure_hf_home  # noqa: E402
ensure_hf_home()

ROOT = HERE.parent
NEW = HERE / "data" / "arxiv"
RESULTS = HERE / "results"

ALPHA = 0.5
DEPTH = 1000

# 20 条方向型题（中文 = 生产口径；覆盖 cs 各子领域，均为"研究方向"而非"找某一篇"）
DIRECTIONS = [
    "面向长文档的多跳问答，如何在不做微调的前提下用图结构组织证据并保证可追溯？",
    "如何用强化学习让大模型智能体在真实软件工具链里可靠地完成多步操作？",
    "只有少量标注数据时，如何用合成数据提升检索模型的跨领域泛化？",
    "如何让视觉语言模型在长视频里做时序定位并给出可验证的时间戳证据？",
    "面向代码仓库级任务，如何构建能跨文件推理的检索增强生成方法？",
    "如何降低大模型推理时的显存占用，同时保持长上下文下的精度？",
    "多智能体协作中，如何避免错误在智能体之间传播并实现可解释的纠错？",
    "如何在表格与文本混合的语料上做统一检索并支持数值推理？",
    "语音-文本联合建模中，如何利用未标注音频提升低资源语言的识别？",
    "面向科学文献，如何自动抽取实验设置与结果并支持跨论文对比？",
    "图神经网络在异构图上如何在小样本条件下做可扩展的节点分类？",
    "如何为大模型的输出提供细粒度引用与事实一致性校验？",
    "推荐系统中，如何缓解长尾物品的冷启动并同时保持推荐多样性？",
    "如何用扩散模型做可控的三维场景生成并保证物理合理性？",
    "联邦学习场景下，如何检测并防御后门攻击而不泄露客户端数据？",
    "面向机器人操作的视觉-语言-动作模型，如何提升跨任务泛化与样本效率？",
    "如何在不重新训练的前提下对大模型做知识编辑并避免副作用？",
    "面向医学影像的小样本分割，如何利用跨模态预训练提升泛化？",
    "如何构建低延迟、可打断、能调用工具的端到端语音对话系统？",
    "时序预测中，如何结合频域信息与外部事件提升长期预测精度？",
]

_SYS_JUDGE = (
    "你是学术检索的相关性评审。给定一个**研究方向**需求和若干候选论文（标题+摘要），"
    "逐篇判断它与该方向的匹配程度，只输出三档之一：\n"
    "  A = 核心：这篇论文正在解决该方向的核心问题，可作为主参考；\n"
    "  B = 相关：同一方向或相邻问题，可作为补充参考；\n"
    "  C = 无关：不同任务/领域，或只是词面相似。\n"
    "要求：只依据标题与摘要判断；不要因为出现相同术语就判 A。\n"
    '只输出 JSON：{"grades": ["A","B",...]}（长度必须等于候选篇数，顺序与候选一致）'
)


def load_pipeline():
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from eval_retrieval import SparseBM25, rrf_fuse
    from llm_rerank import LLM, load_env

    load_env()
    llm = LLM()
    if not llm.ok():
        print("未配置 PAPERPILOT_LLM_*，退出")
        raise SystemExit(2)

    t_l = time.time()
    print("[1/4] 载入语料…", flush=True)
    d = pd.read_parquet(NEW / "meta" / "corpus_text.parquet",
                        columns=["docid", "title", "abstract"])
    titles = d["title"].astype(str).tolist()
    abstracts = d["abstract"].astype(str).tolist()
    print(f"      语料 {len(titles):,} 篇（{time.time() - t_l:.1f}s）", flush=True)
    t_v = time.time()
    print("[2/4] 载入向量（540k × 1024 ≈ 2.2 GB，读 109 个分片）…", flush=True)
    M = np.concatenate([np.load(p) for p in sorted((NEW / "emb").glob("part_*.npy"))], axis=0)
    print(f"      向量 {M.shape}（{time.time() - t_v:.1f}s）", flush=True)
    t_b = time.time()
    print("[3/4] 载入 BM25…", flush=True)
    bm = SparseBM25.load(NEW / "bm25")
    print(f"      BM25 就绪（{time.time() - t_b:.1f}s）", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/4] 载入两个模型（bge-m3 + bge-reranker-v2-m3，设备 **{dev}**）…", flush=True)
    if dev != "cuda":
        print("      ⚠️ **CPU 模式**：本仓库 `uv run` 会把 torch 换成 CPU 版 → 请用 "
              "`./.venv/Scripts/python.exe -u` 直跑（见 retrieval/README.md「环境」节）",
              flush=True)
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev, max_length=512)
    if dev == "cuda":
        try:
            ce.model.half()
        except Exception:  # noqa: BLE001
            pass
    print(f"      模型就绪（**{dev}**，累计 {time.time() - t_l:.1f}s）", flush=True)
    return dict(llm=llm, titles=titles, abstracts=abstracts, M=M, bm=bm,
                enc=enc, ce=ce, rrf_fuse=rrf_fuse)


def grade_batch(llm, query: str, items: list[tuple[int, str, str]]) -> list[str] | None:
    lines = [f"研究方向：{query}", "", "候选论文："]
    for j, (_row, t, a) in enumerate(items, 1):
        lines.append(f"{j}. 标题：{t[:200]}\n   摘要：{a[:600]}")
    lines.append("")
    lines.append(f'只输出 JSON：{{"grades": [...]}}（共 {len(items)} 项）')
    try:
        out = llm.chat(_SYS_JUDGE, "\n".join(lines))
    except Exception as e:  # noqa: BLE001
        print(f"      judge 失败：{e}")
        return None
    m = re.search(r"\[[^\[\]]*\]", out)
    if not m:
        return None
    g = re.findall(r'"(A|B|C)"', m.group(0)) or re.findall(r"\b([ABC])\b", m.group(0))
    return g if len(g) == len(items) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=50, help="交付/评审的前 N 篇")
    ap.add_argument("--pool", type=int, default=400, help="进 CE 的候选数")
    ap.add_argument("--judge-batch", type=int, default=20)
    args = ap.parse_args()

    qs = DIRECTIONS[:args.limit] if args.limit else DIRECTIONS
    P = load_pipeline()
    M, bm, enc, ce, llm = P["M"], P["bm"], P["enc"], P["ce"], P["llm"]
    titles, abstracts, rrf_fuse = P["titles"], P["abstracts"], P["rrf_fuse"]

    print(f"\n编码 {len(qs)} 条方向题…")
    QV = enc.encode(qs, batch_size=16, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True).astype(np.float32)

    recs: list[dict] = []
    t0 = time.time()
    for qi, q in enumerate(qs):
        s = M @ QV[qi]
        order = rrf_fuse([np.argsort(-s)[:DEPTH], bm.topk(q, DEPTH)], [ALPHA, 1 - ALPHA], len(titles))
        cand = order[:args.pool].tolist()
        docs = [titles[g].replace("\n", " ")[:300] + ". " + abstracts[g][:1500].replace("\n", " ")
                for g in cand]
        sc = np.asarray(ce.predict([(q, d) for d in docs], batch_size=64,
                                   show_progress_bar=False), dtype=np.float64)
        top = [cand[i] for i in np.argsort(-sc)[:args.topk]]

        grades: list[str] = []
        for b0 in range(0, len(top), args.judge_batch):
            chunk = top[b0:b0 + args.judge_batch]
            g = grade_batch(llm, q, [(r, titles[r], abstracts[r]) for r in chunk])
            grades += (g if g else ["?"] * len(chunk))
        print(f"  [{qi + 1}/{len(qs)}] {q[:34]}… 判级 {len(grades)} 篇 "
              f"A={grades.count('A')} B={grades.count('B')} C={grades.count('C')} "
              f"（{time.time() - t0:.0f}s）")
        for rank, (r, g) in enumerate(zip(top, grades), 1):
            recs.append(dict(qi=qi, rank=rank, row=int(r), grade=g, title=titles[r][:120]))

    df = pd.DataFrame(recs)
    out_csv = RESULTS / "DIRECTION_RELEVANCE_20260927.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 100}\n【前 N 篇的合适率】方向型题 {len(qs)} 条 ｜ 中文提问 / 54 万 arXiv ｜ "
          f"CE top-{args.topk}\n{'=' * 100}")
    print(f"  {'层':<12}{'P(A) 核心':>12}{'P(A+B) 合适':>14}{'P(C) 无关':>12}{'篇数':>7}")
    layers = [("top5", 1, 5), ("6~10", 6, 10), ("11~20", 11, 20),
              ("21~50", 21, 50), ("top20", 1, 20), ("top50", 1, 50)]
    stats: dict[str, float] = {}
    for name, lo, hi in layers:
        sub = df[(df["rank"] >= lo) & (df["rank"] <= hi)]
        n = len(sub)
        pa = (sub["grade"] == "A").mean()
        pab = (sub["grade"].isin(["A", "B"])).mean()
        pc = (sub["grade"] == "C").mean()
        stats[name] = float(pab)
        print(f"  {name:<12}{pa:>12.1%}{pab:>14.1%}{pc:>12.1%}{n:>7}")

    # nDCG@50（分级增益：A=3 / B=1 / C=0）—— 一条能概括"前 N 质量"的指标
    gain = df["grade"].map({"A": 3.0, "B": 1.0, "C": 0.0, "?": 0.0}).fillna(0.0)
    df = df.assign(gain=gain)
    nd: list[float] = []
    for _qi, sub in df.groupby("qi"):
        disc = 1.0 / np.log2(1.0 + sub["rank"].to_numpy())
        dcg = float((sub["gain"].to_numpy() * disc).sum())
        best = np.sort(sub["gain"].to_numpy())[::-1]
        idcg = float((best * disc).sum()) or 1e-9
        nd.append(dcg / idcg)
    print(f"\n  **nDCG@50（分级增益 A=3/B=1/C=0）= {np.mean(nd):.3f}**"
          f"（1.0 = 前 50 全 A 且都在最前；中位 {np.median(nd):.3f}）")

    print(f"\n{'#' * 100}\n分层权重（用**实测合适率**反推，不是拍脑袋）")
    base = stats.get("top5", 0.0) or 1e-9
    print(f"  {'层':<8}{'合适率':>9}{'相对权重(top5=1)':>18}")
    for name in ("top5", "6~10", "11~20", "21~50"):
        print(f"  {name:<8}{stats[name]:>9.1%}{stats[name] / base:>18.2f}")
    print("\n读法：① 若各层合适率**梯度很缓**（如 top5 90% / 21~50 80%）→ 位置惩罚应该**很轻**；"
          "\n      ② 若梯度陡 → 惩罚有依据；③ `P(C) 无关` 是真正要防的（用户看到会失望）。"
          "\n      ⚠️ LLM judge 必须人工抽查标定：逐篇明细见 " + out_csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
