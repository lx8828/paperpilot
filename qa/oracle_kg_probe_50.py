"""Q4 Oracle-KG upper-bound probe with 50-paper semantic retrieval.

The five target papers are intentionally near-neighbours.  The remaining
45 papers are selected deterministically from reports whose titles are in
the same broad LLM/retrieval/reasoning area.  The experiment uses the
existing RAG-2 retrieval reader, not a knowledge graph, and disables the
output validator so we observe the semantic answerer's raw behaviour.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents import embedder as E
from paperpilot.agents.document_cache import ordered_chunks
from paperpilot.components.context_builder import compose
from paperpilot.tools import llm


TARGETS = [
    "2408.15240.pdf",   # GenRM
    "2504.00891.pdf",   # GenPRM
    "2504.16828.pdf",   # ThinkPRM
    "2509.23152.pdf",   # Mirror-Critique
    "2609.19515.pdf",   # VRR / LLM-as-an-Improver
    "2604.20087.pdf",   # SkillLearnBench
    "2605.09341.pdf",   # SkillMAS
    "2606.18837.pdf",   # Skill-MAS
    "2606.25389.pdf",   # COMAD
    "2609.02094v1.pdf", # MASkills
    "2410.03439.pdf",   # ToolGen
    "2604.11557.pdf",   # UniToolCall
    "2508.15214.pdf",   # SEER
    "2510.19791.pdf",   # ToolDreamer
    "2512.17052.pdf",   # DTDR
    "2502.12110.pdf",   # A-MEM
    "2601.06377.pdf",   # HiMem
    "2608.20631.pdf",   # Weighted Memory Tree
    "2606.10677.pdf",   # Infini Memory
    "2601.01885.pdf",   # AgeMem
]

KEYWORDS = {
    "verif": 5, "reward": 4, "critique": 5, "reason": 3,
    "llm": 2, "language model": 2, "agent": 2, "tool": 2,
    "retriev": 2, "rag": 3, "process": 2, "chain-of-thought": 4,
    "benchmark": 1, "evaluat": 1, "alignment": 1, "reasoning": 3,
}


def select_corpus() -> list[str]:
    """Return exactly 50 PDF names: five targets + 45 near-domain papers."""
    papers_dir = ROOT / "assets" / "papers"
    views_dir = ROOT / "assets" / "artifacts" / "out_views"
    scored: list[tuple[int, str]] = []
    for report in views_dir.glob("*.report.json"):
        pdf = report.name.removesuffix(".report.json") + ".pdf"
        if not (papers_dir / pdf).exists() or pdf in TARGETS:
            continue
        if not (views_dir / (pdf.removesuffix(".pdf") + ".cvec.npy")).exists():
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        title = str(data.get("title") or "").lower()
        score = sum(weight for kw, weight in KEYWORDS.items() if kw in title)
        scored.append((score, pdf))
    scored.sort(key=lambda x: (-x[0], x[1]))
    need = 50 - len(TARGETS)
    near_domain = [pdf for _, pdf in scored[:need]]
    if len(near_domain) < need:
        raise RuntimeError(f"only found {len(near_domain)} near-domain distractor PDFs")
    return TARGETS + near_domain


CASES = [
    {
        "id": "false_join_skill_decouple_alfworld",
        "question": "有哪一篇论文同时做到“把经验保留与参数更新解耦”并且“在 ALFWorld 上报告 94.0% 成功率”？请只回答同一篇论文；如果没有，请明确说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "conjunction",
                          "object": "Skill-MAS mechanism + SkillMAS ALFWorld result"},
    },
    {
        "id": "false_join_meta_skill_hotpotqa",
        "question": "这五篇技能/智能体论文中，哪一篇同时使用可演化 Meta-Skill，并在 HotpotQA、LoCoMo、GAIA 上评测？只回答同一篇论文；不存在就明确说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "conjunction",
                          "object": "Skill-MAS Meta-Skill + MASkills benchmarks"},
    },
    {
        "id": "false_join_vae_lifelong_os",
        "question": "哪一篇论文既通过 VAE 从离线轨迹学习技能，又在 Lifelong Agent Bench OS Task 上评测？必须是同一篇，不存在就说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "conjunction",
                          "object": "COMAD VAE + SkillMAS Lifelong OS benchmark"},
    },
    {
        "id": "false_join_skillmd_alfworld",
        "question": "哪一篇论文既把技能表示成 SKILL.md 等结构化文件，又在 ALFWorld 上报告 94.0%？必须是同一篇，不存在就说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "conjunction",
                          "object": "MASkills structured artifacts + SkillMAS ALFWorld result"},
    },
    {
        "id": "true_join_meta_skill_benchmarks",
        "question": "哪一篇论文把高层编排建模为可演化 Meta-Skill，并在四个复杂基准上评测？它使用哪组基准？",
        "expected": "Skill-MAS",
        "oracle_target": {"paper": "Skill-MAS", "relation": "conjunction",
                          "object": "Meta-Skill + DeepResearchBench/HLE-Math/BrowseComp-Plus/VitaBench"},
    },
    {
        "id": "true_join_buffer_topic_docs",
        "question": "哪一篇工作先把新观察放入缓冲区，再周期性整理为主题文档？请同时说明其主存储形式。",
        "expected": "Infini Memory",
        "oracle_target": {"paper": "Infini Memory", "relation": "conjunction",
                          "object": "buffer staging + topic documents"},
    },
    {
        "id": "true_join_next_token_outcome_filtering",
        "question": "哪一篇验证器同时采用 next-token prediction 训练，并以最终答案正确性做 outcome-based filtering，而不是依赖过程标签？",
        "expected": "GenRM",
        "oracle_target": {"paper": "GenRM", "relation": "conjunction",
                          "object": "next-token prediction + outcome-based filtering"},
    },
    {
        "id": "false_bridge_toolgen_seer",
        "question": "这五篇论文中，有没有一篇既把每个工具编码成唯一 token，又使用逐步经验回放来检索操作轨迹？如果有，请指出；如果没有，请明确说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "bridge",
                          "object": "ToolGen unique token + SEER stepwise replay"},
    },
    {
        "id": "false_bridge_mirror_vrr",
        "question": "这五篇验证器论文中，有没有一篇既聚合多条 critique 形成 verification score，又通过 Verify-Repair-Reselect 生成并重选新候选？必须是同一篇；没有就明确说没有。",
        "expected": "none",
        "oracle_target": {"paper": "none", "relation": "bridge",
                          "object": "Mirror-Critique aggregation + VRR repair/reselect"},
    },
    {
        "id": "ambiguous_skill_nonparametric",
        "question": "如果“非参数化”在这里既可能指不训练基础模型，也可能指不更新任何可学习状态，那么这几篇技能学习论文里，哪些真正不需要更新任何学习状态？请说明口径，不要只回答 training-free。",
        "expected": "scope-dependent",
        "oracle_target": {"paper": "multiple", "relation": "scope",
                          "object": "distinguish frozen base LLM from learned nonparametric state"},
    },
]


def main() -> None:
    llm._load_dotenv(str(ROOT))
    if not llm.is_configured():
        raise SystemExit("LLM is not configured")
    corpus = select_corpus()

    # Build a read-only 50-paper vector index directly from the existing
    # current chunks and cached vectors.  This avoids expensive stale-cache
    # rebuilds while preserving the semantic-retrieval condition.
    views = ROOT / "assets" / "artifacts" / "out_views"
    chunks: list[dict] = []
    vectors: list[np.ndarray] = []
    for pdf in corpus:
        stem = Path(pdf).stem
        vec_path = views / f"{stem}.cvec.npy"
        if not vec_path.exists():
            continue
        chs = ordered_chunks(pdf)
        vec = np.load(vec_path).astype("float32")
        if len(chs) != len(vec):
            continue
        for ch, row in zip(chs, vec):
            chunks.append({
                "pdf": pdf,
                "chunk_id": str(ch.chunk_id),
                "page": int(ch.page_span[0]),
                "section": " > ".join(ch.title_path),
                "text": str(ch.text),
            })
            vectors.append(row)
    if len(chunks) < 100:
        raise RuntimeError(f"too few cached chunks: {len(chunks)}")
    mat = np.vstack(vectors)
    print(f"loaded {len(chunks)} cached chunks from {len(corpus)} papers", flush=True)

    out = {
        "scope": "Oracle-KG upper-bound case study on 50 papers (not an implemented KG)",
        "corpus_size": len(corpus),
        "targets": TARGETS,
        "corpus": corpus,
        "cases": [],
    }
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {case['id']}", flush=True)
        llm.reset_usage()
        try:
            qv = E.encode_texts([case["question"]])[0]
            order = np.argsort(-(mat @ qv))[:12]
            entries = [
                {**chunks[int(j)], "kind": "chunk"} for j in order
            ]
            header = (
                f"=== 语义检索正文（50 篇语料，问题相关 top-12 块）==="
            )
            answer, cites, _facts = compose(
                case["question"], "", header, entries, "L3", {}
            )
            row = {
                **case,
                "answer": str(answer or ""),
                "route": ["semantic_vector_top12", "answer_L3"],
                "top_pdfs": [
                    chunks[int(j)]["pdf"] for j in order
                ],
                "cites": list(cites or []),
                "usage": llm.usage_stats(),
            }
        except Exception as exc:  # noqa: BLE001
            row = {**case, "error": f"{type(exc).__name__}: {exc}"}
        out["cases"].append(row)
        print(row.get("answer", row.get("error", ""))[:450].replace("\n", " "), flush=True)

    path = ROOT / "qa" / "oracle_kg_probe_50_hard.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
