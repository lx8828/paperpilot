"""EMPTY（第一步返回空清单）归因：gold 证据到底在不在检索到的块里？（零 LLM）

对 FACTS_PROBE 里判定 EMPTY 的题：
  gold 可定位到 l3_chunks → **检索送到了、模型漏捞**（第一步自身的锅）
  gold 定位不到          → **检索没送到**（锅在检索，不是第一步）
另打印块数/总字符，解释 context 过小的异常。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import run_retrieval_eval as ree  # noqa: E402
from paperpilot.agents.embedder import BM25Index  # noqa: E402
from paperpilot.agents.nodes.pull_chunk import search_l3  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

EMPTY_QIDS = ["8df89988", "d28260b5", "de12e059", "cd179292", "af75ad21", "b13d0e46"]


def main() -> int:
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")[:8]] = (pid, q)

    L = ["# EMPTY 归因：gold 在不在检索到的块里", "",
         "> 严格 = `locate_gold` 子串命中；兜底 = 仅 BM25 自检索命中（弱证据，单列）", "",
         "| qid | 块数 | 总字符 | gold段数 | 严格命中 | 兜底命中 | 判定 |",
         "|---|---|---|---|---|---|---|"]
    for q8 in EMPTY_QIDS:
        got = qmap.get(q8)
        if not got:
            continue
        pid, q = got
        st = search_l3({"question": q["question"], "pdf": f"qasper_{pid}.qpdf"})
        chunks = list(st.get("l3_chunks") or [])
        texts = [ree.norm(c.get("text") or "") for c in chunks]
        total = sum(len(c.get("text") or "") for c in chunks)
        _gold, evs = gold_answer_full(q)
        bm = BM25Index([c.get("text") or "" for c in chunks]) if chunks else None
        strict = fb = 0
        for ev in evs:
            gi, _ = ree.locate_gold(texts, ev)
            if gi:
                strict += 1
            elif bm is not None:
                fb += 1
        if strict:
            verdict = "**检索严格送到、模型漏捞**"
        elif fb:
            verdict = "仅兜底命中（弱）"
        else:
            verdict = "检索没送到"
        L.append(f"| {q8} | {len(chunks)} | {total} | {len(evs)} | {strict} | {fb} | {verdict} |")
        print(f"{q8}: chunks={len(chunks)} chars={total} ev={len(evs)} "
              f"strict={strict} fallback={fb}")

    txt = "\n".join(L)
    Path("qa/recall/FACTS_EMPTY_CHECK_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
