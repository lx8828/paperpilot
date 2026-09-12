"""诊断：为什么表格块进不了 top-12？（纯向量位次 + BM25 位次逐项拆开）

对 STILL_FAIL_DIAG 里 B 桶（表解析了但没进候选）的题，逐题打印：
  · 目标表格块在**纯向量**下的位次与余弦，对照 top-1 正文块
  · 在 **BM25** 下的位次
  · RRF 融合后的位次
目的：区分「块太大被稀释」/「语义鸿沟（表格是数字网格，问句是自然语言）」/「BM25 能救」

用法：uv run python qa/recall/_diag_tbl_rank.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.agents.embedder import BM25Index, ChunkIndex, encode_query  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402

TBL = Path("assets/artifacts/out_mineru")


def cap_key(s: str) -> str:
    m = re.match(r"\s*(table|figure)\s*([A-Za-z]?\d+)", str(s), re.I)
    return f"{m.group(1).lower()}{m.group(2).lower()}" if m else ""


def main() -> int:
    diag = json.loads(Path("qa/recall/still_fail_diag_20260911.json").read_text(encoding="utf-8"))
    bad = [d for d in diag if d["bucket"].startswith("B ")]
    papers = load_papers()
    print(f"B 桶（表没进候选）{len(bad)} 题\n")
    for d in bad:
        pid, qid = d["pid"], d["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        idx = ChunkIndex(f"qasper_{pid}.qpdf")   # 与线上同源：含表格块、向量取自缓存索引
        chunks = idx._doc_chunks()
        texts = [c.text for c in chunks]
        vecs = idx.vectors()
        qv = encode_query(q["question"])
        cos = (qv @ vecs.T).ravel()
        bm = BM25Index(texts).score(q["question"])
        want = {cap_key(c) for c in d["want"]}
        # 定位目标表格块
        tgt = [i for i, c in enumerate(chunks)
               if str(c.chunk_id).startswith("xtbl") and cap_key(c.text) in want]
        if not tgt:
            print(f"  {qid[:8]} {pid}: 未定位到目标表块（want={want}）")
            continue
        ti = tgt[0]
        vrank = int((cos > cos[ti]).sum()) + 1
        brank = int((bm > bm[ti]).sum()) + 1
        top5 = np.argsort(-cos)[:5]
        print(f"  {qid[:8]} {pid} | {d['score']} 分 | Q: {q['question'][:56]}")
        print(f"     目标表块 {chunks[ti].chunk_id}（{len(texts[ti])} 字符）"
              f" 向量位次 {vrank}/{len(chunks)} (cos={cos[ti]:.3f}) | BM25 位次 {brank}")
        print(f"     向量 top5: " + ", ".join(
            f"{chunks[i].chunk_id}({cos[i]:.3f},{len(texts[i])}字)" for i in top5))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
