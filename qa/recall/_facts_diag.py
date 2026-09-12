"""临时：两步法失效归因——失败题的答案缺失，是"第一步没捞到"还是"第二步没用"。

方法（对 ab_rewrite 里 off 臂失败题）：
  1) 跑真实图（默认 nol3j）拿 state：l3_chunks（12 块全文）+ answer + cites；
  2) 用同一 context 重新调 _extract_facts 拿**完整**清单（debug 里被截断到 80 字）；
  3) 人工可判的三层：gold 内容是否在 (a) 12 块里 → (b) 清单里 → (c) 成稿里。
输出到 qa/recall/FACTS_DIAG_20260910.md 供逐条判读。
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.nodes import answer as A  # noqa: E402
from paperpilot.graph import qa_graph_v3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

N = 12


def main() -> int:
    llm._load_dotenv(str(ROOT))
    recs = json.load(open("qa/recall/ab_rewrite_result.json", encoding="utf-8"))
    fails = [r for r in recs if not r["off"]["pass"]]
    print(f"off 臂失败 {len(fails)} / {len(recs)}")
    # 优先挑 hard（残留主战场），补充 normal
    fails.sort(key=lambda r: (r["grp"] != "hard", r["off"]["score"]))
    sel = fails[:N]

    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)

    L = ["# 两步法失效归因（off 臂失败题）", "",
         "> 判读三层：gold 内容在 (a)12块内 → (b)事实清单内 → (c)成稿内", ""]
    for i, r in enumerate(sel, 1):
        qid = r["qid"]
        pid, q = qmap[qid]
        pdf = f"qasper_{pid}.qpdf"
        gold, ev = gold_answer(q)
        st = qa_graph_v3.ask(q.get("question", ""), pdf)
        chunks = list(st.get("l3_chunks") or [])
        ans = st.get("answer") or ""
        # 重建 context 拿完整 facts
        header = f"标题：{st.get('title','')}\n\n=== 全文检索到的正文（独立检索） ==="
        parts = [header]
        for j, c in enumerate(chunks, 1):
            parts.append(A._fmt_chunk_entry(j, c))
        context = "\n\n".join(parts)
        facts = A._extract_facts(q.get("question", ""), context)

        L.append("=" * 90)
        L.append(f"## [{i}] {qid[:10]} [{r['grp']}] off_score={r['off']['score']}")
        L.append(f"Q: {q.get('question','')}")
        L.append(f"gold: {(gold or '')[:200]}")
        L.append(f"evidence: {(ev or '')[:300]}")
        L.append(f"\n--- (a) 检索到 {len(chunks)} 块 ---")
        for j, c in enumerate(chunks, 1):
            sn = " ".join((c.get("text") or "").split())[:110]
            L.append(f"  [{j}] {c.get('section','')!r} len={len(c.get('text') or '')} | {sn}")
        L.append(f"\n--- (b) 事实清单（{len(facts)} 条）---")
        for f in facts:
            L.append(f"  ({f.get('n')}) {f.get('text')}")
        L.append("\n--- (c) 成稿 ---")
        L.append(ans)
        L.append("")
        print(f"[{i}/{len(sel)}] {qid[:10]} chunks={len(chunks)} facts={len(facts)}", flush=True)

    Path("qa/recall/FACTS_DIAG_20260910.md").write_text("\n".join(L), encoding="utf-8")
    print("written qa/recall/FACTS_DIAG_20260910.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
