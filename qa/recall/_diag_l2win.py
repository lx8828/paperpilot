"""诊断净亏题：judge_l2 判够时 L2 窗口是否含 gold evidence。

手动模拟漏斗（等同 qa_graph 路由），逐步记录：
- judge_l1 判定与 target_sections
- 每次 expand_l2 后 judge_l2 判 enough 时的窗口 chunks 文本
- L2 answer 输出
对比窗口文本 vs gold evidence 的文本重叠（最长公共子串比例）。
"""
from __future__ import annotations
import io
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.nodes import (expand_l2, generate_answer, judge_l0,  # noqa: E402
                                     judge_l1, judge_l2, report_l0, retrieve_claims)
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

CASES = ["8051927f914d", "37c7c62c9216", "2df910c9806f"]  # 1 hard + 2 normal 净亏题


def chunk_text(c) -> str:
    return (c.get("text") or "")


def overlap(win_text: str, ev: str) -> float:
    """最长公共子串长度 / evidence 长度。"""
    a, b = win_text.lower(), (ev or "").lower()
    if not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b))
    return m.size / len(b)


def main() -> int:
    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    for qid in CASES:
        full = next((k for k in qmap if k.startswith(qid)), None)
        if not full:
            print(f"{qid} 未匹配 qid"); continue
        pid, q = qmap[full]
        gold, ev = gold_answer(q)
        print("=" * 90)
        print(f"{qid[:16]}  {q.get('question','')[:150]}")
        print(f"gold: {str(gold)[:200]}")
        print(f"evidence: {str(ev)[:200]}")
        st = {"question": q.get("question", ""), "pdf": f"qasper_{pid}.qpdf"}
        st.update(report_l0(st))
        st.update(judge_l0(st))
        if (st.get("verdict") or {}).get("enough"):
            print("  → L0 enough，直接 L0 作答（非 L2 场景）")
            continue
        st.update(retrieve_claims(st))
        st.update(judge_l1(st))
        v1 = st.get("verdict") or {}
        print(f"judge_l1: enough={v1.get('enough')} targets={v1.get('target_sections')}")
        if v1.get("enough"):
            print("  → L1 enough")
            continue
        targets = v1.get("target_sections") or []
        if not targets:
            print("  → L1 无 target → 直 L3（非 L2 场景）")
            continue
        # L2 自环：expand → judge_l2；判够 → answer
        answered = False
        for step in range(10):
            st.update(expand_l2(st))
            if (st.get("l2") or {}).get("done"):
                print(f"  step{step}: expand done（圆心/预算尽）→ 将降 L3")
                break
            st.update(judge_l2(st))
            v2 = st.get("verdict") or {}
            win = list(st.get("chunks") or [])
            wt = "\n".join(chunk_text(c) for c in win)
            ov = overlap(wt, ev or "")
            print(f"  step{step}: judge_l2 enough={v2.get('enough')} n_chunks={len(win)} "
                  f"| win_len={len(wt)} | 窗口∩evidence 重叠率={ov:.2f}")
            if v2.get("enough"):
                j2 = (st.get("debug") or {}).get("judge_l2") or {}
                print(f"    gap: {str(j2.get('gap'))[:120]}")
                print(f"    ---- 窗口前 600 字 ----")
                print("    " + wt[:600].replace("\n", "\n    "))
                st.update(generate_answer(st))
                ans = (st.get("answer") or "").strip()
                dbg = st.get("debug") or {}
                print(f"    ---- L2 answer (level={(dbg.get('answer') or {}).get('level')}) ----")
                print("    " + ans[:400].replace("\n", "\n    "))
                answered = True
                break
            if not win:
                break
        if not answered:
            print("  L2 循环未判够 → 走 L3（此诊断重点在判够场景）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
