"""对照：净亏题 L3 全文检索窗口 vs gold evidence 重叠率（确认答案在 L2 圆心够不到的地方）。

search_l3 用默认多查询混合（同 B 臂），只检索不判分。输出：
- l3_chunks 总量与文本总长
- 每块与 gold evidence 的最长公共子串重叠率（按 evidence 长度归一）
- evidence 中命中 l3 文本的句子
"""
from __future__ import annotations
import io
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.nodes import search_l3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

CASES = ["8051927f914d", "37c7c62c9216", "2df910c9806f"]


def overlap(win_text: str, ev: str) -> float:
    a, b = win_text.lower(), (ev or "").lower()
    if not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
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
        pid, q = qmap[full]
        gold, ev = gold_answer(q)
        st = {"question": q.get("question", ""), "pdf": f"qasper_{pid}.qpdf"}
        st.update(search_l3(st))
        l3 = list(st.get("l3_chunks") or [])
        wt = "\n".join((c.get("text") or "") for c in l3)
        ov = overlap(wt, ev or "")
        print("=" * 90)
        print(f"{qid[:16]}  {q.get('question','')[:130]}")
        print(f"L3 chunks: {len(l3)} | 文本总长 {len(wt)} | 窗口∩evidence 重叠率 = {ov:.2f}")
        rated = sorted(
            ((overlap(c.get("text") or "", ev or ""), (c.get("text") or "")[:160]) for c in l3),
            key=lambda x: -x[0])
        for o, t in rated[:3]:
            print(f"  块重叠率 {o:.2f}: {t.replace(chr(10),' ')}")
        evs = [s.strip() for s in re.split(r"(?<=[.;])\s", ev or "") if len(s.strip()) > 25]
        print("evidence 句在 L3 文本中的命中：")
        low3 = wt.lower()
        for s in evs:
            key = s[:60].lower()
            hit = "✓" if key in low3 else ("~" if overlap(low3, s) > 0.5 else "×")
            print(f"  {hit} {s[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
