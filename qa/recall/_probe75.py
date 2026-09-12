"""临时：查 cf63a4(2003.09520) 的 7.5 是否有推导素材（时间/倍率上下文）。"""
from __future__ import annotations
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import qasper_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

papers = load_papers()
pid = "2003.09520"
p = papers[pid]
# 找 cf63a4 的 gold
for q in p.get("qas") or []:
    if str(q.get("question_id") or "").startswith("cf63a4f9fe"):
        g, ev = gold_answer(q)
        print("Q:", q.get("question")[:120])
        print("gold:", (g or "")[:200])
        print("gold evidence:", (ev or "")[:250])

chunks = qasper_chunks(pid)
joined = "\n".join(c.text for c in chunks)
# 时间/倍率相关的上下文：找 3 / 2.5 / time / day / hour / manual / faster 等组合
pats = [r"\bthree\b", r"\b3\b", r"2\.5", r"day", r"hour", r"manual", r"faster", r"time[s]?\b", r"annotat\w*"]
print("\n=== 时间/倍率线索片段 ===")
seen = set()
for pat in pats:
    for m in re.finditer(pat, joined):
        s = max(0, m.start() - 150)
        frag = joined[s:m.start() + 200].replace("\n", " | ")
        key = frag[:60]
        if key in seen:
            continue
        seen.add(key)
        print(f"[{pat}] ...{frag}...")
        if len(seen) > 40:
            break
    if len(seen) > 40:
        break
