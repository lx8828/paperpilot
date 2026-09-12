"""临时：查 545ff2 / cf63a4 原文里相关数字的真实形态（判断 A 型还是 B 型根因）。"""
from __future__ import annotations
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.agents.document_cache import qasper_chunks  # noqa: E402

papers = load_papers()
qmap = {}
for pid, p in papers.items():
    for q in p.get("qas") or []:
        qmap[str(q.get("question_id") or "")] = (pid, q)

# (qid前缀, 要查的数字/形态)
PROBES = [
    ("545ff2f769", ["3600", "36 million", "36M", "72 million", "7200", "million"]),
    ("cf63a4f9fe", ["7.5", "7.5 days", "one week", "a week", "week"]),
]
for pre, pats in PROBES:
    qid = next(k for k in qmap if k.startswith(pre))
    pid, q = qmap[qid]
    chunks = qasper_chunks(pid)
    joined = "\n".join(c.text for c in chunks)
    print("=" * 60)
    print(f"{pre[:10]} pid={pid} chunks={len(chunks)}")
    for pat in pats:
        hits = [m.start() for m in re.finditer(re.escape(pat), joined)]
        if hits:
            i = hits[0]
            print(f"  原文含 {pat!r}: 共 {len(hits)} 处 例: ...{joined[max(0,i-90):i+70].replace(chr(10),' | ')}...")
        else:
            # 近似：含数字 token 周边
            print(f"  原文含 {pat!r}: ✗ 无")
