"""临时：查 545ff2/cf63a4 的 gate number 误报根因。

GATE=0 拿原始答案 → 手动 validator.check 拿 detail → 比对数字在题干/被引证据/全篇 chunk 的存在性。
"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

os.environ["PAPERPILOT_V3_NOL3J"] = "1"
os.environ["PAPERPILOT_VALIDATOR_GATE"] = "0"  # 先拿原始试答
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

QIDS = ["545ff2f769", "cf63a4f9fe"]
llm._load_dotenv(str(ROOT))
try:
    llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
except Exception:
    pass

papers = load_papers()
qmap = {}
for pid, p in papers.items():
    for q in p.get("qas") or []:
        qmap[str(q.get("question_id") or "")] = (pid, q)

for pre in QIDS:
    qid = next(k for k in qmap if k.startswith(pre))
    pid, q = qmap[qid]
    qq = q.get("question", "")
    r = graph_ask(qq, f"qasper_{pid}.qpdf")
    raw = (r.get("answer") or "").strip()
    cites = list(r.get("cites") or [])
    dbg = r.get("debug") or {}
    n_ctx = ((dbg.get("answer") or {}).get("n_entries")) or None
    print("=" * 78)
    print(f"Q: {qq[:120]}")
    gold, ev = gold_answer(q)
    print(f"gold: {(gold or '')[:100]}")
    print(f"RAW ANSWER:\n{raw[:900]}")
    # 手动跑 check 拿 detail
    from paperpilot.components import validator
    entries = [{"kind": "chunk", "text": str(c.get("evidence") or ""), "page": c.get("page", 0)}
               for c in cites if c.get("evidence")]
    res = validator.check(qq, raw, entries, n_entries=n_ctx)
    nums = [i.get("detail") for i in res["issues"] if i["type"] == "number"]
    print("\nnumber issues:")
    for d in nums:
        print("  ", d[:160])
    # 提取被报数字 token
    ev_join = "\n".join(str(c.get("evidence") or "") for c in cites).lower()
    import re
    sig = validator._significant_numbers(raw) - validator._significant_numbers(qq)
    sig = sig - set()
    print("\nanswer 显著数字(除题干):", sorted(sig)[:12])
    print(f"cites n={len(cites)} | 这些数字在【被引证据】中: "
          f"{[ (s, s.lower() in ev_join) for s in sorted(sig)[:8] ]}")
    for c in cites[:3]:
        print(f"  cite p{c.get('page')} ev[:150]: {(c.get('evidence') or '')[:150]}")
