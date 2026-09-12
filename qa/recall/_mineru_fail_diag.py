"""临时：MinerU 自测失败题归因——看答案、闸门动作、以及 gold 内容是否在检索块内。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "src"))
os.environ["PAPERPILOT_USE_MINERU"] = "1"

from paperpilot.agents.embedder import ChunkIndex  # noqa: E402

recs = json.loads(Path("qa/recall/mineru_hard_result.json").read_text(encoding="utf-8"))
fails = [r for r in recs if not r["pass"]]
OUT = Path("qa/recall/MINERU_FAIL_DIAG_20260910.md")
L = ["# MinerU 自测失败归因（4 题）", ""]
for r in fails:
    L += ["=" * 88,
          f"## {r['qid']} ({r['pid']}) score={r['score']} [{r['level']}/{r['action']}]",
          f"Q: {r['question']}", f"gold: {r['gold']}",
          f"judge: {r['reason']}", "", "--- 系统回答 ---", (r["answer"] or "")[:1200], ""]
    hits = ChunkIndex(f"{r['pid']}.pdf").search_hybrid(r["question"], top_k=12)
    L.append(f"--- 检索到的 top12（chunk_id / 节 / 长度 / 首80字）---")
    for h in hits:
        sec = (h.get("section") or "")
        txt = " ".join((h.get("text") or "").split())[:80]
        L.append(f"  [{h.get('chunk_id')}] {sec!r} len={len(h.get('text') or '')} | {txt}")
    L.append("")
OUT.write_text("\n".join(L), encoding="utf-8")
print(f"written {OUT} | 失败 {len(fails)} 题")
for r in fails:
    print(f"  {r['qid']} {r['level']}/{r['action']} score={r['score']}")
