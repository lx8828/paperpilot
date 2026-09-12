"""临时：nol3j A/B 明细（loss / gate 干预题）。"""
from __future__ import annotations
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
d = json.load(open("qa/recall/ab_nol3j_result.json", encoding="utf-8"))
loss = [r for r in d if r["pass"] is False and r["oldV_pass"]]
gain = [r for r in d if r["pass"] and not r["oldV_pass"]]
interv = [r for r in d if r["action"] in ("repaired", "fallback")]
print("== LOSS（oldV 过、新 fail）==")
for r in sorted(loss, key=lambda x: x["score"]):
    print(f"  {r['qid'][:10]} {r['grp']:6} oldV=P new={r['score']}({r['level']},{r['action']}) "
          f"issues={r['issues']}")
print("\n== GATE 干预题（repaired/fallback）==")
for r in interv:
    print(f"  {r['qid'][:10]} {r['grp']:6} act={r['action']} score={r['score']} "
          f"oldV_pass={r['oldV_pass']} issues={r['issues']}")
print("\n== GAIN 样例（前 8，按 score）==")
for r in sorted(gain, key=lambda x: -x["score"])[:8]:
    print(f"  {r['qid'][:10]} {r['grp']:6} new={r['score']}({r['level']}) act={r['action']}")
