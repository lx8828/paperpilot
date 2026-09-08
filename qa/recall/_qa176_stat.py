import json
from collections import Counter
from pathlib import Path

run = json.load(open("qa/qa_run_20260909_003404.json", encoding="utf-8"))
base = json.load(open("qa/baseline.json", encoding="utf-8"))
out = []
c = Counter(str(r.get("status")) for r in run)
out.append(f"v3 中文 QA run: {len(run)} 题  status={dict(c)}")
ok = sum(1 for r in run if r.get("status") == "✅")
warn = sum(1 for r in run if r.get("status") == "⚠️")
err = sum(1 for r in run if r.get("status") == "❌" or r.get("error"))
out.append(f"✅ {ok}  ⚠️ {warn}  ❌ {err}")
out.append(f"baseline 共 {len(base)} qid")
now = {r.get("qid"): r.get("status") for r in run}
regress = [q for q, snap in base.items() if q in now
           and isinstance(snap, dict) and snap.get("status") == "✅" and now[q] != "✅"]
improve = [q for q, snap in base.items() if q in now
           and isinstance(snap, dict) and snap.get("status") != "✅" and now[q] == "✅"]
missing = [q for q in base if q not in now]
out.append(f"相对 baseline：回归(✅→非✅) {len(regress)} | 改善(非✅→✅) {len(improve)} | 缺失 {len(missing)}")
for q in regress[:30]:
    b = base[q].get("status") if isinstance(base[q], dict) else base[q]
    out.append(f" 回归 {q}: base={b} now={now[q]}")
for q in improve[:10]:
    b = base[q].get("status") if isinstance(base[q], dict) else base[q]
    out.append(f" 改善 {q}: base={b} now={now[q]}")
Path("qa/recall/qa176_result.txt").write_text("\n".join(out), encoding="utf-8")
print("written")
