import io, json, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from paperpilot.qasper_source import load_papers
from paperpilot.agents.nodes.judge import _HARD_STOP

recs = json.load(open("qa/recall/ab_l2hard_result.json", encoding="utf-8"))
papers = load_papers()
qmap = {}
for pid, p in papers.items():
    for q in p.get("qas") or []:
        qmap[str(q.get("question_id") or "")] = q

STOP = _HARD_STOP


def signals(question):
    sigs = []
    for m in re.finditer(r"\d+(?:\.\d+)?", question):
        if m.group() not in sigs:
            sigs.append(m.group())
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", question):
        w = m.group()
        if w.lower() in STOP or any(c.islower() for c in w[1:]):
            continue
        if w not in sigs:
            sigs.append(w)
    return sigs


target = ["c30c3e0f8450", "8051927f914d", "05671d068679", "45893f31ef07", "0b9021cefca7",
          "345f65eaff16", "f3e96c5487d8", "21c104d14ba3", "37c7c62c9216", "2df910c9806f"]
for r in recs:
    if not r["qid"].startswith(tuple(target)):
        continue
    q = qmap[r["qid"]]
    qq = q.get("question", "")
    print("=" * 80)
    print(r["qid"][:12], r["grp"], "| sigs:", signals(qq))
    print("Q:", qq[:160])
