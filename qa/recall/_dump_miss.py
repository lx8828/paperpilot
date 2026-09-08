"""dump 覆盖但 judge 未选对 gold 节的 12 条，供 A/B/C/D 人工分类。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from paperpilot.agents.document_cache import section_from_path
from paperpilot.agents.embedder import ClaimIndex
from paperpilot.qasper_source import load_papers

recs = json.load(open("qa/recall/diag_l1_result.json", encoding="utf-8"))
setd = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
q2pid = {it["qid"]: it["pid"] for it in setd["items"]}
papers = load_papers()
NUM_RE = re.compile(r"\d")

miss = [r for r in recs if not r["enough"] and r["gold_top"] and r["gold_top"] not in (r["targets"] or [])]
print(f"miss(覆盖或未覆盖未命中) n={len(miss)}")
lines = []
for r in miss:
    pid = q2pid[r["qid"]]
    pdf = f"qasper_{pid}.qpdf"
    q = next(qq for qq in papers[pid].get("qas") or []
             if str(qq.get("question_id") or "") == r["qid"])
    idx = ClaimIndex(pdf)
    hits = idx.search(q.get("question", ""), top_k=12)
    rep = idx.report
    claims_map = {c["claim_id"]: c for c in rep.get("claims", [])}
    lines.append("\n" + "=" * 80)
    lines.append(f"qid={r['qid'][:10]}  gold_top={r['gold_top']!r}  targets={r['targets']}")
    lines.append(f"Q: {q.get('question','')[:160]}")
    seen = set()
    for hi, h in enumerate(hits, 1):
        rep_id = h.get("rep_claim_id") or (h.get("claim_ids") or [""])[0]
        c = claims_map.get(rep_id, {})
        s = section_from_path(list(c.get("title_path") or []))
        key = (hi, s)
        if key in seen:
            continue
        seen.add(key)
        text = (c.get("text") or "")[:110].replace("\n", " ")
        num = "NUM" if NUM_RE.search(text or "") else "no-num"
        mark = "  << GOLD" if s == r["gold_top"] else ""
        sel = "  ^^SELECTED" if s in (r["targets"] or []) else ""
        lines.append(f"  [{hi}] {s!r} score={round(float(h.get('score',0)),3)} {num}{mark}{sel}\n"
                     f"       {text}")
    # 各候选节的命中 claim 数
    from collections import Counter
    cand = Counter()
    for h in hits:
        rep_id = h.get("rep_claim_id") or (h.get("claim_ids") or [""])[0]
        c = claims_map.get(rep_id, {})
        s = section_from_path(list(c.get("title_path") or []))
        if s:
            cand[s] += 1
    lines.append(f"  候选节分布(hits/section): {dict(cand)}")
Path("qa/recall/_miss_review.txt").write_text("\n".join(lines), encoding="utf-8")
print("written", len(miss), "cases")
