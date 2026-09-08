"""把 target_sections 命中率拆成：claims 覆盖度 × judge 挑选准确率。

对 diag_l1 里 not-enough 且能定位 gold 节的题：
  cov   = gold_top ∈ candidates（claims 检索有没有召回答案所在节）
  pick  = 覆盖的前提下 judge 的 target_sections 是否选了 gold_top
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from paperpilot.agents.document_cache import section_from_path
from paperpilot.agents.embedder import ClaimIndex

recs = json.load(open("qa/recall/diag_l1_result.json", encoding="utf-8"))
pdfs = {r["qid"]: None for r in recs}
# 需要 pid → pdf：从 recall_set 映射
from paperpilot.qasper_source import load_papers
setd = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
q2pid = {it["qid"]: it["pid"] for it in setd["items"]}
papers = load_papers()

nn = [r for r in recs if not r["enough"] and r["gold_top"]]
n = len(nn)
cov_hit = 0
pick_hit = 0
pick_denom = 0
examples = []
for r in nn:
    pdf = f"qasper_{q2pid[r['qid']]}.qpdf"
    # 重现 candidates：与 judge_l1 相同的 retrieved→home_section 去重
    q = next(qq for qq in papers[q2pid[r["qid"]]].get("qas") or []
             if str(qq.get("question_id") or "") == r["qid"])
    hits = ClaimIndex(pdf).search(q.get("question", ""), top_k=12)
    rep = ClaimIndex(pdf).report
    claims_map = {c["claim_id"]: c for c in rep.get("claims", [])}
    seen, cands = set(), []
    for h in hits:
        rep_id = h.get("rep_claim_id") or (h.get("claim_ids") or [""])[0]
        c = claims_map.get(rep_id, {})
        s = section_from_path(list(c.get("title_path") or []))
        if s and s not in seen:
            seen.add(s)
            cands.append(s)
    gt = r["gold_top"]
    cov = gt in cands
    if cov:
        cov_hit += 1
    if cov:
        pick_denom += 1
        if gt in (r["targets"] or []):
            pick_hit += 1
    if len(examples) < 6:
        examples.append({"qid": r["qid"][:10], "gold_top": gt,
                         "cov": cov, "targets": r["targets"], "cands_n": len(cands)})

print(f"可定位 not-enough 题 n={n}")
print(f"claims 覆盖度：gold 节 ∈ candidates {cov_hit}/{n} = {cov_hit/n:.0%}")
print(f"judge 挑选准确率（覆盖前提下选对）：{pick_hit}/{pick_denom} = {pick_hit/max(pick_denom,1):.0%}")
print(f"整体命中（36%）≈ 覆盖度 × 挑选准确率："
      f"{cov_hit/n:.2f} × {pick_hit/max(pick_denom,1):.2f} = "
      f"{cov_hit/max(n,1)*pick_hit/max(pick_denom,1):.2f}")
for e in examples:
    print("  ", e)
