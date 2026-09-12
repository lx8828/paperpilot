"""并集 A/B 的翻转归因：q0(12 块) vs q2(12~14 块)。

关键问题：q2 丢掉的题，是不是**上下文被追加了块**的那些题？
  是 → "上下文变多本身有害"有了直接证据（用户的担心成立）；
  否（n_entries 没变） → 翻转纯属流程噪声。
"""
import json
from collections import Counter
from pathlib import Path

R = Path("qa/recall")
q0 = {x["qid"]: x for x in json.loads((R / "union_ab_q0_20260911.json").read_text(encoding="utf-8"))}
q2 = {x["qid"]: x for x in json.loads((R / "union_ab_q2_20260911.json").read_text(encoding="utf-8"))}
print(f"q0 pass {sum(1 for x in q0.values() if x['pass'])}/36 | "
      f"q2 pass {sum(1 for x in q2.values() if x['pass'])}/36")
print()
gain = [q for q in q2 if q2[q]["pass"] and not q0[q]["pass"]]
loss = [q for q in q2 if q0[q]["pass"] and not q2[q]["pass"]]
print(f"翻绿 {len(gain)} 题 / 翻红 {len(loss)} 题（churn={len(gain)+len(loss)}）")
print()
print("| 方向 | qid | 论文 | q0 分数 | q2 分数 | q0 条目 | q2 条目 | q2 追加块 | q2 是否含表块 |")
print("|---|---|---|---|---|---|---|---|---|")
for q in gain + loss:
    a, b = q0[q], q2[q]
    na, nb = a.get("n_entries"), b.get("n_entries")
    ext = [c for c in (b.get("entry_ids") or []) if str(c).startswith("xtbl")]
    d = "翻绿" if q in gain else "翻红"
    print(f"| {d} | {q[:8]} | {a['pid']} | {a['score']} | {b['score']} | {na} | {nb} | "
          f"{(nb or 0)-(na or 0):+d} | {len(ext)} 个 |")
print()
loss_extra = sum(1 for q in loss if (q2[q].get("n_entries") or 0) > (q0[q].get("n_entries") or 0))
gain_extra = sum(1 for q in gain if (q2[q].get("n_entries") or 0) > (q0[q].get("n_entries") or 0))
print(f"翻红题里「上下文被加长」的: {loss_extra}/{len(loss)}")
print(f"翻绿题里「上下文被加长」的: {gain_extra}/{len(gain)}")
print()
print("分数变化分布（q2 − q0）:", dict(sorted(Counter(
    (q2[q]["score"] or 0) - (q0[q]["score"] or 0) for q in q0).items())))
