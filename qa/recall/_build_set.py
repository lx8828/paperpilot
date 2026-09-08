"""建 Recall@k 样本集 recall_set_v1.json（普通 + hard 分层，各带 gold evidence 可用）。

分层：normal = QASPER 有答案有 evidence 的普通题；hard = final_lock 仍 fail/unknown 的题。
每篇 ≤4 题；gold 必须可定位（有 evidence 文本）；seed 固定可复现。group: normal/hard。
"""
from __future__ import annotations
import json
import random
from pathlib import Path

sys_p = __import__("sys")
sys_p.path.insert(0, str(Path("src").resolve()))
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

papers = load_papers()

# qid(full) → (pid, q)，且必须 gold 有答案 + 有 evidence
index: dict[str, tuple[str, dict]] = {}
for pid, p in papers.items():
    for q in p.get("qas") or []:
        gold, ev = gold_answer(q)
        qid = str(q.get("question_id") or "")
        if gold and ev:
            index[qid] = (pid, q)

hard = json.load(open("qa/qasper_final_lock_20260907_183225.json", encoding="utf-8"))
hard_pool = [r["qid"] for r in hard if r.get("new_status") in ("fail", "unknown_ok")
             and r["qid"] in index]
normal_pool = [qid for qid in index if qid not in hard_pool]

rng = random.Random(20260908)
rng.shuffle(hard_pool)
rng.shuffle(normal_pool)

cap = 4
per_pid: dict[str, int] = {}


def can_take(pid):
    return per_pid.get(pid, 0) < cap


items = []
# hard ~100
for qid in hard_pool:
    pid, q = index[qid]
    if not can_take(pid):
        continue
    per_pid[pid] = per_pid.get(pid, 0) + 1
    items.append({"pid": pid, "qid": qid, "group": "hard",
                  "question": q.get("question", "")})
    if len(items) >= 100:
        break
# normal 补齐到 250
for qid in normal_pool:
    pid, q = index[qid]
    if not can_take(pid):
        continue
    per_pid[pid] = per_pid.get(pid, 0) + 1
    items.append({"pid": pid, "qid": qid, "group": "normal",
                  "question": q.get("question", "")})
    if len(items) >= 250:
        break

out = Path("qa/recall/recall_set_v1.json")
out.write_text(json.dumps({"version": "v1", "seed": 20260908,
                           "items": items}, ensure_ascii=False, indent=1),
               encoding="utf-8")
from collections import Counter
print("total", len(items), Counter(i["group"] for i in items),
      "papers", len(set(i["pid"] for i in items)))
