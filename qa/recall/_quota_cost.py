""""分池 + 名额保底"的代价：给表池留 m 个名额，文本侧会丢多少？

方案：最终候选 = 文本池 top-(K-m) ∪ 表池 top-m（K=12），**不需要路由、不需要分类器**。
代价 = 纯文本 gold 在文本池里位次 >= 12-m 的比例（这些题会因腾槽而失去 gold）。
收益 = 表池 top-m 命中目标表的比例（来自 `_only_table_retr.py`，同一套 21 题）。

用法：uv run python qa/recall/_quota_cost.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))

from _table_policy_ab import encode_new, new_chunks, text_vectors  # noqa: E402
from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from run_retrieval_eval import locate_gold, norm  # noqa: E402

K = 12
EQ = "norm"


def main() -> int:
    papers = load_papers()
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    by_pid: dict[str, list] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    gold_text_rank: list[int] = []     # 纯文本 gold 在**文本池**内的位次
    n_ext_gold = 0
    for pid, items in by_pid.items():
        texts = ordered_chunks(f"qasper_{pid}.qpdf")
        if not texts:
            continue
        tv = text_vectors(pid, len(texts))
        if tv is None:
            continue
        ext = new_chunks(pid, EQ)
        chunks = list(texts) + ext
        n_t = len(texts)
        vecs = np.vstack([tv, encode_new(ext)]) if ext else tv
        bm = BM25Index([c.text for c in chunks])
        ntexts = [norm(c.text) for c in chunks]
        for it in items:
            q = next((qq for qq in papers[pid]["qas"]
                      if str(qq.get("question_id") or "") == it["qid"]), None)
            if q is None:
                continue
            _g, evs = gold_answer_full(q)
            cg: set[int] = set()
            for e in evs:
                gi, _ = locate_gold(ntexts, e)
                if not gi:
                    try:
                        gi = {int(np.argmax(bm.score(norm(e))))}
                    except Exception:  # noqa: BLE001
                        gi = set()
                cg |= gi
            if not cg:
                continue
            if any(i >= n_t for i in cg):
                n_ext_gold += 1
                continue                      # 只统计"纯文本 gold"的题
            vs = (encode_query(q["question"]) @ vecs.T).ravel()
            bs = bm.score(q["question"])
            # 文本池内 RRF 名次（生产语义：两路满权重）
            rv = np.empty(n_t)
            rv[np.argsort(-vs[:n_t], kind="stable")] = np.arange(n_t)
            rb = np.empty(n_t)
            rb[np.argsort(-bs[:n_t], kind="stable")] = np.arange(n_t)
            order = np.argsort(-(1.0/(60+rv+1) + 1.0/(60+rb+1)), kind="stable")
            pos = next((k for k, c in enumerate(order) if c in cg), None)
            gold_text_rank.append(pos if pos is not None else 10 ** 6)

    f = np.array(gold_text_rank)
    n = len(f)
    print(f"纯文本 gold 的题 n={n}；gold 落在外部块的题 n={n_ext_gold}（占 {(n_ext_gold)/(n+n_ext_gold):.1%}）\n")
    print("文本池内 gold 位次分布：")
    for k in (1, 3, 5, 8, 9, 10, 11, 12, 16):
        print(f"  位次 < {k:2d}: {(f<k).sum():3d}/{n} = {(f<k).mean():6.1%}")
    print(f"  位次中位 = {np.median(f):.0f}")
    print("\n| 给表池名额 m | 文本侧保留 top-(12-m) | **文本侧丢 gold 的题** | 表池 top-m 命中目标表 |")
    print("|---|---|---|---|")
    # 表池命中率来自 _only_table_retr.py（21 题，可定位目标表）：top-1 13, top-3 17, top-5 18
    ext_hit = {0: "—", 1: "13/21 = 62%", 3: "17/21 = 81%", 5: "18/21 = 86%"}
    for m in (0, 1, 3, 5):
        keep = K - m
        lost = int((f >= keep).sum())
        print(f"| {m} | {keep} | {lost}/{n} = {lost/n:.2%} | {ext_hit.get(m, '—')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
