"""临时：S_merge 方案量化——只并超小块、不切任何健康块。

目标：确定合并阈值 & 评估副作用面（零 encode 静态）。
规则（候选）：
    base 块序遍历；块 len < TH 且非末块 → 并入下一块（text 拼接）；
    被并入块标记 merged（其 gold 命中随之迁移到接收块）；
    接收块变大可能超 4000 → 若超 4500 则放弃并入（宁保留碎块不把接收块切爆）。

评估输出：
    1) 不同 TH(700/900/1100) 下：碎块数、被合并数、接收块>4500 放弃数
    2) 200 题中 gold 块 < TH 的题数（受益面上限）；gold 块是接收块的题数（副作用面）
    3) gold 碎块被合并后，接收块的新大小（是否进入 1000-3000 优区）
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np  # noqa: E402
import run_retrieval_eval as ree  # noqa: E402

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

HARD_CAP = 4500


def merge_plan(chunks, th: int):
    """返回 (merged_idx_set, receiver_idx→新len, dropped_idx_set)。"""
    n = len(chunks)
    lens = [len(c.text) for c in chunks]
    merged: set[int] = set()
    dropped: set[int] = set()
    newlen = {i: l for i, l in enumerate(lens)}  # 拷贝，逐一改接收者
    i = 0
    while i < n:
        if i not in merged and lens[i] < th:
            # 找下一个非 merged 接收者
            j = i + 1
            while j < n and j in merged:
                j += 1
            if j >= n:
                # 末块无接收者：并到前一个
                j = i - 1
                while j >= 0 and j in merged:
                    j -= 1
            if j < 0:
                dropped.add(i)
                i += 1
                continue
            nl = newlen.get(j, lens[j]) + lens[i]
            if nl > HARD_CAP:
                dropped.add(i)   # 接收块会超 cap → 放弃并入
                i += 1
                continue
            newlen[j] = nl
            merged.add(i)
        i += 1
    return merged, newlen, dropped


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    by_pid: dict[str, list[dict]] = {}
    for it in data["items"]:
        by_pid.setdefault(it["pid"], []).append(it)

    # 预收集: pid → (chunks, gold_idx_by_qid)
    papers_info = {}
    for pid, its in by_pid.items():
        chunks = ordered_chunks(f"qasper_{pid}.qpdf")
        ntexts = [ree.norm(c.text) for c in chunks]
        qas = {str(q.get("question_id") or ""): q for q in papers[pid].get("qas") or []}
        gidx = {}
        for it in its:
            q = qas.get(it["qid"])
            if not q:
                continue
            _g, ev = gold_answer(q)
            if not ev:
                continue
            cg, _ = ree.locate_gold(ntexts, ev)
            if cg:
                gidx[it["qid"]] = next(iter(cg))
        papers_info[pid] = (chunks, gidx)

    for th in (700, 900, 1100):
        total_chunks = 0
        n_merge = 0
        n_drop = 0
        over_cap = 0
        gold_tiny = 0          # gold 块 < TH 的题
        gold_in_merged = 0     # gold 块本身被并入
        gold_in_receiver = 0   # gold 块是接收者（副作用面）
        recv_new_lens = []
        recv_new_in_good = 0
        for pid, (chunks, gidx) in papers_info.items():
            merged, newlen, dropped = merge_plan(chunks, th)
            total_chunks += len(chunks)
            n_merge += len(merged)
            n_drop += len(dropped)
            over_cap += sum(1 for i in dropped if i in gidx.values())  # drop 掉的是 gold 碎块？
            # 按 qid 统计
            for qid, gi in gidx.items():
                if len(chunks[gi].text) < th:
                    gold_tiny += 1
                    if gi in merged:
                        gold_in_merged += 1
                if gi not in merged and gi in newlen and newlen[gi] > len(chunks[gi].text):
                    gold_in_receiver += 1
                    recv_new_lens.append(newlen[gi])
                    if 900 <= newlen[gi] <= 3200:
                        recv_new_in_good += 1
        # 汇总
        print(f"── TH={th} ──")
        print(f"  base 块 {total_chunks} | 并入 {n_merge} | 放弃(接收超cap) {n_drop}")
        print(f"  受益面: gold块<{th} 题 {gold_tiny}（其中被并入 {gold_in_merged}）")
        print(f"  副作用: gold块是接收者 {gold_in_receiver} 题"
              + (f" | 接收后 900-3200 优区 {recv_new_in_good}/{gold_in_receiver}"
                 if gold_in_receiver else ""))
        if recv_new_lens:
            print(f"  接收块新大小 中位 {np.median(recv_new_lens):.0f} | max {max(recv_new_lens)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
