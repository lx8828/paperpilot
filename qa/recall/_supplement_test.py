"""临时验证：数字 supplement 机制（不烧 LLM，只验证机器层分流 + gate 分支）。"""
from __future__ import annotations
import io
import sys
from pathlib import Path

sys.path.insert(0, "src")

from paperpilot.components import validator  # noqa: E402


def mkchunk(cid: str, text: str, page: int = 1):
    return {"chunk_id": cid, "page": page, "text": text}


def main() -> int:
    fails = 0
    full_chunks = [
        mkchunk("c1", "The corpus contains 36 million sentence pairs total.", 2),
        mkchunk("c2", "We evaluate on 3 benchmark datasets and report accuracy 78.6%.", 3),
        mkchunk("c3", "Section 7.5 discusses related work; no time estimates given.", 4),
        mkchunk("c4", "Fine-tuning took about 3 days with 8 GPUs.", 5),
    ]

    # ── 1) 数字被引证据缺失 + 全篇命中 → supplements（非 high） ──
    ev1 = ["We use the parallel corpus to train the NMT system."]  # 被引证据无 36 million
    issues, sups, unv = validator.validate_numbers(
        "What data were used to train the multilingual encoder?",
        "语料包含 3600 万句对 [1]。", evidence_texts=ev1, extra_chunks=full_chunks)
    ok = ((not issues) and (not unv) and len(sups) == 1
          and sups[0]["num_text"] == "36000000"
          and sups[0]["chunks"][0]["chunk_id"] == "c1")
    print(f"[{'OK ' if ok else 'FAIL'}] 被引无+全篇有 → supplement={[s['num_text'] for s in sups]} issues={issues}")
    fails += 0 if ok else 1

    # ── 2) 数字被引证据 + 全篇都有 → 无 issue 无 supplement ──
    ev2 = ["we report 78.6% top-1 accuracy on ImageNet"]
    issues, sups, unv = validator.validate_numbers(
        "What is the accuracy?", "准确率 78.6% [1]。", evidence_texts=ev2, extra_chunks=full_chunks)
    ok = (not issues) and (not sups) and (not unv)
    print(f"[{'OK ' if ok else 'FAIL'}] 被引有 → 空  issues={issues} sups={sups}")
    fails += 0 if ok else 1

    # ── 3) 都没有 → **unverified（交裁决）**，不再由机器直接判 high（2026-09-11 改） ──
    ev3 = ["The model achieves SOTA results."]
    issues, sups, unv = validator.validate_numbers(
        "How fast is training?", "训练耗时 4200 分钟 [1]。", evidence_texts=ev3, extra_chunks=full_chunks)
    ok = (not issues) and (not sups) and 4200.0 in unv
    print(f"[{'OK ' if ok else 'FAIL'}] 都没有 → unverified={unv} issues={issues}")
    fails += 0 if ok else 1

    # ── 4) 歧义场景：7.5 天 vs 全篇只有"章节 7.5"→ supplement 指向 c3 ──
    issues, sups, unv = validator.validate_numbers(
        "How long did manual correction take?", "纯人工需至少 7.5 天 [2]。",
        evidence_texts=["10-fold cross validation was used."], extra_chunks=full_chunks)
    ok = (not issues) and len(sups) == 1 and sups[0]["chunks"][0]["chunk_id"] == "c3"
    print(f"[{'OK ' if ok else 'FAIL'}] 7.5天 vs 章节7.5 → supplement={[ (s['num_text'], [c['chunk_id'] for c in s['chunks']]) for s in sups]}")
    fails += 0 if ok else 1

    # ── 5) 单位归一：7200 万 = 72 million 全篇没有 → unverified（交裁决） ──
    issues, sups, unv = validator.validate_numbers(
        "How many pairs?", "合计 7200 万句对 [1]。",
        evidence_texts=["wmt data"], extra_chunks=full_chunks)
    ok = (not issues) and (not sups) and 72000000.0 in unv
    print(f"[{'OK ' if ok else 'FAIL'}] 72M 全篇无 → unverified={unv} sups={sups}")
    fails += 0 if ok else 1

    # ── 6) gate 分流（mock supplement cb）：high 优先 repair；无 high supplement → 走 supplement ──
    # 6a 无任何问题 → pass
    entries = [{"kind": "chunk", "text": "we report 78.6% top-1 accuracy", "page": 3}]
    g = validator.gate("What is the accuracy?", "准确率 78.6% [1]。",
                       [{"evidence": "we report 78.6% top-1 accuracy", "page": 3}],
                       supplement=lambda q, a, sups, c: ("改", []),
                       n_entries=1, extra_chunks=full_chunks)
    ok = g["action"] == "pass" and not g["supplements"]
    print(f"[{'OK ' if ok else 'FAIL'}] 被引含数字 → gate pass (action={g['action']})")
    fails += 0 if ok else 1
    # 6b 被引无、全篇有 → supplement cb 触发；cb 说无需改 → pass
    g = validator.gate(
        "How fast is training?",
        "纯人工需至少 7.5 天 [1]。",
        [{"evidence": "10-fold cross validation was used.", "page": 2}],
        supplement=lambda q, a, sups, c: None,  # Generator 判断无需修改
        n_entries=1, extra_chunks=full_chunks)
    ok = g["action"] == "pass" and len(g["supplements"]) == 1
    print(f"[{'OK ' if ok else 'FAIL'}] supplement cb 返回 None(无需改) → pass，sups={len(g['supplements'])} (action={g['action']})")
    fails += 0 if ok else 1
    # 6c 被引无、全篇有 → supplement cb 返回新答案 → repaired
    g = validator.gate(
        "How fast is training?",
        "纯人工需至少 7.5 天 [1]。",
        [{"evidence": "10-fold cross validation was used.", "page": 2}],
        supplement=lambda q, a, sups, c: ("新版答案，删除无据的 7.5 天。", []),
        n_entries=1, extra_chunks=full_chunks)
    ok = g["action"] == "repaired" and g["answer"].startswith("新版答案")
    print(f"[{'OK ' if ok else 'FAIL'}] supplement cb 返回新答案 → repaired (action={g['action']})")
    fails += 0 if ok else 1
    # 6d 真编数（全篇无）→ 无 supplement cb，走 fallback（repair None）
    g = validator.gate(
        "How fast is training?", "训练耗时 4200 分钟 [1]。",
        [{"evidence": "The model achieves SOTA results.", "page": 2}],
        repair=lambda q, a, i, c: None, supplement=lambda q, a, s, c: None,
        n_entries=1, extra_chunks=full_chunks)
    ok = g["action"] == "fallback"
    print(f"[{'OK ' if ok else 'FAIL'}] 真编数且修不动 → fallback (action={g['action']})")
    fails += 0 if ok else 1
    # 6e 【新】全篇无 + 裁决判"支持" → 放行 pass（机器不再判死）
    g = validator.gate(
        "How fast is training?", "训练耗时 4200 分钟 [1]。",
        [{"evidence": "The model achieves SOTA results.", "page": 2}],
        repair=lambda q, a, i, c: None,
        adjudicate=lambda q, a, nums, c: (True, "文中以 4200 表示"),
        n_entries=1, extra_chunks=full_chunks)
    ok = g["action"] == "pass"
    print(f"[{'OK ' if ok else 'FAIL'}] 裁决判支持 → pass (action={g['action']})")
    fails += 0 if ok else 1
    # 6f 【新】全篇无 + 裁决判"不支持" → 升 high → 修不动 → fallback（真编数仍被拦）
    g = validator.gate(
        "How fast is training?", "训练耗时 4200 分钟 [1]。",
        [{"evidence": "The model achieves SOTA results.", "page": 2}],
        repair=lambda q, a, i, c: None,
        adjudicate=lambda q, a, nums, c: (False, "片段无此数"),
        n_entries=1, extra_chunks=full_chunks)
    ok = (g["action"] == "fallback"
          and any("复核判定不支持" in i["detail"] for i in g["issues"]))
    print(f"[{'OK ' if ok else 'FAIL'}] 裁决判不支持 → fallback (action={g['action']})")
    fails += 0 if ok else 1

    print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
