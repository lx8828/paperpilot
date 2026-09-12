"""判据 2/3：facts 覆盖率 与 证据位次×通过率。

判据 3（位置效应）：目标表块在**本次上下文条目顺序**中的位次 × 端到端结果。
  若靠后位次通过率明显更低 → "中间丢失/末尾丢失"成立 → 才值得做顺序优化/压缩。
判据 2（facts 是有损中间层？）：目标块在位次 p（1-based）时，`p ∈ facts_n` 与否 → 通过率对比。
  若"facts 覆盖"组显著高于"未覆盖"组 → 瓶颈在 facts 抽取层，而不是候选数量。

目标块由 gold 的 "Table N" ↔ 表块 caption 匹配（确定性，来自检索视图）。

用法：uv run python qa/recall/_judge_facts_pos.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))
import os  # noqa: E402

os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

ARMS = [("quota=0（现状）", "qa/recall/union_ab_q0_20260911.json"),
        ("quota=2（并集）", "qa/recall/union_ab_q2_20260911.json")]


def ck(t: str) -> str:
    m = re.match(r"\s*(table|figure)\s*([A-Za-z]?\d+)", str(t), re.I)
    return f"{m.group(1).lower()}{m.group(2).lower()}" if m else ""


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    targets: dict[str, set[str]] = {}
    cache: dict[str, dict[str, str]] = {}
    for x in exp:
        pid, qid = x["pid"], x["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        _g, evs = gold_answer_full(q)
        want = set()
        for e in evs:
            m = re.match(r"\s*(?:FLOAT SELECTED:)?\s*(table|figure)\s*([A-Za-z]?\d+)",
                         str(e), re.I)
            if m:
                want.add(f"{m.group(1).lower()}{m.group(2).lower()}")
        if not want:
            continue
        if pid not in cache:
            cache[pid] = {c.chunk_id: c.text for c in retrieval_chunks(f"qasper_{pid}.qpdf")
                          if str(c.chunk_id).startswith("xtbl")}
        targets[qid] = {cid for cid, t in cache[pid].items() if ck(t) in want}

    for label, path in ARMS:
        p = Path(path)
        if not p.exists():
            print(f"=== {label}: 记录未生成（{path}）===\n")
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        print(f"=== {label}（n={len(recs)}，pass {sum(1 for r in recs if r['pass'])}）===")
        pos_pass: Counter = Counter()
        pos_tot: Counter = Counter()
        cov_pass: Counter = Counter()
        cov_tot: Counter = Counter()
        for r in recs:
            tgt = targets.get(r["qid"]) or set()
            if not tgt:
                continue          # 无表目标 → 不进分桶（避免混淆）
            ids = list(r.get("entry_ids") or [])
            pos = next((i + 1 for i, cid in enumerate(ids) if cid in tgt), None)
            bucket = ("1-4" if pos and pos <= 4 else "5-8" if pos and pos <= 8
                      else "9-12" if pos and pos <= 12 else "13+" if pos else "不在候选")
            pos_tot[bucket] += 1
            pos_pass[bucket] += int(bool(r["pass"]))
            if pos:
                in_facts = pos in set(r.get("facts_n") or [])
                cov_tot[in_facts] += 1
                cov_pass[in_facts] += int(bool(r["pass"]))
        print("判据 3 · 证据位次 × 通过率（只在【有可定位表目标】的题上算，"
              "无表目标的题单独列作基线）：")
        print("| 目标块位次 | 题数 | pass | 通过率 |")
        print("|---|---|---|---|")
        for b in ("1-4", "5-8", "9-12", "13+", "不在候选"):
            if pos_tot[b]:
                print(f"| {b} | {pos_tot[b]} | {pos_pass[b]} | {pos_pass[b]/pos_tot[b]:.0%} |")
        nt_tot = sum(1 for r in recs if not (targets.get(r["qid"]) or set()))
        nt_ok = sum(1 for r in recs if not (targets.get(r["qid"]) or set()) and r["pass"])
        if nt_tot:
            print(f"| （无表目标的题，基线） | {nt_tot} | {nt_ok} | {nt_ok/nt_tot:.0%} |")
        print("判据 2 · facts 是否覆盖目标块（仅统计目标块在候选里的题）：")
        print("| facts 覆盖目标块 | 题数 | pass | 通过率 |")
        print("|---|---|---|---|")
        for k in (True, False):
            if cov_tot[k]:
                print(f"| {'是' if k else '否'} | {cov_tot[k]} | {cov_pass[k]} | "
                      f"{cov_pass[k]/cov_tot[k]:.0%} |")
        nf = [r.get("n_facts") or 0 for r in recs]
        print(f"facts 条数：中位 {sorted(nf)[len(nf)//2] if nf else 0}，范围 {min(nf) if nf else 0}~{max(nf) if nf else 0}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
