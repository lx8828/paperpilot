"""从全池扫"多方法/全局对比"题，抽 40（per_pid≤2）存 multiq_set.json。"""
from __future__ import annotations
import io
import json
import random
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

_SIG = re.compile(
    r"difference|differ from|compare|comparison|versus|\bvs\b|outperform|"
    r"better (than|accuracy|performance)|worse|among (the )?(methods|models|approaches|"
    r"baselines|algorithms)|which of the|which one|more effective|superior|"
    r"superiority|state-of-the-art methods?|the (methods|models|algorithms) .{0,40}(evaluated|compared|tested|presented)",
    re.I)


def main() -> int:
    papers = load_papers()
    view = ROOT / "assets/artifacts/out_views"
    pool: list[tuple[str, str, str]] = []  # (qid, pid, question)
    for pid, p in papers.items():
        if not (view / f"qasper_{pid}.report.json").exists():
            continue
        for q in p.get("qas") or []:
            gold, ev = gold_answer(q)
            qq = str(q.get("question", ""))
            if gold and ev and _SIG.search(qq):
                pool.append((str(q.get("question_id")), pid, qq))
    rng = random.Random(20260908)
    rng.shuffle(pool)
    per_pid: dict[str, int] = {}
    items = []
    for qid, pid, qq in pool:
        if per_pid.get(pid, 0) >= 2:
            continue
        per_pid[pid] = per_pid.get(pid, 0) + 1
        items.append({"qid": qid, "pid": pid, "question": qq[:200]})
        if len(items) >= 40:
            break
    Path("qa/recall/multiq_set.json").write_text(
        json.dumps({"n": len(items), "items": items}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"pool={len(pool)} → 抽 {len(items)}（papers {len(set(i['pid'] for i in items))}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
