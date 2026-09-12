"""扫描全池"多方法/全局对比"类题（QASPER），供多维分支可行性判断采样。"""
from __future__ import annotations
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

# 对比/多方法信号词（答案往往需要跨多对象拼接/对比）
_SIG = re.compile(
    r"difference|differ from|compare|comparison|versus|\bvs\b|outperform|"
    r"better (than|accuracy|performance)|worse|among (the )?(methods|models|approaches|"
    r"baselines|algorithms)|which of the|which one|more effective|superior|"
    r"superiority|state-of-the-art methods?|the (methods|models|algorithms) .{0,40}(evaluated|compared|tested|presented)",
    re.I)


def main() -> int:
    papers = load_papers()
    view = ROOT / "assets/artifacts/out_views"
    hits = []
    total_gold = 0
    for pid, p in papers.items():
        if not (view / f"qasper_{pid}.report.json").exists():
            continue
        for q in p.get("qas") or []:
            gold, ev = gold_answer(q)
            if not (gold and ev):
                continue
            total_gold += 1
            qq = str(q.get("question", ""))
            qtype = str(q.get("question_type", ""))
            if _SIG.search(qq):
                # 数值/大写专名计数（多方法题常含 ≥2 专名或 between A and B）
                caps = len(set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", qq)))
                hits.append({"qid": str(q.get("question_id")), "pid": pid, "type": qtype,
                             "caps": caps, "q": qq[:150]})
    print(f"gold 可定位题总数: {total_gold}")
    print(f"对比信号命中: {len(hits)}")
    print("按 question_type:", dict(Counter(h["type"] for h in hits)))
    print("按 caps(大写专名≥2):", sum(1 for h in hits if h["caps"] >= 2))
    # 打印样本（先多专名+对比，最像"多方法对比"）
    cands = sorted(hits, key=lambda h: (-h["caps"], h["qid"]))
    for h in cands[:80]:
        print(f"[{h['type'][:10]:10}] caps={h['caps']} {h['qid'][:12]} {h['q']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
