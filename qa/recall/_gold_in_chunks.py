"""临时：决定性检验——失败题的 gold 数值，是否出现在检索到的 12 块正文里。

若"不在" → 失败是数据源/检索问题（第一步无从捞起）；
若"在"  → 才轮到"清单漏捞 / 成稿没用"这类两步法问题。
零 LLM，复用生产检索（search_hybrid, top_k=12, 改写关）。
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def main() -> int:
    recs = json.load(open("qa/recall/ab_rewrite_result.json", encoding="utf-8"))
    fails = [r for r in recs if not r["off"]["pass"]]
    fails.sort(key=lambda r: (r["grp"] != "hard", r["off"]["score"]))
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)

    L = ["# gold 内容是否在检索块内（失败样本）", ""]
    n_in = n_out = 0
    for r in fails:
        qid = r["qid"]
        pid, q = qmap[qid]
        gold, ev = gold_answer(q)
        hits = ChunkIndex(f"qasper_{pid}.qpdf").search_hybrid(q.get("question", ""), top_k=12)
        blob = " ".join((h.get("text") or "") for h in hits)
        nums = sorted({m.group(0) for m in NUM.finditer(gold or "")}, key=len, reverse=True)
        # 只测"有区分度的数"（≥2 位，排除 1/2 之类）
        nums = [x for x in nums if len(x.replace(".", "").replace(",", "")) >= 2]
        found = [x for x in nums if x in blob]
        miss = [x for x in nums if x not in blob]
        if nums:
            if found and not miss:
                n_in += 1
            elif miss and not found:
                n_out += 1
        L.append(f"- {qid[:10]} [{r['grp']}] score={r['off']['score']} "
                 f"| gold数 {len(nums)} | 块内命中 {len(found)} | 缺 {len(miss)}")
        L.append(f"    gold: {(gold or '')[:120]}")
        if miss:
            L.append(f"    块内缺失的数: {miss[:8]}")
        if found:
            L.append(f"    块内已有的数: {found[:8]}")
    L.append("")
    L.append(f"> 全部 gold 数都在块内 {n_in} 题 | 全都不在 {n_out} 题 | 总失败 {len(fails)}")
    txt = "\n".join(L)
    Path("qa/recall/GOLD_IN_CHUNKS_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
