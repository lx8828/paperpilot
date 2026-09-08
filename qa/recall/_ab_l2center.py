"""实验策略④：L2 多候选圆心 A/B on L2 目标题群 67 题。

圆心诊断（_diag_center.py）证明：净亏题 L1 池 70% 含 gold 块、节 70% 命中，
但"每节 score 最高单 claim"圆心命中仅 10%（答案块在池内被浪费）。
修法：PAPERPILOT_L2_MULTI_CENTER=1 → _pick_centers 把节内全部命中 claim 的 chunk
都列入候选圆心（预算 ≤8），expand 自环依次试。
本臂 = 原 graph_ask + env 开关（judge_l2 保持默认松判，验证"圆心喂对"本身的效果）。
对照历史：base(现状)40 / strict(严判够)44 / B(直L3)44。
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_l2center_result.json")

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案（可能含原文摘录）；"
    "③ 待评测系统回答与它引用的论文原文。请独立判断系统回答是否**准确**回答了问题（与标准答案一致、不编造、"
    '不遗漏关键信息）。只输出 JSON：{"score": 1-5, "reason": "一句话"}。5=完全正确且细节齐全；'
    "4=正确仅缺次要细节；3=部分正确有重要遗漏或轻微错误；2=明显错误或遗漏核心；1=答非所问或编造。")
_JUDGE_USER = """【问题】
{question}

【标准答案】
gold: {gold}
evidence: {evidence}

【系统回答】
{answer}

【引用原文】
{cites}

请打分(1-5)。引用必须真实支撑，否则视为编造降分。"""


def external_judge(q, answer, cites):
    gold, ev = gold_answer(q)
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（无答案，应诚实说明）",
                              evidence=(ev or ""), answer=answer[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    os.environ["PAPERPILOT_L2_MULTI_CENTER"] = "1"
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    prev = json.load(open("qa/recall/ab_l2target_result.json", encoding="utf-8"))
    items = [r for r in prev if r["l2_target"]]
    if args.limit:
        items = items[: args.limit]

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        qid = it["qid"]
        pid, q = qmap[qid]
        llm.reset_usage()
        try:
            out = graph_ask(q.get("question", ""), f"qasper_{pid}.qpdf")
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            j2 = dbg.get("judge_l2") or {}
            n_centers = ((dbg.get("l2_step") or {}).get("n_centers")) or 0
        except Exception as e:  # noqa: BLE001
            answer, cites, level = f"ERR {e}", [], "err"
            n_centers = -1
        u = llm.usage_stats()
        score = external_judge(q, answer, cites)
        recs.append({"qid": qid, "grp": it["grp"],
                     "prevA": it["A"]["pass"], "prevB": it["B"]["pass"],
                     "score": score, "pass": score >= PASS, "level": level,
                     "calls": u["calls"], "prompt": u["prompt_tokens"],
                     "n_centers": n_centers})
        print(f"[{i}/{len(items)}] {qid[:10]} mc={score}({level},{u['calls']}c,{n_centers}c) "
              f"prev(A={int(it['A']['pass'])},B={int(it['B']['pass'])})", flush=True)
        if i % 10 == 0 or i == len(items):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    ps = sum(1 for r in recs if r["pass"])
    pt = sum(r["prompt"] for r in recs) / len(recs)
    ca = sum(r["calls"] for r in recs) / len(recs)
    print("=" * 60)
    print(f"multicent(多候选圆心) pass {ps}/{len(recs)}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}")
    print(f"对照: base(现状)=40  strict(严判够)=44  B(直L3)=44")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
