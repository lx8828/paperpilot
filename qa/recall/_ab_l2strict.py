"""实验：judge_l2_strict（严判够，防首窗假阳性放行）on L2 目标题群 67 题。

样本 = ab_l2target_result.json 的 l2_target=True 题。
单臂 strict：build 完整漏斗图但 judge_l2 换成 judge_l2_strict，端到端 + 同外部裁判。
对照历史稳定值：base(现状) pass 40 / B(直L3) pass 44（两次 run 均复现 base=40）。
产出 pass/calls/prompt + 逐题，分析净亏 10 题是否被救回、双过 34 题是否误伤。
"""
from __future__ import annotations
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from paperpilot.agents.nodes import (answer_unknown, expand_l2, generate_answer,  # noqa: E402
                                     judge_l0, judge_l1, judge_l3,
                                     report_l0, retrieve_claims, search_l3)
from paperpilot.agents.nodes.judge import judge_l2_strict  # noqa: E402
from paperpilot.agents.state import QAState  # noqa: E402
from paperpilot.graph.qa_graph import (route_after_expand,  # noqa: E402
                                       route_after_judge0, route_after_judge1,
                                       route_after_judge2, route_after_judge3)
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_l2strict_result.json")


def build_full():
    g = StateGraph(QAState)
    for name, fn in [("report_l0", report_l0), ("judge_l0", judge_l0),
                     ("retrieve_claims", retrieve_claims), ("judge_l1", judge_l1),
                     ("expand_l2", expand_l2), ("judge_l2", judge_l2_strict),
                     ("search_l3", search_l3), ("judge_l3", judge_l3),
                     ("answer", generate_answer), ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "report_l0")
    g.add_edge("report_l0", "judge_l0")
    g.add_conditional_edges("judge_l0", route_after_judge0,
                            {"answer": "answer", "retrieve_claims": "retrieve_claims"})
    g.add_edge("retrieve_claims", "judge_l1")
    g.add_conditional_edges("judge_l1", route_after_judge1,
                            {"answer": "answer", "expand_l2": "expand_l2",
                             "search_l3": "search_l3"})
    g.add_conditional_edges("expand_l2", route_after_expand,
                            {"judge_l2": "judge_l2", "search_l3": "search_l3"})
    g.add_conditional_edges("judge_l2", route_after_judge2,
                            {"answer": "answer", "expand_l2": "expand_l2"})
    g.add_edge("search_l3", "judge_l3")
    g.add_conditional_edges("judge_l3", route_after_judge3,
                            {"answer": "answer", "answer_unknown": "answer_unknown"})
    g.add_edge("answer", END)
    g.add_edge("answer_unknown", END)
    return g.compile()


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
    app = build_full()
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
        st = {"question": q.get("question", ""), "pdf": f"qasper_{pid}.qpdf"}
        llm.reset_usage()
        try:
            out = app.invoke(st)
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            j2 = dbg.get("judge_l2") or {}
            strict_judge = j2.get("judge") == "strict"
        except Exception as e:  # noqa: BLE001
            answer, cites, level = f"ERR {e}", [], "err"
            strict_judge = False
        u = llm.usage_stats()
        score = external_judge(q, answer, cites)
        recs.append({"qid": qid, "grp": it["grp"],
                     "prevA": it["A"]["pass"], "prevB": it["B"]["pass"],
                     "score": score, "pass": score >= PASS, "level": level,
                     "calls": u["calls"], "prompt": u["prompt_tokens"],
                     "strict_judge": strict_judge})
        print(f"[{i}/{len(items)}] {qid[:10]} strict={score}({level},{u['calls']}c) "
              f"prev(A={int(it['A']['pass'])},B={int(it['B']['pass'])})", flush=True)
        if i % 10 == 0 or i == len(items):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    ps = sum(1 for r in recs if r["pass"])
    pt = sum(r["prompt"] for r in recs) / len(recs)
    ca = sum(r["calls"] for r in recs) / len(recs)
    saved = sum(1 for r in recs if not r["prevA"] and r["prevB"] and r["pass"])
    hurt = sum(1 for r in recs if r["prevA"] and not r["pass"])
    print("=" * 60)
    print(f"strict(严判够) pass {ps}/{len(recs)}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}")
    print(f"对照: base(现状)=40  B(直L3)=44  | 净亏题救回(曾A挂B过且现在过): {saved} "
          f"| 曾过现在挂(误伤): {hurt}")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
