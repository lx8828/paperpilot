"""实验1 A/B：砍掉 L2（L1 不够→直接 L3）vs 现状完整漏斗。

两臂同题端到端 + 同一外部裁判(glm)：A=graph.ask(现状)；B=noL2 变体图。
样本：recall hard 25 + normal 25。
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

from paperpilot.agents.nodes import (answer_unknown, generate_answer,  # noqa: E402
                                     judge_l0, judge_l1, judge_l3,
                                     report_l0, retrieve_claims, search_l3)
from paperpilot.agents.state import QAState  # noqa: E402
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.graph.qa_graph import (route_after_judge0,  # noqa: E402
                                       route_after_judge3)
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4


def route_j1_noL2(state):
    return "answer" if (state.get("verdict") or {}).get("enough") else "search_l3"


def build_noL2():
    g = StateGraph(QAState)
    for name, fn in [("report_l0", report_l0), ("judge_l0", judge_l0),
                     ("retrieve_claims", retrieve_claims), ("judge_l1", judge_l1),
                     ("search_l3", search_l3), ("judge_l3", judge_l3),
                     ("answer", generate_answer), ("answer_unknown", answer_unknown)]:
        g.add_node(name, fn)
    g.add_edge(START, "report_l0")
    g.add_edge("report_l0", "judge_l0")
    g.add_conditional_edges("judge_l0", route_after_judge0,
                            {"answer": "answer", "retrieve_claims": "retrieve_claims"})
    g.add_edge("retrieve_claims", "judge_l1")
    g.add_conditional_edges("judge_l1", route_j1_noL2,
                            {"answer": "answer", "search_l3": "search_l3"})
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
    ap.add_argument("--n-hard", type=int, default=25)
    ap.add_argument("--n-normal", type=int, default=25)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    noL2 = build_noL2()
    papers = load_papers()
    items = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))["items"]
    sel = [it for it in items if it["group"] == "hard"][: args.n_hard] + \
          [it for it in items if it["group"] == "normal"][: args.n_normal]
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    t0 = time.time()
    recs = []
    for i, it in enumerate(sel, 1):
        q = next(qq for qq in papers[it["pid"]].get("qas") or []
                 if str(qq.get("question_id") or "") == it["qid"])
        pdf = f"qasper_{it['pid']}.qpdf"
        st = {"question": q.get("question", ""), "pdf": pdf}
        llm.reset_usage()
        try:
            ra = graph_ask(q.get("question", ""), pdf)
            aa = (ra.get("answer") or "").strip()
            ca = list(ra.get("cites") or [])
            lva = ((ra.get("debug") or {}).get("answer") or {}).get("level", "")
            sca = external_judge(q, aa, ca)
        except Exception as e:  # noqa: BLE001
            aa, ca, lva, sca = f"ERR {e}", [], "err", 0
        ua = llm.usage_stats()
        llm.reset_usage()
        try:
            rb = noL2.invoke(st)
            ab = (rb.get("answer") or "").strip()
            cb = list(rb.get("cites") or [])
            lvb = ((rb.get("debug") or {}).get("answer") or {}).get("level", "")
            scb = external_judge(q, ab, cb)
        except Exception as e:  # noqa: BLE001
            ab, cb, lvb, scb = f"ERR {e}", [], "err", 0
        ub = llm.usage_stats()
        recs.append({"qid": it["qid"], "group": it["group"],
                     "A": {"score": sca, "pass": sca >= PASS, "level": lva, "calls": ua["calls"]},
                     "B": {"score": scb, "pass": scb >= PASS, "level": lvb, "calls": ub["calls"]}})
        print(f"[{i}] {it['qid'][:10]} A={sca}({lva}) B={scb}({lvb})", flush=True)
        if i % 10 == 0:
            print(f"   t={time.time()-t0:.0f}s", flush=True)
    Path("qa/recall/ab_noL2_result.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ("hard", "normal", "all"):
        grp = [r for r in recs if g == "all" or r["group"] == g]
        pa = sum(1 for r in grp if r["A"]["pass"])
        pb = sum(1 for r in grp if r["B"]["pass"])
        ca = round(sum(r["A"]["calls"] for r in grp) / max(len(grp), 1), 2)
        cb = round(sum(r["B"]["calls"] for r in grp) / max(len(grp), 1), 2)
        print(f"{g} n={len(grp)}: A(现状) pass {pa} | B(砍L2) pass {pb} | "
              f"Δ={pb-pa} | 调用/题 A={ca} B={cb}")
    print(f"t={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
