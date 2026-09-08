"""实验 L2-target：在"L2 目标题群"上对比 A(现状扩窗) vs B(砍L2直L3)。

方法论修正（2026-09-08）：
    之前用 recall hard/normal 测 L2 存废是错的——hard(端到端 fail/unknown) 本就该走
    全局 L3；L2 的服务对象是 "judge_l1 判不够但给了 target_sections（有局部方向）"的
    题（多为中等问题），L2 在局部扩窗解决以省下全文 L3 的 token。
本实验：
    样本 = 全量 QASPER 抽题（要求 out_views 有 report、gold 可定位），按
           final_lock fail/unknown(hard 倾向) 与其余(normal) 分层混抽；
    每臂只变"judge_l1 判不够后怎么办"：A=现状(进 expand_l2 扩窗) / B=noL2(直接 search_l3)，
    同 judge_l1 判定同题端到端 + 同一外部裁判(glm-4-flash)；
    记录每臂 debug.judge_l1(enough/target_sections) + route(是否真进 L2) + calls + tokens。
分析口径（跑完脚本即打印）：
    L2 目标题群 = A.debug.judge_l1.enough==False 且 target_sections 非空（图会进 L2）的题
    - 准度：A(扩窗) pass vs B(直L3) pass → L2 净赢/净亏
    - 成本：A vs B 的 prompt tokens → L2 减负是否成立（省 token）
"""
from __future__ import annotations
import io
import json
import random
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
OUT = Path("qa/recall/ab_l2target_result.json")


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


# ── 样本池构建：有 report + gold 可定位 ───────────────────────────────────
def build_pool():
    papers = load_papers()
    final_lock = json.load(open("qa/qasper_final_lock_20260907_183225.json", encoding="utf-8"))
    lock = {r["qid"]: r for r in final_lock}
    view = ROOT / "out_views"
    hard_qids = {r["qid"] for r in final_lock if r.get("new_status") in ("fail", "unknown_ok")}
    index: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        if not (view / f"qasper_{pid}.report.json").exists():
            continue
        for q in p.get("qas") or []:
            gold, ev = gold_answer(q)
            qid = str(q.get("question_id") or "")
            if gold and ev:
                index[qid] = (pid, q)
    return index, hard_qids, lock


def sample(index: dict, hard_qids: set, n_hard: int, n_normal: int, seed: int):
    hard_pool = [q for q in index if q in hard_qids]
    normal_pool = [q for q in index if q not in hard_qids]
    rng = random.Random(seed)
    rng.shuffle(hard_pool)
    rng.shuffle(normal_pool)
    per_pid: dict[str, int] = {}
    sel: list[tuple[str, str]] = []  # (qid, pid)
    cnt = {"hard": 0, "normal": 0}

    def take(qid):
        pid = index[qid][0]
        grp = "hard" if qid in hard_qids else "normal"
        if per_pid.get(pid, 0) >= 3:
            return False
        if (grp == "hard" and cnt["hard"] >= n_hard) or \
           (grp == "normal" and cnt["normal"] >= n_normal):
            return False
        per_pid[pid] = per_pid.get(pid, 0) + 1
        cnt[grp] += 1
        sel.append((qid, pid))
        return True

    for q in hard_pool + normal_pool:
        if cnt["hard"] >= n_hard and cnt["normal"] >= n_normal:
            break
        take(q)
    return sel


def summarize(recs):
    def grp(recs, pred, name):
        sub = [r for r in recs if pred(r)]
        if not sub:
            print(f"{name}: n=0")
            return
        a_pass = sum(1 for r in sub if r["A"]["pass"])
        b_pass = sum(1 for r in sub if r["B"]["pass"])
        l2w = sum(1 for r in sub if r["A"]["pass"] and not r["B"]["pass"])
        l2l = sum(1 for r in sub if not r["A"]["pass"] and r["B"]["pass"])
        at = sum(r["A"]["prompt"] for r in sub) / len(sub)
        bt = sum(r["B"]["prompt"] for r in sub) / len(sub)
        ac = sum(r["A"]["calls"] for r in sub) / len(sub)
        bc = sum(r["B"]["calls"] for r in sub) / len(sub)
        print(f"{name}: n={len(sub)}  A(扩窗) pass {a_pass}  B(直L3) pass {b_pass}  "
              f"| L2净赢 {l2w} 净亏 {l2l} | prompt-tok/题 A={at:.0f} B={bt:.0f} "
              f"| calls/题 A={ac:.1f} B={bc:.1f}")

    print("=" * 70)
    print(f"总样本 {len(recs)}（hard {sum(1 for r in recs if r['grp']=='hard')} / "
          f"normal {sum(1 for r in recs if r['grp']=='normal')}）")
    grp(recs, lambda r: True, "全体")
    grp(recs, lambda r: r["l2_target"], "L2目标题群(judge_l1不够+有target)")
    grp(recs, lambda r: r["l2_used"], "真走L2的题(route含L2)")
    grp(recs, lambda r: not r["l2_target"] and r["grp"] == "normal", "normal非L2目标(对照)")
    grp(recs, lambda r: r["grp"] == "hard", "hard(本应直L3的对照)")
    # judge_l1 A/B 不一致率（噪声监测）
    diff = sum(1 for r in recs if r["A"]["j1_enough"] != r["B"]["j1_enough"])
    print(f"judge_l1 enough A/B 不一致: {diff}/{len(recs)}")
    print("=" * 70)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hard", type=int, default=35)
    ap.add_argument("--n-normal", type=int, default=45)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--limit", type=int, default=0, help="冒烟只跑前 N 题")
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    noL2 = build_noL2()
    papers = load_papers()

    index, hard_qids, _ = build_pool()
    sel = sample(index, hard_qids, args.n_hard, args.n_normal, args.seed)
    if args.limit:
        sel = sel[: args.limit]
    print(f"样本 {len(sel)} 题（hard 源 {sum(1 for s in sel if s[0] in hard_qids)}）", flush=True)

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs: list[dict] = []
    t0 = time.time()
    for i, (qid, pid) in enumerate(sel, 1):
        q = index[qid][1]
        pdf = f"qasper_{pid}.qpdf"
        st = {"question": q.get("question", ""), "pdf": pdf}

        def run(graph):
            llm.reset_usage()
            try:
                out = graph_ask(q.get("question", ""), pdf) if graph == "A" else noL2.invoke(st)
                answer = (out.get("answer") or "").strip()
                cites = list(out.get("cites") or [])
                dbg = out.get("debug") or {}
                level = ((dbg.get("answer") or {}).get("level", "")) or ""
                j1 = dbg.get("judge_l1") or {}
                route = list(out.get("route") or [])
            except Exception as e:  # noqa: BLE001
                answer, cites, level = f"ERR {e}", [], "err"
                j1, route = {}, []
            u = llm.usage_stats()
            return {"answer": answer, "cites": cites, "level": level,
                    "j1_enough": bool(j1.get("enough")), "j1_targets": list(j1.get("target_sections") or []),
                    "route": route, "calls": u["calls"], "prompt": u["prompt_tokens"],
                    "comp": u["completion_tokens"]}

        ra = run("A")
        llm.reset_usage()
        sa = external_judge(q, ra["answer"], ra["cites"])
        ua = llm.usage_stats()
        rb = run("B")
        llm.reset_usage()
        sb = external_judge(q, rb["answer"], rb["cites"])
        ub = llm.usage_stats()

        rec = {
            "qid": qid, "pid": pid, "grp": "hard" if qid in hard_qids else "normal",
            "l2_target": (not ra["j1_enough"]) and bool(ra["j1_targets"]),
            "l2_used": "L2" in ra["route"],
            "A": {"score": sa, "pass": sa >= PASS, "level": ra["level"], "calls": ra["calls"],
                  "prompt": ra["prompt"] + ua["prompt_tokens"], "j1_enough": ra["j1_enough"],
                  "j1_targets": ra["j1_targets"]},
            "B": {"score": sb, "pass": sb >= PASS, "level": rb["level"], "calls": rb["calls"],
                  "prompt": rb["prompt"] + ub["prompt_tokens"], "j1_enough": rb["j1_enough"],
                  "j1_targets": rb["j1_targets"]},
        }
        recs.append(rec)
        print(f"[{i}/{len(sel)}] {qid[:10]} {rec['grp'][:1]} A={sa}({ra['level']},{ra['calls']}c) "
              f"B={sb}({rb['level']},{rb['calls']}c)  l2_target={rec['l2_target']} l2_used={rec['l2_used']}",
              flush=True)
        if i % 10 == 0 or i == len(sel):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    summarize(recs)
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
