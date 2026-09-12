"""A/B：补充复核是否有用（数字型题 16，同主图输出分两臂）。

  A = gate(supplement=None)           旧行为：被引证据缺数字、全篇某 chunk 有 → 静默放行
  B = gate(supplement=_sup)           新行为：同一信号 → 命中块喂回 Generator 复核一次

同一次 qa_graph_v3.ask 输出出发，仅 supplement 参数不同；对 A/B 答案各自外部裁判打分，
对比 scoreA/scoreB：B>A 说明复核修正了会漏过的错数；A==B 且 action 不同说明白多一次调用。
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

os.environ["PAPERPILOT_VALIDATOR_GATE"] = "0"  # graph.ask 外层 gate 关，本脚本手动分臂
from paperpilot.graph import qa_graph_v3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/_supp_ab_result.json")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-only", action="store_true", help="只跑数字型题")
    ap.add_argument("--skip-judge", action="store_true", help="只跑门不裁判（快）")
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    from paperpilot.components import repairer, validator
    from paperpilot.components.validator import is_numeric_question
    from paperpilot.agents.document_cache import ordered_chunks

    papers = load_papers()
    pmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                pmap[qid] = (pid, q)
    base = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))
    if args.num_only:
        cases = [r["qid"] for r in base
                 if is_numeric_question(pmap.get(r["qid"], ("", {}))[1].get("question", ""))]
        print(f"数字型题 {len(cases)}", flush=True)
    else:
        cases = [r["qid"] for r in base]
        print(f"全量 {len(cases)}", flush=True)
    if args.limit:
        cases = cases[: args.limit]

    recs = []
    t0 = time.time()
    for i, qid in enumerate(cases, 1):
        pid, q = pmap[qid]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"
        err = ""
        try:
            out = qa_graph_v3.ask(qq, pdf)  # 主图一次，A/B 共用
            raw_answer = (out.get("answer") or "").strip()
            raw_cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            n_ctx = ((dbg.get("answer") or {}).get("n_entries")) or None
            chunks = list(ordered_chunks(pdf))
        except Exception as e:  # noqa: BLE001
            recs.append({"qid": qid, "err": f"{type(e).__name__}: {e}"})
            continue
        # ── A 臂：旧行为（不补复核） ──
        gA = validator.gate(qq, raw_answer, raw_cites, supplement=None,
                            n_entries=n_ctx, extra_chunks=chunks)
        # ── B 臂：新行为（补复核） ──
        def _sup(q2, ans2, supplements, cites2):
            return repairer.supplement(q2, pdf, ans2, supplements, cites2)
        gB = validator.gate(qq, raw_answer, raw_cites, supplement=_sup,
                            n_entries=n_ctx, extra_chunks=chunks)
        ansA, ansB = gA["answer"], gB["answer"]
        # 语义一致再省一次 judge
        if args.skip_judge:
            scoreA = scoreB = 0
        elif ansA == ansB and gA["action"] == gB["action"]:
            sc = external_judge(q, ansA, list(gA.get("cites") or raw_cites))
            scoreA = scoreB = sc
        else:
            scoreA = external_judge(q, ansA, list(gA.get("cites") or raw_cites))
            scoreB = external_judge(q, ansB, list(gB.get("cites") or raw_cites))
        recs.append({
            "qid": qid, "grp": next((r["grp"] for r in base if r["qid"] == qid), ""),
            "baseV_pass": next((bool(r["V"]["pass"]) for r in base if r["qid"] == qid), None),
            "n_supp": len(gA.get("supplements") or []),
            "supp_nums": [s.get("num_text") for s in (gA.get("supplements") or [])][:4],
            "A": {"action": gA["action"], "score": scoreA},
            "B": {"action": gB["action"], "score": scoreB},
            "changed": ansA != ansB,
            "ansA": ansA[:200], "ansB": ansB[:200],
        })
        tag = ("IMPROVE" if scoreB > scoreA else "WORSE" if scoreB < scoreA else "same")
        print(f"[{i}/{len(cases)}] {qid[:10]} supp={recs[-1]['n_supp']} "
              f"A={gA['action']}({scoreA}) B={gB['action']}({scoreB}) {tag}", flush=True)
        if i % 5 == 0 or i == len(cases):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s n={len(recs)}", flush=True)

    # 汇总
    d = [r for r in recs if not r.get("err")]
    trig = [r for r in d if r["n_supp"]]
    changed = [r for r in trig if r["changed"]]
    improved = [r for r in trig if r["A"]["score"] < r["B"]["score"]]
    worse = [r for r in trig if r["A"]["score"] > r["B"]["score"]]
    print("=" * 66)
    print(f"数字题 {len(d)} | 触发 supplement {len(trig)} | 复核改动 {len(changed)}")
    if trig:
        avg = lambda key: sum(r[key] for r in trig) / len(trig)  # noqa: E731
        print(f"触发组: A均分 {avg('A') and (sum(r['A']['score'] for r in trig)/len(trig)):.2f} | "
              f"B均分 {sum(r['B']['score'] for r in trig)/len(trig):.2f} | "
              f"improve {len(improved)} / worse {len(worse)} / 持平 {len(trig)-len(improved)-len(worse)}")
    for r in changed:
        print(f"  [{r['qid'][:10]}] supp={r['supp_nums']} "
              f"A={r['A']['score']}({r['A']['action']}) B={r['B']['score']}({r['B']['action']})")
        print(f"     A: {r['ansA'][:150]}")
        print(f"     B: {r['ansB'][:150]}")
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
