"""深水对比 runner：对 deep_set 逐题跑 B0/B1/B2 + glm 裁判，记录 schema 与 run_compare 对齐。

用法：uv run python qa/compare/_run_deep.py --column B0|B1|B2 --out qa/compare/deep_<ts>_B0.json
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
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
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


def judge(q, answer, cites_text):
    gold, ev = gold_answer(q)
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（该题无答案，应诚实说明）",
                              evidence=(ev or ""), answer=(answer or "")[:1500],
                              cites=cites_text or "（无引用）")
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0, \
            str(raw.get("reason", ""))[:160] if isinstance(raw, dict) else ""
    except Exception as e:  # noqa: BLE001
        return 0, f"judge err {e}"


def full_text(paper):
    parts = []
    if paper.get("title"):
        parts.append("TITLE: " + str(paper["title"]))
    if paper.get("abstract"):
        parts.append("ABSTRACT: " + str(paper["abstract"]))
    for sec in paper.get("full_text") or []:
        paras = [" ".join(str(p).split()) for p in (sec.get("paragraphs") or []) if str(p).strip()]
        if not paras:
            continue
        parts.append("=== " + str(sec.get("section_name") or "").strip() + " ===")
        parts.append("\n".join(paras))
    return "\n\n".join(parts)


def qtype_of(q):
    for a in q.get("answers") or []:
        inner = a.get("answer") or {}
        if inner.get("unanswerable"):
            continue
        if inner.get("free_form_answer"):
            return "free_form"
        if inner.get("yes_no") is not None:
            return "yes_no"
        if inner.get("extractive_spans"):
            return "extractive"
    return "unanswerable"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True, choices=["B0", "B1", "B2"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    data = json.load(open(ROOT / "qa/compare/deep_set.json", encoding="utf-8"))
    papers = load_papers()
    # qid→q（含 pid）索引
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    llm.reset_usage()
    recs = []
    err = 0
    for i, it in enumerate(data["items"], 1):
        pid, q = qmap[it["qid"]]
        pdf = f"qasper_{pid}.qpdf"
        llm.reset_usage()
        t0 = time.time()
        gold, _ = gold_answer(q)
        try:
            if args.column == "B2":
                from paperpilot.graph import ask as graph_ask
                r = graph_ask(q.get("question", ""), pdf)
                answer = (r.get("answer") or "").strip()
                cites = list(r.get("cites") or [])
                dbg = r.get("debug") or {}
                level = (dbg.get("answer") or {}).get("level", "")
                route = list(r.get("route") or [])
            elif args.column == "B1":
                from paperpilot.agents.embedder import ChunkIndex
                hits = ChunkIndex(pdf).search(q.get("question", ""), top_k=12)
                if not hits:
                    ctx = "（未检索到相关段落）"
                else:
                    def _sec(h):
                        tp = list(h.get("title_path") or [])
                        return tp[-1] if tp else "?"
                    ctx = "\n\n".join("[%d] (sec: %s)\n%s" % (j, _sec(h), h.get("text", ""))
                                      for j, h in enumerate(hits, 1))
                answer = llm.chat_text("你是严谨的论文问答助手，只依据检索片段作答，材料不足则说明，不编造。",
                                       ctx + "\n\n问题：" + q.get("question", ""), temperature=0.0)
                cites, level, route = [], "", []
            else:  # B0
                full = full_text(papers[pid])
                answer = llm.chat_text("你是严谨的论文问答助手，只依据全文作答，材料不足则说明，不编造。",
                                       "【论文全文】\n%s\n\n【问题】\n%s" % (full, q.get("question", "")),
                                       temperature=0.0)
                cites, level, route = [], "", []
        except Exception as e:  # noqa: BLE001
            u = llm.usage_stats()
            recs.append({"column": args.column, "pid": pid, "qid": it["qid"],
                         "qtype": qtype_of(q), "unanswerable": not gold,
                         "question": q.get("question", ""), "answer": "", "level": "error",
                         "route": [], "score": 0, "reason": f"问答异常: {e}"[:160],
                         "status": "error", "time_s": round(time.time() - t0, 1),
                         "time_s_ans": 0, "gen": u, "judge": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                         "cites_n": 0})
            err += 1
            print(f"  [err] {it['qid'][:10]} {e}", flush=True)
            continue
        u_ans = llm.usage_stats()
        t_ans = time.time() - t0
        cite_text = "\n".join(f"[{j+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                              for j, c in enumerate(cites[:5]))
        score, reason = judge(q, answer, cite_text)
        jdg = llm.usage_stats()
        jdg = {k: jdg[k] - u_ans[k] for k in ("calls", "prompt_tokens", "completion_tokens")}
        status = "pass" if score >= PASS else "fail"
        recs.append({"column": args.column, "pid": pid, "qid": it["qid"],
                     "qtype": qtype_of(q), "unanswerable": not gold,
                     "question": q.get("question", ""), "answer": answer[:600], "level": level,
                     "route": route, "score": score, "reason": reason[:160], "status": status,
                     "time_s": round(time.time() - t0, 1), "time_s_ans": round(t_ans, 1),
                     "gen": u_ans, "judge": jdg, "cites_n": len(cites)})
        print(f"  [{i}/30] {it['qid'][:10]} {qtype_of(q):10} score={score} {status} "
              f"level={level} t={recs[-1]['time_s']}s", flush=True)
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[完成] {args.column} {len(recs)} 条, error={err} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
