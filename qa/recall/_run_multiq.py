"""多方法全局对比题 × V0(纯两级 L0→L3)：看全局检索能不能处理这类"跨对象对比/列举"。

若 pass 明显低于 V0 全链水平(64%)→ 支持做多维分解分支；同时存 answer 供失败模式分析。
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

from paperpilot.graph.qa_graph_v3 import build_qa_graph_v3  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_multiq_result.json")

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
    app0 = build_qa_graph_v3()  # V0 纯两级
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    items = json.load(open("qa/recall/multiq_set.json", encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        qid, pid = it["qid"], it["pid"]
        q = qmap[qid][1]
        qq = q.get("question", "")
        llm.reset_usage()
        try:
            out = app0.invoke({"question": qq, "pdf": f"qasper_{pid}.qpdf"})
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
        except Exception as e:  # noqa: BLE001
            answer, cites, level = f"ERR {e}", [], "err"
        u = llm.usage_stats()
        score = external_judge(q, answer, cites)
        recs.append({"qid": qid, "pid": pid, "question": qq, "gold": gold_answer(q)[0],
                     "score": score, "pass": score >= PASS, "level": level,
                     "answer": answer[:900], "calls": u["calls"],
                     "prompt": u["prompt_tokens"]})
        print(f"[{i}/{len(items)}] {qid[:10]} V0={score}({level},{u['calls']}c)", flush=True)
        if i % 10 == 0 or i == len(items):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    ps = sum(1 for r in recs if r["pass"])
    pt = sum(r["prompt"] for r in recs) / len(recs)
    print("=" * 60)
    print(f"多方法对比题 V0(纯两级): pass {ps}/{len(recs)} ({ps/len(recs):.0%})  "
          f"prompt/题 {pt:.0f}")
    print("参照: V0 全链 64/100 (64%)；这类题如果明显更低 → 支持做多维分支")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
