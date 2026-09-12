"""临时：数字归一修复后，重跑 ab_nol3j 中 gate 干预的 5 题（2 fallback + 3 repaired）。"""
from __future__ import annotations
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

os.environ["PAPERPILOT_V3_NOL3J"] = "1"
os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

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


def ext_judge(q, answer, cites):
    gold, ev = gold_answer(q)
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "（无答案，应诚实说明）",
                              evidence=(ev or ""), answer=(answer or "")[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    prev = json.load(open("qa/recall/ab_nol3j_result.json", encoding="utf-8"))
    targets = [r for r in prev if r.get("action") in ("repaired", "fallback")]
    qmap = {}
    for pid, p in load_papers().items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    print(f"回归 {len(targets)} 题（修复后 gate 行为变化）：")
    for r in targets:
        qid = r["qid"]
        pid, q = qmap[qid]
        rr = graph_ask(q.get("question", ""), f"qasper_{pid}.qpdf")
        v = rr.get("validator") or {}
        llm.reset_usage()
        score = ext_judge(q, rr.get("answer") or "", list(rr.get("cites") or []))
        old = f"{r.get('action')}({r.get('score')},oldV_pass={r.get('oldV_pass')})"
        print(f"  {qid[:10]} {q.get('question','')[:45]}  | 旧 {old}  →  新 "
              f"action={v.get('action')} score={score} pass={score>=4} issues={[(i['sev'],i['type']) for i in (v.get('issues') or [])][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
