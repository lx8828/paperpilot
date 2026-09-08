"""Step2：v3+L1(claims 分支) 单臂，与 Step1 已存的 A(现状60)/V0(纯两级64) 同 100 题对照。

样本与 ab_v3base_result.json 完全一致（同 seed 抽出的 100 题），只补跑 V1 臂。
V1 = build_qa_graph_v3_l1：L0 → claims 分支(judge_l1 够则 L1 直答) → L3。
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

from paperpilot.graph.qa_graph_v3 import build_qa_graph_v3_l1  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_v3l1_result.json")

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
    app1 = build_qa_graph_v3_l1()
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    prev = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))  # 同 100 题
    items = prev
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
        qq = q.get("question", "")
        llm.reset_usage()
        try:
            out = app1.invoke({"question": qq, "pdf": f"qasper_{pid}.qpdf"})
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
        except Exception as e:  # noqa: BLE001
            answer, cites, level = f"ERR {e}", [], "err"
        u = llm.usage_stats()
        score = external_judge(q, answer, cites)
        recs.append({"qid": qid, "grp": it["grp"],
                     "prevA": it["A"]["pass"], "prevV0": it["V"]["pass"],
                     "score": score, "pass": score >= PASS, "level": level,
                     "calls": u["calls"], "prompt": u["prompt_tokens"]})
        print(f"[{i}/{len(items)}] {qid[:10]} V1={score}({level},{u['calls']}c) "
              f"prev(A={int(it['A']['pass'])},V0={int(it['V']['pass'])})", flush=True)
        if i % 10 == 0 or i == len(items):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    ps = sum(1 for r in recs if r["pass"])
    pt = sum(r["prompt"] for r in recs) / len(recs)
    ca = sum(r["calls"] for r in recs) / len(recs)
    print("=" * 60)
    print(f"V1(v3+L1 claims 分支) pass {ps}/{len(recs)}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}")
    print(f"对照(同100题历史): A(现状v2)=60  V0(纯两级)=64")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
