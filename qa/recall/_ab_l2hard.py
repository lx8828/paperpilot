"""实验策略③：judge_l2 硬指标 A/B（base vs HARD）on L2 目标题群 67 题。

样本 = ab_l2target_result.json 里 l2_target=True 的题（judge_l1 不够+有 target，本该走 L2）。
同题同进程跑两模式：base（现状 judge_l2）vs hard（PAPERPILOT_JUDGE_L2_HARD=1，
LLM 判够后硬检查窗口是否命中问题数字/专名，全缺 → 强制不够 → 扩窗耗尽降 L3）。
同一外部裁判 glm-4-flash。记录每臂 calls/prompt + judge_l2 硬拦截次数。

对照基线（历史 ab_l2target_result）：A(扩窗)pass 40、B(直L3)pass 44。
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
OUT = Path("qa/recall/ab_l2hard_result.json")

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


def run_once(q, pdf, hard_on):
    os.environ["PAPERPILOT_JUDGE_L2_HARD"] = "1" if hard_on else "0"
    llm.reset_usage()
    try:
        out = graph_ask(q.get("question", ""), pdf)
        answer = (out.get("answer") or "").strip()
        cites = list(out.get("cites") or [])
        dbg = out.get("debug") or {}
        level = ((dbg.get("answer") or {}).get("level", "")) or ""
        j2 = dbg.get("judge_l2") or {}
        hard_n = 1 if j2.get("hard_override") else 0
    except Exception as e:  # noqa: BLE001
        answer, cites, level = f"ERR {e}", [], "err"
        hard_n = -1
    u = llm.usage_stats()
    score = external_judge(q, answer, cites)
    return {"score": score, "pass": score >= PASS, "level": level,
            "calls": u["calls"], "prompt": u["prompt_tokens"],
            "hard_override": hard_n}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))

    prev = json.load(open("qa/recall/ab_l2target_result.json", encoding="utf-8"))
    items = [r for r in prev if r["l2_target"]]
    if args.limit:
        items = items[: args.limit]
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)

    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    recs = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        qid = it["qid"]
        pid, q = qmap[qid]
        pdf = f"qasper_{pid}.qpdf"
        base = run_once(q, pdf, hard_on=False)
        hard = run_once(q, pdf, hard_on=True)
        recs.append({"qid": qid, "grp": it["grp"],
                     "prevA": it["A"]["pass"], "prevB": it["B"]["pass"],
                     "base": base, "hard": hard})
        print(f"[{i}/{len(items)}] {qid[:10]} base={base['score']}({base['level']},{base['calls']}c) "
              f"hard={hard['score']}({hard['level']},{hard['calls']}c) "
              f"override={hard['hard_override']}", flush=True)
        if i % 10 == 0 or i == len(items):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    def agg(runs, key):
        ps = sum(1 for r in runs if r[key]["pass"])
        pt = sum(r[key]["prompt"] for r in runs) / len(runs)
        ca = sum(r[key]["calls"] for r in runs) / len(runs)
        return ps, pt, ca

    for name, key in [("base(现状judge_l2)", "base"), ("hard(策略③硬指标)", "hard")]:
        ps, pt, ca = agg(recs, key)
        print(f"{name}: pass {ps}/{len(recs)}  prompt/题 {pt:.0f}  calls/题 {ca:.1f}")
    ov = sum(1 for r in recs if r["hard"]["hard_override"] == 1)
    print(f"hard 模式触发硬拦截的题: {ov}/{len(recs)}")
    print(f"对照基线(历史): A(扩窗) pass {sum(1 for r in recs if r['prevA'])} / "
          f"B(直L3) pass {sum(1 for r in recs if r['prevB'])}")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
