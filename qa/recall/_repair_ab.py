"""修复回路 A/B：10 道历史 V fail 题，GATE off vs on(REPAIR_MID=1)，同裁判双打分。

  off = 纯主图原答案（无 gate）
  on  = GATE=1 + PAPERPILOT_VALIDATOR_REPAIR_MID=1（mid 也修 → Self-Refine 路径可触发）
  指标：judge pass/score off vs on；on action 分布；修复前 issues 类型分布；
        refine 类 issue(off_topic/contradiction/vague) 是否真触发。
  净增益 = on pass − off pass（同题同裁判）。
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

os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"      # on 臂（off 臂跑前临时置 0）
os.environ["PAPERPILOT_VALIDATOR_REPAIR_MID"] = "1"  # 允许 mid 触发 repair（测 Self-Refine）
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_repair_ab_result.json")

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


def judge(q, answer, cites):
    gold, ev = gold_answer(q)
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {c.get('evidence','')}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=q.get("question", ""), gold=gold or "",
                              evidence=(ev or ""), answer=(answer or "")[:1500], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:
        return 0


def run_off(qq, pdf):
    """GATE=0：纯主图原答案。"""
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = "0"
    try:
        r = graph_ask(qq, pdf)
        return (r.get("answer") or "").strip(), list(r.get("cites") or []), "", ""
    except Exception as e:  # noqa: BLE001
        return "", [], "", f"{type(e).__name__}: {e}"


def run_on(qq, pdf):
    """GATE=1 + REPAIR_MID=1。"""
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
    try:
        r = graph_ask(qq, pdf)
    except Exception as e:  # noqa: BLE001
        return "", [], "err", f"{type(e).__name__}: {e}"
    v = r.get("validator") or {}
    issues = [(i.get("sev"), i.get("type")) for i in (v.get("issues") or [])]
    return (r.get("answer") or "").strip(), list(r.get("cites") or []), \
        v.get("action", "pass"), issues


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    prev = json.load(open("qa/recall/ab_gate_fail_result.json", encoding="utf-8"))
    cases = [r for r in prev if r.get("qid") and r.get("pid")]
    if args.limit:
        cases = cases[: args.limit]
    papers = load_papers()
    pmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                pmap[qid] = (pid, q)
    print(f"对照 {len(cases)} 题（历史 V fail）", flush=True)
    recs = []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        qid = c["qid"]
        pid = c["pid"]
        q = pmap.get(qid, (None, {}))[1]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"
        # off 臂
        ans_off, cites_off, _, err_off = run_off(qq, pdf)
        score_off = judge(q, ans_off, cites_off) if ans_off else 0
        # on 臂
        ans_on, cites_on, action, issues = run_on(qq, pdf)
        score_on = judge(q, ans_on, cites_on) if ans_on else 0
        recs.append({
            "qid": qid, "pid": pid,
            "off": {"score": score_off, "pass": score_off >= 4},
            "on": {"score": score_on, "pass": score_on >= 4,
                   "action": action, "issues": issues},
            "changed": ans_off != ans_on,
            "ans_off": ans_off[:120], "ans_on": ans_on[:120],
        })
        d = score_on - score_off
        tag = "GAIN" if score_on > score_off else ("LOSS" if score_on < score_off else "same")
        print(f"[{i}/{len(cases)}] {qid[:10]} off={score_off} on={score_on}({action}) "
              f"Δ{d:+d} {tag} issues={[x[1] for x in issues][:4]}", flush=True)
        if i % 5 == 0 or i == len(cases):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)
    # 汇总
    from collections import Counter
    off_pass = sum(1 for r in recs if r["off"]["pass"])
    on_pass = sum(1 for r in recs if r["on"]["pass"])
    print("=" * 66)
    print(f"judge pass: off {off_pass}/{len(recs)}  on {on_pass}/{len(recs)}  "
          f"Δ{on_pass-off_pass:+d}")
    acts = Counter(r["on"]["action"] for r in recs)
    print("on action:", dict(acts))
    avg = lambda k: sum(r[k]["score"] for r in recs) / len(recs)  # noqa: E731
    print(f"平均分: off {avg('off'):.2f}  on {avg('on'):.2f}")
    gen = [r for r in recs if any(t in ("off_topic", "contradiction", "vague")
                                  for _, t in r["on"]["issues"])]
    print(f"含 generation 型 issue(→Self-Refine 候选): {len(gen)}/{len(recs)}")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
