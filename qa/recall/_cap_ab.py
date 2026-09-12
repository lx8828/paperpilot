"""节级去重 cap=1 端到端 A/B：在 ab_v3base 同 100 题单上单臂跑 cap=1，
对照历史 V(v3) pass=64（无 cap，同裁判同题）。env 开关默认关 → 本脚本开启跑 cap 臂。

口径同 _v3c14_regress.py（同 100 题 + 同 external judge）。gate 默认关（只测检索变化）。
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

os.environ["PAPERPILOT_RETRIEVE_SECTION_CAP"] = "1"  # cap 臂
from paperpilot.graph import ask as graph_ask  # noqa: E402  当前默认 v3 + gate 默认关
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/_cap_result.json")

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
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass

    base = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))
    papers = load_papers()
    pmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                pmap[qid] = (pid, q)
    cases = [(r["qid"], r["grp"], bool(r["V"]["pass"])) for r in base]
    if args.limit:
        cases = cases[: args.limit]
    recs = json.loads(OUT.read_text(encoding="utf-8")) if args.resume and OUT.exists() else []
    done = {r["qid"] for r in recs}
    t0 = time.time()
    for i, (qid, grp, oldv) in enumerate(cases, 1):
        if qid in done:
            continue
        pid, q = pmap[qid]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"
        llm.reset_usage()
        err = ""
        try:
            out = graph_ask(qq, pdf)
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            n_l3 = len((dbg.get("answer") or {}).get("facts") or [])
        except Exception as e:  # noqa: BLE001
            answer, cites, level, err = f"ERR {e}", [], "err", f"{type(e).__name__}: {e}"
        ug = llm.usage_stats()
        llm.reset_usage()
        sc = external_judge(q, answer, cites)
        recs.append({"qid": qid, "pid": pid, "grp": grp, "baseV_pass": oldv,
                     "pass": sc >= PASS, "score": sc, "level": level,
                     "err": err, "prompt": ug["prompt_tokens"], "calls": ug["calls"]})
        flag = "same" if oldv == (sc >= PASS) else ("GAIN" if sc >= PASS else "LOSS")
        print(f"[{i}/{len(cases)}] {qid[:10]} V={'P' if oldv else 'f'} "
              f"cap={'P' if sc >= PASS else 'f'}({sc},{level}) {flag}", flush=True)
        if i % 10 == 0 or i == len(cases):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s  n={len(recs)}", flush=True)

    # 汇总 vs baseV
    d = [r for r in recs if not r.get("err")]
    new = sum(1 for r in d if r["pass"]); old = sum(1 for r in d if r["baseV_pass"])
    gain = [r for r in d if r["pass"] and not r["baseV_pass"]]
    loss = [r for r in d if not r["pass"] and r["baseV_pass"]]
    print("=" * 66)
    print(f"cap=1 pass {new}/{len(d)}  vs  baseV {old}/{len(d)}  Δ{new-old:+d}  "
          f"(gain {len(gain)} / loss {len(loss)})")
    for r in sorted(gain, key=lambda x: -x["score"])[:6]:
        print(f"  GAIN {r['qid'][:10]} {r['grp']:6} {r['score']} {r['level']}")
    for r in sorted(loss, key=lambda x: x["score"])[:6]:
        print(f"  LOSS {r['qid'][:10]} {r['grp']:6} {r['score']} {r['level']}")
    for g in ("hard", "normal"):
        sub = [r for r in d if r["grp"] == g]
        if sub:
            nw = sum(1 for r in sub if r["pass"]); od = sum(1 for r in sub if r["baseV_pass"])
            print(f"{g:6}: cap {nw}/{len(sub)}  baseV {od}/{len(sub)}  Δ{nw-od:+d}")
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
