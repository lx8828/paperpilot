"""单臂回归：当前默认图（v3 Router 包装 + Generator SYSTEM 准则14 硬引用，gate 默认关）
在 ab_v3base 同 100 题单上跑，对照历史 V(v3 纯两级/老 SYSTEM) pass=64。

目的：验证"Router 等价接入 + 准则14"这两个本轮改动是否伤 v3 基线 pass。
口径：外部裁判 prompt 与 _ab_v3base 完全一致；level/route 取自 debug。
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask  # noqa: E402  当前默认：qa_graph_v3(Router) + gate(默认关)
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PASS = 4
OUT = Path("qa/recall/ab_v3c14_result.json")

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
            out = ask(qq, pdf)
            answer = (out.get("answer") or "").strip()
            cites = list(out.get("cites") or [])
            dbg = out.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            route = ",".join(out.get("route") or [])
        except Exception as e:  # noqa: BLE001
            answer, cites, level, route, err = f"ERR {e}", [], "err", "", f"{type(e).__name__}: {e}"
        ug = llm.usage_stats()
        llm.reset_usage()
        sc = external_judge(q, answer, cites)
        recs.append({"qid": qid, "pid": pid, "grp": grp, "oldV_pass": oldv,
                     "pass": sc >= PASS, "score": sc, "level": level, "route": route,
                     "err": err, "prompt": ug["prompt_tokens"], "calls": ug["calls"]})
        flag = "same" if oldv == (sc >= PASS) else ("GAIN" if sc >= PASS else "LOSS")
        print(f"[{i}/{len(cases)}] {qid[:10]} oldV={'P' if oldv else 'f'} "
              f"new={'P' if sc >= PASS else 'f'}({sc},{level},{route}) {flag}", flush=True)
        if i % 10 == 0 or i == len(cases):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s  n={len(recs)}", flush=True)

    # 汇总
    def stat(sub):
        new = sum(1 for r in sub if r["pass"])
        old = sum(1 for r in sub if r["oldV_pass"])
        return new, old, len(sub)
    print("=" * 66)
    for name, key in [("all", lambda r: True), ("hard", lambda r: r["grp"] == "hard"),
                      ("normal", lambda r: r["grp"] == "normal")]:
        sub = [r for r in recs if key(r)]
        if not sub:
            continue
        new, old, n = stat(sub)
        gain = sum(1 for r in sub if r["pass"] and not r["oldV_pass"])
        loss = sum(1 for r in sub if not r["pass"] and r["oldV_pass"])
        print(f"{name:6}: new pass {new}/{n}  oldV {old}/{n}  Δ{new-old:+d}  "
              f"(gain {gain} / loss {loss})  prompt/题 {sum(r['prompt'] for r in sub)/n:.0f}")
    if recs and all(not r["err"] for r in recs):
        pass
    errs = [r for r in recs if r["err"]]
    if errs:
        print("ERR:", [(r["qid"][:10], r["err"][:80]) for r in errs])
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s  → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
