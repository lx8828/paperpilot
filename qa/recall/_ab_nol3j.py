"""A/B：v3 现图(judge_l3→unknown) vs 新图(nol3j 试答 + GATE)。

样本 = ab_v3base 同 100 题（历史 V = 现图 pass 64）。B 臂跑：
  env V3_NOL3J=1 + VALIDATOR_GATE=1（删 L3 judge → L3 一律试答 → gate 验证/repair/兜底）。
同外部裁判判定 pass；记录 gate action/issues 与 LLM 成本。
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

os.environ["PAPERPILOT_V3_NOL3J"] = "1"
os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/ab_nol3j_result.json")
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
    base = json.load(open("qa/recall/ab_v3base_result.json", encoding="utf-8"))
    papers = load_papers()
    qmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                qmap[qid] = (pid, q)
    cases = [(r["qid"], r["grp"], bool(r["V"]["pass"])) for r in base]

    recs = []
    t0 = time.time()
    for i, (qid, grp, oldv) in enumerate(cases, 1):
        pid, q = qmap[qid]
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"
        llm.reset_usage()
        err = ""
        try:
            r = graph_ask(qq, pdf)
            dbg = r.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            v = r.get("validator") or {}
        except Exception as e:  # noqa: BLE001
            err, r, v, level = f"{type(e).__name__}: {e}", {}, {}, "err"
        ug = llm.usage_stats()
        llm.reset_usage()
        score = ext_judge(q, r.get("answer") or "", list(r.get("cites") or [])) if not err else 0
        recs.append({
            "qid": qid, "pid": pid, "grp": grp, "oldV_pass": oldv,
            "pass": score >= PASS, "score": score, "level": level,
            "action": v.get("action") if not err else "err",
            "issues": [(i.get("sev"), i.get("type")) for i in (v.get("issues") or [])][:4],
            "err": err, "calls": ug.get("calls"), "prompt": ug.get("prompt_tokens"),
        })
        flag = "same" if oldv == (score >= PASS) else ("GAIN" if score >= PASS else "LOSS")
        print(f"[{i}/{len(cases)}] {qid[:10]} oldV={'P' if oldv else 'f'} "
              f"new={'P' if score >= PASS else 'f'}({score},{level},{v.get('action')}) {flag}",
              flush=True)
        if i % 10 == 0 or i == len(cases):
            OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   checkpoint t={time.time()-t0:.0f}s", flush=True)

    def stat(sub):
        return (sum(1 for r in sub if r["pass"]),
                sum(1 for r in sub if r["oldV_pass"]), len(sub))
    print("=" * 66)
    for nm, key in [("all", lambda r: True), ("hard", lambda r: r["grp"] == "hard"),
                    ("normal", lambda r: r["grp"] == "normal")]:
        sub = [r for r in recs if key(r)]
        if not sub:
            continue
        new, old, n = stat(sub)
        g = sum(1 for r in sub if r["pass"] and not r["oldV_pass"])
        l = sum(1 for r in sub if not r["pass"] and r["oldV_pass"])
        print(f"{nm:6}: new {new}/{n}  oldV {old}/{n}  d{new-old:+d} (gain {g}/loss {l}) "
              f"prompt/q {sum(r['prompt'] for r in sub)/n:.0f}")
    from collections import Counter
    print("action:", dict(Counter(r["action"] for r in recs)))
    errs = [r for r in recs if r["err"]]
    print("err:", len(errs))
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
