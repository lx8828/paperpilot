"""修复回路考场：抽 10 道 QASPER 历史 fail 题（v3regress V 臂 fail 且有 gold evidence），
GATE on 跑一遍，external judge 判 pass —— 看 gate 检出哪些、修复救回哪些。

基线：这些题历史 V pass=0（同裁判口径）。能触发 repaired/fallback/翻绿都是新信息。
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

os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.qasper_source import gold_answer, load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/ab_gate_fail_result.json")
N = 10

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


def build_fail_pool() -> list[tuple[str, str, dict]]:
    reg = json.load(open("qa/recall/ab_v3regress_result.json", encoding="utf-8"))
    papers = load_papers()
    qmap: dict[str, tuple[str, dict]] = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qid = str(q.get("question_id") or "")
            if qid:
                qmap[qid] = (pid, q)
    out = []
    seen_pid: set[str] = set()
    for r in reg:
        if (r.get("V") or {}).get("pass"):
            continue
        qid, pid = r["qid"], r.get("pid", "")
        if pid in seen_pid:
            continue  # 每篇只抽 1 道，避免论文聚集
        if qid not in qmap:
            continue
        q = qmap[qid][1]
        gold, ev = gold_answer(q)
        if not gold:
            continue  # unanswerable/noise 不考修复
        seen_pid.add(pid)
        out.append((qid, pid, q))
        if len(out) >= N:
            break
    return out


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    pool = build_fail_pool()
    print(f"fail 池（V fail + 有 gold evidence + 每篇 1 道）n={len(pool)}")
    if len(pool) < N:
        print("不足 10 道——用可得的", len(pool))
    recs = []
    t0 = time.time()
    for i, (qid, pid, q) in enumerate(pool, 1):
        qq = q.get("question", "")
        pdf = f"qasper_{pid}.qpdf"
        llm.reset_usage()
        try:
            r = graph_ask(qq, pdf)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {qid[:10]} ERR {e}", flush=True)
            recs.append({"qid": qid, "err": str(e)[:120]}); continue
        u = llm.usage_stats()
        v = r.get("validator") or {}
        llm.reset_usage()
        score = ext_judge(q, r.get("answer") or "", list(r.get("cites") or []))
        recs.append({
            "qid": qid, "pid": pid, "grp": "hard",  # 保留字段
            "action": v.get("action", "pass"),
            "issues": [(i.get("sev"), i.get("type")) for i in (v.get("issues") or [])][:5],
            "score": score, "pass": score >= 4,
            "level": ((r.get("debug") or {}).get("answer") or {}).get("level", ""),
            "calls": u.get("calls"), "prompt": u.get("prompt_tokens"),
            "q": qq[:90], "answer": (r.get("answer") or "")[:160],
        })
        print(f"[{i}/{len(pool)}] {qid[:10]} act={v.get('action')} score={score} "
              f"issues={[x[1] for x in recs[-1]['issues']]}", flush=True)
        OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 60)
    from collections import Counter
    print("action:", dict(Counter(r.get("action") for r in recs)))
    print(f"GATE on 后 judge pass: {sum(1 for r in recs if r.get('pass'))}/{len(recs)}  "
          f"(历史 V fail 基线 0/{len(pool)})")
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t={time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
